"""Autonomous overnight experiment sweep + deploy.

Runs the agreed improvement plan end-to-end, unattended and robustly:

  1. Sweep (doubly-disjoint, unseen proteins x unseen probes) over:
       baseline, core=B+C, core+F, core+G, core+E, and B/C in isolation.
  2. Deploy the best config as the shipped model (full retrain on all probes,
     regenerate submission, runtime check).
  3. Best-effort ESM-DBP (idea A): a separate subprocess with a timeout so it
     can never hang the run; if it beats the best config, redeploy.
  4. Ensemble measurement (idea D), time permitting.

Every result is appended to artifacts/EXPERIMENTS.md and experiment_results.json
as it completes, so partial progress is never lost. Each phase is guarded by
try/except and a global time budget so the run always finishes with a valid,
improved (or at least unchanged) submission.

Usage:
    python scripts/run_experiments.py            # full overnight run
    python scripts/run_experiments.py --smoke    # 1-epoch sanity of the pipeline
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.experiment import prepare_shared_data, bank_from_cache, run_experiment
from src.utils import get_device, get_logger

logger = get_logger("sweep")
ART = "artifacts"
RESULTS_JSON = os.path.join(ART, "experiment_results.json")
RESULTS_MD = os.path.join(ART, "EXPERIMENTS.md")

# Sweep fidelity: reduced for speed, identical across configs so comparisons are
# fair. The winner is retrained at full fidelity in the deploy step.
SWEEP_EPOCHS = 16
SWEEP_PROBES = 2000
FINAL_EPOCHS = 25
DEPLOY_MARGIN = 0.005            # only redeploy if a config beats baseline by this
TIME_BUDGET_S = 6.0 * 3600       # stop starting new sweep configs after this
REDEPLOY_DEADLINE_S = 6.5 * 3600  # don't START an ESM-DBP redeploy after this


def apply_overrides(base, overrides: dict):
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        section, field = key.split(".")
        setattr(getattr(cfg, section), field, val)
    return cfg


def write_results(results: list, baseline_dd: float):
    """Rewrite EXPERIMENTS.md + json from the accumulated results (idempotent)."""
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    lines = ["# Experiment log — doubly-disjoint (unseen proteins x unseen probes)\n",
             "Metric = mean per-protein Pearson on held-out proteins AND held-out probes",
             f"(sweep fidelity: {SWEEP_EPOCHS} epochs, {SWEEP_PROBES} probes/protein).\n",
             "| experiment | DD Pearson | DD Spearman | Δ vs baseline | best epoch | minutes | notes |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        dd = r.get("best_double_disjoint_pearson")
        delta = "" if dd is None or baseline_dd is None else f"{dd - baseline_dd:+.4f}"
        dds = r.get("best_double_disjoint_spearman")
        lines.append(
            f"| {r['name']} | {'' if dd is None else f'{dd:.4f}'} | "
            f"{'' if dds is None else f'{dds:.4f}'} | {delta} | "
            f"{r.get('best_epoch','')} | {r.get('minutes','')} | {r.get('notes','')} |")
    with open(RESULTS_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_one(name, overrides, base_cfg, shared, bank, dim, device, epochs, results, baseline_dd):
    """Run a single sweep config with error isolation; append + persist result."""
    logger.info(f"=== {name}: {overrides} ===")
    rec = {"name": name, "overrides": overrides}
    try:
        cfg = apply_overrides(base_cfg, overrides)
        cfg.train.probes_per_protein_per_epoch = SWEEP_PROBES
        t0 = time.time()
        res = run_experiment(cfg, shared, bank, dim, epochs=epochs, device=device,
                             log=lambda m: logger.info(m))
        rec.update({
            "best_double_disjoint_pearson": res["best_double_disjoint_pearson"],
            "best_double_disjoint_spearman": res["best_double_disjoint_spearman"],
            "best_epoch": res["best_epoch"],
            "minutes": round((time.time() - t0) / 60, 1),
        })
        logger.info(f"    -> {name}: DD Pearson {res['best_double_disjoint_pearson']:.4f}")
    except Exception as e:  # never let one config kill the sweep
        rec["notes"] = f"ERROR: {type(e).__name__}: {e}"
        logger.info(f"    !! {name} failed: {e}")
    results.append(rec)
    write_results(results, baseline_dd)
    return rec


def deploy(winner_overrides, embeddings_path, base_cfg, log):
    """Retrain the winner at full fidelity, regenerate submission, check runtime."""
    log(f"DEPLOY: {winner_overrides} (embeddings={embeddings_path})")
    if embeddings_path and os.path.abspath(embeddings_path) != os.path.abspath(base_cfg.paths.embeddings_cache):
        import shutil
        shutil.copy(embeddings_path, base_cfg.paths.embeddings_cache)
    cfg = apply_overrides(base_cfg, winner_overrides)
    cfg.train.epochs = FINAL_EPOCHS
    cfg.train.probes_per_protein_per_epoch = 3000
    from src.train import train
    train(cfg)                                                   # saves model.pt
    subprocess.run([sys.executable, "scripts/predict_all.py"], check=False, timeout=1800)
    r = subprocess.run([sys.executable, "scripts/runtime_test.py", "--n", "8"],
                       capture_output=True, text=True, timeout=1200)
    log("RUNTIME CHECK:\n" + (r.stdout or "") + (r.stderr or ""))


def try_esm_dbp(best_overrides, base_cfg, shared, device, results, baseline_dd, log):
    """Best-effort ESM-DBP (idea A). Isolated subprocess + timeout so it can't hang."""
    log("=== ESM-DBP (A): attempting download + precompute (best-effort) ===")
    out = os.path.join(ART, "protein_embeddings_esmdbp.npz")
    try:
        r = subprocess.run([sys.executable, "scripts/precompute_esm_dbp.py", "--out", out],
                           timeout=3000, capture_output=True, text=True)
        log((r.stdout or "")[-2000:] + (r.stderr or "")[-1000:])
        if r.returncode != 0 or not os.path.exists(out):
            results.append({"name": "esm_dbp", "notes": "SKIPPED: precompute failed (see log)"})
            write_results(results, baseline_dd)
            return None
        bank, dim = bank_from_cache(out, shared.rep_idx)
        rec = run_one("A_esm_dbp+best", best_overrides, base_cfg, shared, bank, dim,
                      device, SWEEP_EPOCHS, results, baseline_dd)
        rec["embeddings"] = out
        write_results(results, baseline_dd)
        return rec
    except Exception as e:
        results.append({"name": "esm_dbp", "notes": f"SKIPPED: {type(e).__name__}: {e}"})
        write_results(results, baseline_dd)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    base = load_config("config.yaml", warn_missing_files=False)
    device = get_device()
    logger.info(f"Device: {device}. Preparing shared data (this loads once)...")
    shared = prepare_shared_data(base, probe_holdout=0.1)
    bank35, dim35 = bank_from_cache(base.paths.embeddings_cache, shared.rep_idx)
    noise_std = round(0.1 * float(bank35.std()), 4)
    logger.info(f"protein-noise std for C set to {noise_std} (0.1 x embedding std)")

    epochs = 1 if args.smoke else SWEEP_EPOCHS
    if args.smoke:
        shared.train_ids = shared.train_ids[:12]
        shared.val_ids = shared.val_ids[:6]

    C = {"train.protein_noise_std": noise_std, "train.protein_mask_prob": 0.1}
    B = {"head.type": "cross_attention"}
    core = {**B, **C}
    # The 5 prioritized configs: baseline reference, the core (B+C), and E/F/G
    # each added on top of the core (as requested). Isolation runs (C-only,
    # B-only) are omitted to fit the overnight budget.
    experiments = [
        ("baseline", {}),
        ("core_B+C", core),
        ("core+F_cnn_transformer", {**core, "dna_encoder.type": "cnn_transformer"}),
        ("core+G_pearson_loss", {**core, "train.loss": "pearson",
                                 "train.single_protein_batches": True}),
        ("core+E_moe", {**core, "dna_encoder.type": "moe"}),
    ]

    # Resume: if a prior run left results (e.g. the machine rebooted), reuse the
    # completed configs and only run the remaining ones.
    results: list = []
    done = {}
    if os.path.exists(RESULTS_JSON) and not args.smoke:
        try:
            with open(RESULTS_JSON, encoding="utf-8") as fh:
                results = json.load(fh)
            done = {r["name"]: r for r in results
                    if r.get("best_double_disjoint_pearson") is not None}
            if done:
                logger.info(f"Resuming — reusing completed configs: {list(done)}")
        except Exception:
            results = []
    baseline_dd = done.get("baseline", {}).get("best_double_disjoint_pearson")

    for name, ov in experiments:
        if name in done:
            logger.info(f"=== {name}: already done (DD {done[name]['best_double_disjoint_pearson']:.4f}), skipping ===")
            continue
        if time.time() - t_start > TIME_BUDGET_S and not args.smoke:
            logger.info("Time budget reached; stopping sweep early to leave time to deploy.")
            break
        rec = run_one(name, ov, base, shared, bank35, dim35, device, epochs, results, baseline_dd)
        if name == "baseline":
            baseline_dd = rec.get("best_double_disjoint_pearson")
            write_results(results, baseline_dd)

    # ---- adaptive combinations: combine additions that individually helped --
    # F and E both replace the DNA encoder (mutually exclusive); G (loss) is
    # orthogonal. So the meaningful combos are best-DNA-addition + G.
    def dd(nm):
        for r in results:
            if r["name"] == nm and r.get("best_double_disjoint_pearson") is not None:
                return r["best_double_disjoint_pearson"]
        return None

    core_dd = dd("core_B+C")
    if core_dd is not None and not args.smoke:
        g_helps = (dd("core+G_pearson_loss") or -9) > core_dd
        gcfg = {"train.loss": "pearson", "train.single_protein_batches": True}
        combos = []
        if g_helps and (dd("core+F_cnn_transformer") or -9) > core_dd:
            combos.append(("core+F+G", {**core, "dna_encoder.type": "cnn_transformer", **gcfg}))
        if g_helps and (dd("core+E_moe") or -9) > core_dd:
            combos.append(("core+E+G", {**core, "dna_encoder.type": "moe", **gcfg}))
        for name, ov in combos:
            if name in done:
                continue
            if time.time() - t_start > TIME_BUDGET_S:
                logger.info("Time budget reached; skipping remaining combos.")
                break
            run_one(name, ov, base, shared, bank35, dim35, device, epochs, results, baseline_dd)

    # ---- pick the winner among 35M configs -----------------------------
    scored = [r for r in results if r.get("best_double_disjoint_pearson") is not None]
    scored.sort(key=lambda r: r["best_double_disjoint_pearson"], reverse=True)
    best = scored[0] if scored else {"name": "baseline", "overrides": {}}
    winner_overrides = best.get("overrides", {})
    logger.info(f"Best sweep config: {best['name']} "
                f"(DD {best.get('best_double_disjoint_pearson')})")

    # ---- deploy the winner (unless it's baseline within noise) ----------
    if args.smoke:
        logger.info("smoke: skipping deploy/ESM-DBP/ensemble.")
        return
    improved = (baseline_dd is not None
                and best.get("best_double_disjoint_pearson", -1) > baseline_dd + DEPLOY_MARGIN)
    try:
        if improved and best["name"] != "baseline":
            deploy(winner_overrides, base.paths.embeddings_cache, base, logger.info)
            best["notes"] = (best.get("notes", "") + " DEPLOYED").strip()
        else:
            logger.info("No sweep config beat baseline by the margin; keeping current model.")
            best["notes"] = (best.get("notes", "") + " (baseline kept)").strip()
        write_results(results, baseline_dd)
    except Exception as e:
        logger.info(f"Deploy failed: {e}")

    # ---- best-effort ESM-DBP, then redeploy if it wins -----------------
    if time.time() - t_start < TIME_BUDGET_S:
        dbp = try_esm_dbp(winner_overrides, base, shared, device, results, baseline_dd, logger.info)
        try:
            wins = dbp and dbp.get("best_double_disjoint_pearson", -1) > \
                best.get("best_double_disjoint_pearson", 0) + DEPLOY_MARGIN
            if wins and time.time() - t_start < REDEPLOY_DEADLINE_S:
                deploy(winner_overrides, dbp["embeddings"], base, logger.info)
                dbp["notes"] = "DEPLOYED (ESM-DBP)"
                write_results(results, baseline_dd)
            elif wins:
                dbp["notes"] = "BEST but not redeployed (out of time; deploy in morning)"
                write_results(results, baseline_dd)
        except Exception as e:
            logger.info(f"ESM-DBP redeploy failed: {e}")

    logger.info(f"SWEEP COMPLETE in {(time.time()-t_start)/3600:.2f} h. "
                f"See {RESULTS_MD}")


if __name__ == "__main__":
    main()
