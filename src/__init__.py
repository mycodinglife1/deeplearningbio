"""PBM binding predictor — a config-driven two-tower model.

The package is organized so that slow/offline work (ESM-2 embedding,
training) is cleanly separated from the fast/online prediction path used by
``main.py``. See README.md for the full picture.
"""
