"""Pytest bootstrap: ensure the repo root is importable so ``import src.*`` works.

Placing this at the repo root means pytest adds the root to ``sys.path`` and the
``src`` package imports cleanly without an editable install.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
