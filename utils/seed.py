"""Deterministic seeding for reproducible runs.

Set as early as possible in any entry point — before importing torch
modules that allocate generators, before constructing DataLoaders.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and PYTHONHASHSEED in one call.

    PyTorch is imported lazily so this module is safe to import in
    environments that haven't installed torch yet (e.g. config-only
    smoke tests).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only=True: some ops have no deterministic implementation
    # on CPU but we still want max determinism elsewhere.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError):
        pass
