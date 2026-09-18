"""Regenerate tests/data/ia2pt_reference_gausscov.npz (see test_gausscov.py)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_gausscov import compute_synthetic, synthetic  # noqa: E402

syn = synthetic.__wrapped__()
np.savez(Path(__file__).parent / "data" / "ia2pt_reference_gausscov.npz", **compute_synthetic(syn))
print("wrote")
