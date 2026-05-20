"""Ensure repo-root domain packages are importable from app entrypoints."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = [
    REPO_ROOT,
    REPO_ROOT / "models" / "grasp-intent-classification" / "src",
    REPO_ROOT / "models" / "latent-retargeting" / "src",
]

for path in reversed(PATHS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
