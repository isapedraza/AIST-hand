"""
Build hograspnet_abl14.csv from hograspnet_abl11.csv.

ABL14 keeps the ABL11 metadata and wrist-local XYZ columns, replaces the
20 Dong quaternions with R6 rotation columns, and does not write quaternion
columns to the output CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = REPO_ROOT / "human" / "datasets" / "hograspnet" / "processed"
DEFAULT_INPUT = PROCESSED / "hograspnet_abl11.csv"
DEFAULT_OUTPUT = PROCESSED / "hograspnet_abl14.csv"

QUAT_COLS = [f"q{j}_{c}" for j in range(1, 21) for c in ("w", "x", "y", "z")]
R6_COLS = [
    f"q{j}_r6_{c}"
    for j in range(1, 21)
    for c in ("c1x", "c1y", "c1z", "c2x", "c2y", "c2z")
]


def _quat_wxyz_to_r6(q: np.ndarray) -> np.ndarray:
    """Convert ``[...,4]`` wxyz quaternions to ``[...,6]`` R6 columns."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    q = q / norm
    w, x, y, z = np.moveaxis(q, -1, 0)

    c1 = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        axis=-1,
    )
    c2 = np.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        axis=-1,
    )
    return np.concatenate([c1, c2], axis=-1).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input ABL11-style CSV with q*_w/x/y/z columns.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output ABL14 CSV path.")
    p.add_argument("--chunksize", type=int, default=100_000, help="Rows per pandas chunk.")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(out_path)
        out_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_write = True
    total = 0
    print(f"[abl14] input  = {in_path}")
    print(f"[abl14] output = {out_path}")

    for chunk in pd.read_csv(in_path, chunksize=args.chunksize):
        missing = [c for c in QUAT_COLS if c not in chunk.columns]
        if missing:
            raise ValueError(f"Input CSV is missing quaternion columns, first missing: {missing[:5]}")

        quats = chunk[QUAT_COLS].to_numpy(dtype=np.float32).reshape(-1, 20, 4)
        r6 = _quat_wxyz_to_r6(quats).reshape(len(chunk), 20 * 6)
        r6_df = pd.DataFrame(r6, columns=R6_COLS, index=chunk.index)

        out_chunk = pd.concat([chunk.drop(columns=QUAT_COLS), r6_df], axis=1)
        out_chunk.to_csv(out_path, mode="a", header=first_write, index=False)
        first_write = False
        total += len(chunk)
        print(f"[abl14] wrote {total:,} rows", flush=True)

    print(f"[abl14] done. rows={total:,}")


if __name__ == "__main__":
    main()
