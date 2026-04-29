from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    "/media/yareeez/94649A33649A1856/Dexonomy/data/"
    "Dexonomy_GRASP_shadow/succ_collect"
)
DEFAULT_OUT = Path(
    "grasp-model/data/processed/"
    "dexonomy_shadow_eigengrasps_balanced_sample.npz"
)

JOINTS22 = [
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
]

LOW22 = np.array(
    [
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -1.047,
        0.0,
        -0.209,
        -0.698,
        -0.2618,
    ],
    dtype=np.float32,
)
HIGH22 = np.array(
    [
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.7854,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        1.047,
        1.2217,
        0.209,
        0.698,
        1.5708,
    ],
    dtype=np.float32,
)


def class_sort_key(name: str) -> int:
    return int(name.split("_", 1)[0])


def sample_paths_by_class(root: Path, max_files_per_class: int, seed: int) -> tuple[dict[str, list[Path]], int]:
    rng = random.Random(seed)
    paths_by_class: dict[str, list[Path]] = defaultdict(list)
    total = 0
    seen_by_class: Counter[str] = Counter()

    for path in root.rglob("*.npy"):
        total += 1
        cls = path.relative_to(root).parts[0]
        seen_by_class[cls] += 1
        bucket = paths_by_class[cls]
        if len(bucket) < max_files_per_class:
            bucket.append(path)
            continue

        j = rng.randrange(seen_by_class[cls])
        if j < max_files_per_class:
            bucket[j] = path

    return dict(paths_by_class), total


def load_qpos(paths_by_class: dict[str, list[Path]]) -> tuple[np.ndarray, Counter, Counter, list[tuple[str, object]]]:
    q_rows: list[np.ndarray] = []
    row_counts: Counter[str] = Counter()
    shape_counts: Counter[tuple[int, ...]] = Counter()
    bad: list[tuple[str, object]] = []

    for cls, paths in sorted(paths_by_class.items()):
        for path in paths:
            try:
                data = np.load(path, allow_pickle=True).item()
                grasp_qpos = data["grasp_qpos"]
                shape_counts[tuple(grasp_qpos.shape)] += 1
                if len(grasp_qpos.shape) != 2 or grasp_qpos.shape[1] != 29:
                    bad.append((str(path), grasp_qpos.shape))
                    continue
                rows = grasp_qpos[:, 7:].astype(np.float32)
                q_rows.append(rows)
                row_counts[cls] += rows.shape[0]
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                bad.append((str(path), repr(exc)))

    if not q_rows:
        raise RuntimeError("No valid Dexonomy qpos rows found.")
    return np.concatenate(q_rows, axis=0), shape_counts, row_counts, bad


def print_joint_ranges(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmin = qpos.min(axis=0)
    qmax = qpos.max(axis=0)
    qmean = qpos.mean(axis=0)
    viol_low = (qpos < (LOW22 - 1e-4)).sum(axis=0)
    viol_high = (qpos > (HIGH22 + 1e-4)).sum(axis=0)

    print("joint_ranges:")
    for i, name in enumerate(JOINTS22):
        print(
            f"  {i:02d} {name}: "
            f"min={qmin[i]: .4f} mean={qmean[i]: .4f} max={qmax[i]: .4f} "
            f"limits=[{LOW22[i]: .4f},{HIGH22[i]: .4f}] "
            f"viol_low={int(viol_low[i])} viol_high={int(viol_high[i])}"
        )
    return qmin, qmax, qmean


def compute_pca(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = HIGH22 - LOW22
    qnorm = (qpos - LOW22) / scale
    mean = qnorm.mean(axis=0)
    centered = qnorm - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    explained = (singular_values * singular_values) / max(1, centered.shape[0] - 1)
    ratio = explained / explained.sum()
    cumulative = np.cumsum(ratio)
    return mean, components, ratio, cumulative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-files-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    paths_by_class, total_npy = sample_paths_by_class(args.root, args.max_files_per_class, args.seed)

    print(f"root={args.root}")
    print(f"total_npy={total_npy}")
    print(f"classes={len(paths_by_class)}")
    for cls, paths in sorted(paths_by_class.items(), key=lambda item: class_sort_key(item[0])):
        print(f"class_sample {cls}: {len(paths)}")

    qpos, shape_counts, row_counts, bad = load_qpos(paths_by_class)
    print(f"loaded_qpos_rows={qpos.shape[0]}")
    print(f"shape_counts={dict(shape_counts)}")
    print(f"row_counts_by_class={dict(sorted(row_counts.items(), key=lambda item: class_sort_key(item[0])))}")
    print(f"bad_count={len(bad)}")
    if bad[:5]:
        print(f"bad_examples={bad[:5]}")

    qmin, qmax, qmean = print_joint_ranges(qpos)
    mean, components, ratio, cumulative = compute_pca(qpos)

    print("pca_normalized:")
    for i, value in enumerate(ratio):
        print(f"  PC{i + 1:02d}: ratio={value:.5f} cum={cumulative[i]:.5f}")
    for threshold in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"  k_for_{int(threshold * 100)}={k}")

    print("top_pc_loadings:")
    for pc_idx in range(6):
        comp = components[pc_idx]
        order = np.argsort(np.abs(comp))[::-1][:8]
        terms = ", ".join(f"{JOINTS22[j]}:{comp[j]:+.3f}" for j in order)
        print(f"  PC{pc_idx + 1:02d}: {terms}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        source_root=str(args.root),
        seed=args.seed,
        max_files_per_class=args.max_files_per_class,
        joint_names=np.array(JOINTS22),
        joint_low=LOW22,
        joint_high=HIGH22,
        mean_norm=mean.astype(np.float32),
        components_norm=components.astype(np.float32),
        explained_ratio=ratio.astype(np.float32),
        explained_cumulative=cumulative.astype(np.float32),
        q_min=qmin.astype(np.float32),
        q_max=qmax.astype(np.float32),
        q_mean=qmean.astype(np.float32),
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
