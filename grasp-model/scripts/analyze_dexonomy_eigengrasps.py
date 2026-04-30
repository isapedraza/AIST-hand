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
DEFAULT_QPOS_KEYS = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")

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


def load_qpos(
    paths_by_class: dict[str, list[Path]],
    qpos_keys: tuple[str, ...],
) -> tuple[dict[str, dict[str, list[np.ndarray]]], Counter, Counter, Counter, list[tuple[str, object]]]:
    rows_by_class_key: dict[str, dict[str, list[np.ndarray]]] = {
        cls: {key: [] for key in qpos_keys} for cls in paths_by_class
    }
    row_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    shape_counts: Counter[tuple[int, ...]] = Counter()
    bad: list[tuple[str, object]] = []

    for cls, paths in sorted(paths_by_class.items()):
        for path in paths:
            try:
                data = np.load(path, allow_pickle=True).item()
                for key in qpos_keys:
                    if key not in data:
                        bad.append((str(path), f"missing {key}"))
                        continue
                    qpos = data[key]
                    shape_counts[tuple(qpos.shape)] += 1
                    if len(qpos.shape) != 2 or qpos.shape[1] != 29:
                        bad.append((str(path), f"{key} shape {qpos.shape}"))
                        continue
                    rows = qpos[:, 7:].astype(np.float32)
                    rows_by_class_key[cls][key].append(rows)
                    row_counts[cls] += rows.shape[0]
                    key_counts[key] += rows.shape[0]
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                bad.append((str(path), repr(exc)))

    if not any(chunks for rows_by_key in rows_by_class_key.values() for chunks in rows_by_key.values()):
        raise RuntimeError("No valid Dexonomy qpos rows found.")
    return rows_by_class_key, shape_counts, row_counts, key_counts, bad


def parse_qpos_key_weights(qpos_keys: tuple[str, ...], raw_weights: list[int] | None) -> dict[str, int]:
    if raw_weights is None:
        raw_weights = [1] * len(qpos_keys)
    if len(raw_weights) != len(qpos_keys):
        raise ValueError(f"Expected {len(qpos_keys)} qpos key weights, got {len(raw_weights)}")
    if any(weight <= 0 for weight in raw_weights):
        raise ValueError(f"All qpos key weights must be positive, got {raw_weights}")
    return dict(zip(qpos_keys, raw_weights, strict=True))


def build_balanced_qpos(
    rows_by_class_key: dict[str, dict[str, list[np.ndarray]]],
    qpos_keys: tuple[str, ...],
    key_weights: dict[str, int],
    rows_per_weight_unit: int | None,
    seed: int,
) -> tuple[np.ndarray, Counter, Counter, int]:
    rng = np.random.default_rng(seed)
    available: dict[tuple[str, str], int] = {}

    for cls, rows_by_key in rows_by_class_key.items():
        for key in qpos_keys:
            available[(cls, key)] = sum(rows.shape[0] for rows in rows_by_key[key])

    max_unit = min(available[(cls, key)] // key_weights[key] for cls in rows_by_class_key for key in qpos_keys)
    if max_unit <= 0:
        raise RuntimeError("At least one class/qpos-key bucket has no usable rows.")

    unit = max_unit if rows_per_weight_unit is None else rows_per_weight_unit
    if unit <= 0:
        raise ValueError(f"--rows-per-weight-unit must be positive, got {unit}")
    if unit > max_unit:
        raise ValueError(
            f"--rows-per-weight-unit={unit} exceeds no-replacement limit {max_unit}. "
            "Lower it or reduce qpos key weights."
        )

    selected_rows: list[np.ndarray] = []
    selected_by_class: Counter[str] = Counter()
    selected_by_key: Counter[str] = Counter()

    for cls, rows_by_key in sorted(rows_by_class_key.items(), key=lambda item: class_sort_key(item[0])):
        for key in qpos_keys:
            rows = np.concatenate(rows_by_key[key], axis=0)
            n_select = unit * key_weights[key]
            indices = rng.choice(rows.shape[0], size=n_select, replace=False)
            selected = rows[indices]
            selected_rows.append(selected)
            selected_by_class[cls] += selected.shape[0]
            selected_by_key[key] += selected.shape[0]

    return np.concatenate(selected_rows, axis=0), selected_by_class, selected_by_key, unit


def build_unbalanced_qpos(
    rows_by_class_key: dict[str, dict[str, list[np.ndarray]]],
    qpos_keys: tuple[str, ...],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for _, rows_by_key in sorted(rows_by_class_key.items(), key=lambda item: class_sort_key(item[0])):
        for key in qpos_keys:
            rows.extend(rows_by_key[key])
    return np.concatenate(rows, axis=0)


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
    parser.add_argument(
        "--qpos-keys",
        nargs="+",
        default=list(DEFAULT_QPOS_KEYS),
        help=(
            "Dexonomy qpos arrays to include as PCA rows. "
            "Examples: grasp_qpos pregrasp_qpos squeeze_qpos"
        ),
    )
    parser.add_argument(
        "--qpos-key-weights",
        nargs="+",
        type=int,
        help="Positive integer weights aligned with --qpos-keys. Default: one weight per key.",
    )
    parser.add_argument(
        "--rows-per-weight-unit",
        type=int,
        help=(
            "Rows sampled per class per unit of qpos-key weight. "
            "Default: maximum possible without replacement across all class/key buckets."
        ),
    )
    parser.add_argument(
        "--no-balance-rows",
        action="store_true",
        help="Use every loaded row after file-level class sampling instead of row-balancing by class/key.",
    )
    args = parser.parse_args()
    qpos_keys = tuple(args.qpos_keys)
    key_weights = parse_qpos_key_weights(qpos_keys, args.qpos_key_weights)

    paths_by_class, total_npy = sample_paths_by_class(args.root, args.max_files_per_class, args.seed)

    print(f"root={args.root}")
    print(f"qpos_keys={qpos_keys}")
    print(f"total_npy={total_npy}")
    print(f"classes={len(paths_by_class)}")
    for cls, paths in sorted(paths_by_class.items(), key=lambda item: class_sort_key(item[0])):
        print(f"class_sample {cls}: {len(paths)}")

    rows_by_class_key, shape_counts, row_counts, key_counts, bad = load_qpos(paths_by_class, qpos_keys)
    if args.no_balance_rows:
        qpos = build_unbalanced_qpos(rows_by_class_key, qpos_keys)
        selected_by_class = row_counts
        selected_by_key = key_counts
        rows_per_weight_unit = 0
        balance_mode = "none"
    else:
        qpos, selected_by_class, selected_by_key, rows_per_weight_unit = build_balanced_qpos(
            rows_by_class_key,
            qpos_keys,
            key_weights,
            args.rows_per_weight_unit,
            args.seed,
        )
        balance_mode = "class_key_no_replacement"

    print(f"loaded_qpos_rows={qpos.shape[0]}")
    print(f"balance_mode={balance_mode}")
    print(f"qpos_key_weights={key_weights}")
    print(f"rows_per_weight_unit={rows_per_weight_unit}")
    print(f"shape_counts={dict(shape_counts)}")
    print(f"available_row_counts_by_qpos_key={dict(key_counts)}")
    print(f"available_row_counts_by_class={dict(sorted(row_counts.items(), key=lambda item: class_sort_key(item[0])))}")
    print(f"selected_row_counts_by_qpos_key={dict(selected_by_key)}")
    print(f"selected_row_counts_by_class={dict(sorted(selected_by_class.items(), key=lambda item: class_sort_key(item[0])))}")
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
        qpos_keys=np.array(qpos_keys),
        qpos_key_weights=np.array([key_weights[key] for key in qpos_keys], dtype=np.int32),
        balance_mode=balance_mode,
        rows_per_weight_unit=rows_per_weight_unit,
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
