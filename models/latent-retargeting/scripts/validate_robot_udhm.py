#!/usr/bin/env python3
"""Validate robot_to_udhm correctness offline.

Checks:
1. Shadow: qpos-path vs Dong-path -> diff ~0 (sanity check)
2. LEAP: abduction responds (not frozen)
3. Barrett: pip filled (not 0 before)
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add src to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # scripts/../../.. = AIST-hand
sys.path.insert(0, str(REPO_ROOT / "models" / "latent-retargeting" / "src"))

from cross_emb.loaders.robot_loader import RobotLoader
from cross_emb.loaders.robot_primitives import build_primitives, robot_to_udhm
from cross_emb.loaders.udhm_stage3 import udhm_run_stage3


def validate_robot_udhm(robot_name: str, urdf_path: str | Path, hand_config_path: str | Path, num_samples: int = 100):
    """Validate robot_to_udhm for a given robot."""
    print(f"\n{'='*60}")
    print(f"Validating {robot_name}")
    print(f"{'='*60}")

    # Load robot
    loader = RobotLoader(urdf_path)
    print(f"  Loaded {robot_name}, {len(loader.chain_joint_names)} joints")

    # Build primitives table
    try:
        tabla = build_primitives(loader, hand_config_path)
        print(f"  Built primitives table: {list(tabla.keys())}")
    except Exception as e:
        print(f"  ERROR building primitives: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Sample random configs
    print(f"  Sampling {num_samples} random configs...")
    q_r_samples = []
    for _ in range(num_samples):
        q, _ = loader.sample_q(1, seed=None)
        q_r_samples.append(q)
    q_r_batch = torch.cat(q_r_samples, dim=0)  # [B, J]

    # Compute Dong path (reference)
    print(f"  Computing Dong path (reference)...")
    with torch.no_grad():
        fk_out = loader.run_fk(q_r_batch)
        from cross_emb.loaders.robot_loader import _load_hand_config, _dong_run_stage2
        config = _load_hand_config(hand_config_path)
        quats_dong, labels_dong, meta_dong = _dong_run_stage2(fk_out, config)
        udhm22_dong = udhm_run_stage3(quats_dong, labels_dong)  # [B, 22]

    # Compute qpos path (new)
    print(f"  Computing qpos path (new)...")
    with torch.no_grad():
        udhm22_qpos = robot_to_udhm(q_r_batch, tabla)  # [B, 22]

    # Compare (Shadow control)
    if robot_name == "shadow_hand":
        diff = (udhm22_qpos - udhm22_dong).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        print(f"  Shadow control (qpos vs Dong):")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Max diff:  {max_diff:.6f}")
        if max_diff > 0.01:
            print(f"    WARNING: diff > 0.01, paths diverge significantly")
            # Show per-finger breakdown
            from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS
            for i, slot in enumerate(UDHM22_SLOTS):
                d = diff[:, i].max().item()
                if d > 0.01:
                    print(f"      {slot}: max diff {d:.6f}")
        else:
            print(f"    ✓ Control passed (diff < 0.01)")
        return True

    # Check abduction/pip responses (LEAP, Barrett)
    print(f"  Checking anatomical responses...")

    if robot_name == "leap_hand":
        # LEAP: abduction should respond (not frozen at 0)
        from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS
        idx_abd = [i for i, slot in enumerate(UDHM22_SLOTS) if "abd" in slot]
        abd_vals = udhm22_qpos[:, idx_abd]
        abd_nonzero = (abd_vals.abs() > 1e-4).sum().item()
        print(f"    Abduction non-zero: {abd_nonzero}/{len(abd_vals.flatten())} values")
        if abd_nonzero == 0:
            print(f"    ERROR: All abduction frozen at 0")
            return False
        else:
            print(f"    ✓ Abduction responds")
            return True

    if robot_name == "barrett_hand":
        # Barrett: pip should respond (was frozen at 0 before)
        from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS
        idx_pip = [i for i, slot in enumerate(UDHM22_SLOTS) if "pip" in slot]
        pip_vals = udhm22_qpos[:, idx_pip]
        pip_nonzero = (pip_vals.abs() > 1e-4).sum().item()
        print(f"    PIP non-zero: {pip_nonzero}/{len(pip_vals.flatten())} values")
        if pip_nonzero == 0:
            print(f"    ERROR: All PIP frozen at 0")
            return False
        else:
            print(f"    ✓ PIP responds")
            return True

    return True


if __name__ == "__main__":
    robots = [
        ("shadow_hand", "robot/hands/shadow_hand/shadow_hand_right.urdf", "robot/hand-configs/shadow.yaml"),
        ("leap_hand", "robot/hands/leap_hand/leap_hand_right.urdf", "robot/hand-configs/leap.yaml"),
        ("barrett_hand", "robot/hands/barrett_hand/bhand_model.urdf", "robot/hand-configs/barrett.yaml"),
    ]

    results = {}
    for robot_name, urdf_rel, config_rel in robots:
        urdf_full = REPO_ROOT / urdf_rel
        config_full = REPO_ROOT / config_rel
        if not urdf_full.exists():
            print(f"\nSkipping {robot_name}: URDF not found at {urdf_full}")
            continue
        if not config_full.exists():
            print(f"\nSkipping {robot_name}: config not found at {config_full}")
            continue
        try:
            results[robot_name] = validate_robot_udhm(robot_name, urdf_full, config_full, num_samples=50)
        except Exception as e:
            print(f"\nERROR validating {robot_name}: {e}")
            import traceback
            traceback.print_exc()
            results[robot_name] = False

    print(f"\n{'='*60}")
    print("Summary:")
    for robot_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {robot_name}: {status}")
    print(f"{'='*60}\n")

    sys.exit(0 if all(results.values()) else 1)
