#!/usr/bin/env python3
"""
Generate synthetic open/close Barrett poses to anchor the eigengrasp PCA basis.

MultiDex barrett has only final grasps (no phases), so the basis would not span a
fully-open hand nor a closed fist. We synthesize open + close parametrically and
collision-filter against the dex-urdf bhand_model.urdf, then feed them to
build_barrett_eigengrasps.py via --synthetic-qpos-npz (key qpos8).

URDF qpos order (8): f1(spread,med,dist), f2(spread,med,dist), f3(med,dist).
  spread: f1 [-3.14,0], f2 [0,3.14]   med: [-2.44,0]   dist: [-0.785,0]
Open = all zeros (flex 0 = extended, spread 0 = neutral). Close = fingers curled.

Usage:
    python generate_synthetic_open_close_barrett.py --rows 30000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HD = ROOT / "robot/hands/barrett_hand"
LOW8 = np.array([-3.140, -2.440, -0.785, 0.000, -2.440, -0.785, -2.440, -0.785])
HIGH8 = np.array([0.000, 0.000, 0.000, 3.140, 0.000, 0.000, 0.000, 0.000])

SPREAD = [0, 3]              # f1, f2 base rotation
MED = [1, 4, 6]             # medial flexion (3 fingers)
DIST = [2, 5, 7]            # distal flexion (3 fingers)


def build_urdf() -> Path:
    txt = (HD / "bhand_model.urdf").read_text()
    inj = (f'<mujoco><compiler meshdir="{HD.resolve()}" strippath="false" '
           f'balanceinertia="true" discardvisual="false"/></mujoco>')
    out = HD / ".bhand_synth.urdf"
    out.write_text(re.sub(r'(<robot[^>]*>)', r'\1\n  ' + inj, txt, count=1))
    return out


def open_target() -> np.ndarray:
    return np.zeros(8)   # extended, neutral spread


def close_target(curl: float = 0.7) -> np.ndarray:
    t = np.zeros(8)
    for i in MED:
        t[i] = curl * LOW8[i]    # toward -2.44 (curl)
    for i in DIST:
        t[i] = curl * LOW8[i]    # toward -0.785
    # spread neutral (fingers curl straight in); collision filter prunes the rest
    return t


def sample_around(target, n, jitter, rng):
    span = (HIGH8 - LOW8)[None, :]
    q = target[None, :] + rng.uniform(-jitter, jitter, size=(n, 8)) * span
    return np.clip(q, LOW8, HIGH8)


def collision_ok(m, d, q8, tol):
    d.qpos[:] = q8
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    if d.ncon == 0:
        return True
    return min(float(d.contact[i].dist) for i in range(d.ncon)) >= -tol


def generate(name, target, rows, jitter, seed, out, tol):
    m = mujoco.MjModel.from_xml_path(str(build_urdf()))
    d = mujoco.MjData(m)
    # baseline self-contact at the target (barrett collision cylinders overlap a bit
    # by construction), measured so the filter is relative to a feasible pose.
    d.qpos[:] = np.clip(target, LOW8, HIGH8); d.qvel[:] = 0; mujoco.mj_forward(m, d)
    base = 0.0 if d.ncon == 0 else -min(float(d.contact[i].dist) for i in range(d.ncon))
    allow = base + tol
    print(f"{name}: baseline self-pen {base*1000:.1f}mm, allow {allow*1000:.1f}mm")
    rng = np.random.default_rng(seed)
    acc = []
    gen = 0
    maxc = rows * 200
    while len(acc) < rows and gen < maxc:
        batch = sample_around(target, 4096, jitter, rng)
        gen += batch.shape[0]
        for q in batch:
            if collision_ok(m, d, q, allow):
                acc.append(q.astype(np.float32))
                if len(acc) >= rows:
                    break
    if not acc:
        print(f"ERROR: 0 {name} accepted. soften curl or raise tol."); return
    q = np.stack(acc[:rows])
    print(f"{name}: accepted {q.shape[0]}/{rows} (gen {gen}, {100*len(acc)/gen:.1f}%)")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, synthetic_pose_name=f"synthetic_{name}_barrett",
                        target=target.astype(np.float32), base_qpos8=target.astype(np.float32),
                        joint_low=LOW8.astype(np.float32), joint_high=HIGH8.astype(np.float32),
                        qpos8=q)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    proc = HD / "datasets/processed"
    ap.add_argument("--out-open", type=Path, default=proc / "synthetic_open_barrett.npz")
    ap.add_argument("--out-close", type=Path, default=proc / "synthetic_close_barrett.npz")
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--jitter", type=float, default=0.06)
    ap.add_argument("--contact-tol", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--curl", type=float, default=0.7)
    args = ap.parse_args()
    generate("open", open_target(), args.rows, args.jitter, args.seed, args.out_open, args.contact_tol)
    generate("close", close_target(args.curl), args.rows, args.jitter, args.seed + 1, args.out_close, args.contact_tol)


if __name__ == "__main__":
    main()
