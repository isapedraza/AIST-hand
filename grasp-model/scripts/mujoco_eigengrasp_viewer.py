"""
Interactive Shadow Hand viewer controlled by Dexonomy PCA eigengrasps.

Controls (terminal, no need to press Enter):
  LEFT / RIGHT   previous / next eigengrasp knob
  UP / DOWN      increase / decrease selected coefficient
  T              toggle target real pose vs eigengrasp reconstruction
  0              reset all coefficients
  1..9           jump to eigengrasp 1..9
  Q / ESC        quit

Run from AIST-hand/:
    python grasp-model/scripts/mujoco_eigengrasp_viewer.py
    python grasp-model/scripts/mujoco_eigengrasp_viewer.py --coeffs +1.0 -0.5 +0.2
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"
DEFAULT_EIGEN = ROOT / "grasp-model" / "data" / "processed" / "dexonomy_shadow_eigengrasps_balanced_sample.npz"

HAND_QPOS_DIM = 24
POSE_ALPHA = 0.20


def build_scene() -> Path:
    upright = SHADOW_DIR / ".right_hand_upright.xml"
    scene = SHADOW_DIR / ".scene_eigengrasp_viewer.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright.write_text(hand_text)
    scene.write_text(
        f"""<mujoco model="eigengrasp_viewer">
  <include file="{upright.name}"/>
  <statistic extent="0.3" center="0 0 0.2"/>
  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="8192"/>
    <global azimuth="145" elevation="-18"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1"/>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""
    )
    return scene


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return f"\x1b{ch2}{ch3}"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def load_eigengrasps(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {"mean_norm", "components_norm", "joint_low", "joint_high", "explained_ratio"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return {k: data[k] for k in data.files}


def reconstruct_qpos(eigen: dict[str, np.ndarray], coeffs: np.ndarray, n_knobs: int) -> np.ndarray:
    mean = eigen["mean_norm"].astype(np.float64)
    comps = eigen["components_norm"][:n_knobs].astype(np.float64)
    low = eigen["joint_low"].astype(np.float64)
    high = eigen["joint_high"].astype(np.float64)

    q_norm = mean + coeffs[:n_knobs] @ comps
    q_norm = np.clip(q_norm, 0.0, 1.0)
    q22 = q_norm * (high - low) + low
    return np.concatenate([np.zeros(2, dtype=np.float64), q22])


def project_q22(eigen: dict[str, np.ndarray], q22: np.ndarray, n_knobs: int) -> np.ndarray:
    low = eigen["joint_low"].astype(np.float64)
    high = eigen["joint_high"].astype(np.float64)
    mean = eigen["mean_norm"].astype(np.float64)
    comps = eigen["components_norm"][:n_knobs].astype(np.float64)
    q_norm = (q22.astype(np.float64) - low) / (high - low)
    return (q_norm - mean) @ comps.T


def load_target_qpos(path: Path, row: int, qpos_key: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True).item()
    if qpos_key not in raw:
        available = ", ".join(sorted(str(key) for key in raw.keys()))
        raise KeyError(f"Missing {qpos_key}. Available keys: {available}")
    qpos = raw[qpos_key]
    if len(qpos.shape) != 2 or qpos.shape[1] != 29:
        raise ValueError(f"Expected {qpos_key} shape [N,29], got {qpos.shape}")
    if row < 0 or row >= qpos.shape[0]:
        raise IndexError(f"row={row} out of bounds for {qpos_key} with {qpos.shape[0]} rows")
    return np.concatenate([np.zeros(2, dtype=np.float64), qpos[row, 7:].astype(np.float64)])


def print_status(
    idx: int,
    coeffs: np.ndarray,
    explained: np.ndarray,
    n_knobs: int,
    step: float,
    target_mode: bool,
) -> None:
    active = coeffs[:n_knobs]
    coeff_txt = " ".join(f"{i + 1}:{active[i]:+.2f}" for i in range(n_knobs))
    mode = "target-real" if target_mode else "reconstruction"
    print(
        f"\r  mode={mode:<14s} knob={idx + 1:02d}/{n_knobs}  step={step:.2f}  "
        f"PC{idx + 1} var={explained[idx] * 100:.1f}%  [{coeff_txt}]      ",
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigengrasps", type=Path, default=DEFAULT_EIGEN)
    parser.add_argument("--n-knobs", type=int, default=9)
    parser.add_argument("--step", type=float, default=0.08)
    parser.add_argument(
        "--coeffs",
        type=float,
        nargs="+",
        help="Initial eigengrasp coefficients. Missing coefficients are padded with zero.",
    )
    parser.add_argument("--target-npy", type=Path)
    parser.add_argument("--target-row", type=int, default=0)
    parser.add_argument("--target-qpos-key", default="grasp_qpos")
    args = parser.parse_args()

    eigen = load_eigengrasps(args.eigengrasps)
    n_available = int(eigen["components_norm"].shape[0])
    n_knobs = int(np.clip(args.n_knobs, 1, n_available))
    coeffs = np.zeros(n_knobs, dtype=np.float64)
    explained = eigen["explained_ratio"].astype(np.float64)
    if args.coeffs:
        n_given = min(len(args.coeffs), n_knobs)
        coeffs[:n_given] = np.asarray(args.coeffs[:n_given], dtype=np.float64)
        if len(args.coeffs) > n_knobs:
            print(f"Ignoring {len(args.coeffs) - n_knobs} coeffs beyond --n-knobs={n_knobs}.")

    target_qpos = None
    if args.target_npy:
        target_qpos = load_target_qpos(args.target_npy, args.target_row, args.target_qpos_key)
        coeffs[:] = project_q22(eigen, target_qpos[2:], n_knobs)
        print("Projected target coefficients:")
        print(" ".join(f"{i + 1}:{v:+.4f}" for i, v in enumerate(coeffs)))

    state = {"idx": 0, "quit": False, "dirty": True, "target_mode": False}

    print(f"Loaded eigengrasps: {args.eigengrasps}")
    print(f"Using {n_knobs} knobs. LEFT/RIGHT=select  UP/DOWN=change  T=target/recon  0=reset  1..9=jump  Q=quit\n")

    def keyboard_thread() -> None:
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[C":
                state["idx"] = (state["idx"] + 1) % n_knobs
                state["dirty"] = True
            elif ch == "\x1b[D":
                state["idx"] = (state["idx"] - 1) % n_knobs
                state["dirty"] = True
            elif ch == "\x1b[A":
                coeffs[state["idx"]] += args.step
                state["dirty"] = True
            elif ch == "\x1b[B":
                coeffs[state["idx"]] -= args.step
                state["dirty"] = True
            elif ch in ("t", "T") and target_qpos is not None:
                state["target_mode"] = not state["target_mode"]
                state["dirty"] = True
            elif ch == "0":
                coeffs[:] = 0.0
                state["dirty"] = True
            elif ch.isdigit() and ch != "0":
                target = int(ch) - 1
                if target < n_knobs:
                    state["idx"] = target
                    state["dirty"] = True

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    target = reconstruct_qpos(eigen, coeffs, n_knobs)
    data.qpos[:HAND_QPOS_DIM] = target
    mujoco.mj_forward(model, data)
    print_status(state["idx"], coeffs, explained, n_knobs, args.step, state["target_mode"])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                if state["target_mode"] and target_qpos is not None:
                    target = target_qpos
                else:
                    target = reconstruct_qpos(eigen, coeffs, n_knobs)
                print_status(state["idx"], coeffs, explained, n_knobs, args.step, state["target_mode"])
                state["dirty"] = False

            hand_qpos = data.qpos[:HAND_QPOS_DIM]
            hand_qpos[:] += POSE_ALPHA * (target - hand_qpos)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    state["quit"] = True
    print("\nDone.")


if __name__ == "__main__":
    main()
