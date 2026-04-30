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
DEFAULT_NPZ = ROOT / "grasp-model" / "data" / "processed" / "synthetic_close_hand_shadow_qpos.npz"

HAND_QPOS_DIM = 24
POSE_ALPHA = 0.18


def build_scene() -> Path:
    upright_hand = SHADOW_DIR / ".right_hand_upright.xml"
    scene = SHADOW_DIR / ".scene_synthetic_qpos_npz_viewer.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright_hand.write_text(hand_text)
    scene.write_text(
        f"""<mujoco model="synthetic_qpos_npz_viewer">
  <include file="{upright_hand.name}"/>
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


def load_qpos24(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    if "qpos24" in data:
        qpos24 = data["qpos24"].astype(np.float64)
    elif "qpos22" in data and "base_qpos24" in data:
        qpos22 = data["qpos22"].astype(np.float64)
        wrist = np.repeat(data["base_qpos24"][None, :2].astype(np.float64), qpos22.shape[0], axis=0)
        qpos24 = np.concatenate([wrist, qpos22], axis=1)
    else:
        raise KeyError(f"{path} must contain qpos24, or qpos22 plus base_qpos24")

    if qpos24.ndim != 2 or qpos24.shape[1] != HAND_QPOS_DIM:
        raise ValueError(f"Expected qpos24 shape [N,{HAND_QPOS_DIM}], got {qpos24.shape}")

    meta = {key: data[key] for key in data.files if key != "qpos24"}
    return qpos24, meta


def print_status(idx: int, qpos24: np.ndarray, meta: dict[str, np.ndarray]) -> None:
    parts = [f"sample={idx + 1:05d}/{qpos24.shape[0]}"]
    if "sample_ncon" in meta:
        parts.append(f"ncon={int(meta['sample_ncon'][idx])}")
    if "sample_min_contact_dist" in meta:
        parts.append(f"min_dist={float(meta['sample_min_contact_dist'][idx]):+.6f}")
    if "base_qpos24" in meta:
        delta = qpos24[idx] - meta["base_qpos24"].astype(np.float64)
        parts.append(f"max_delta={np.max(np.abs(delta[2:])):.4f}")
    print("\r  " + "  ".join(parts) + "      ", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    qpos24, meta = load_qpos24(args.npz)
    if qpos24.shape[0] == 0:
        raise ValueError(f"No qpos rows in {args.npz}")
    if args.step <= 0:
        raise ValueError(f"--step must be positive, got {args.step}")

    idx0 = int(np.clip(args.start, 0, qpos24.shape[0] - 1))
    state = {"idx": idx0, "quit": False, "dirty": True}
    target = qpos24[state["idx"]].copy()

    print(f"Loaded {args.npz}")
    if "synthetic_pose_name" in meta:
        print(f"pose={str(meta['synthetic_pose_name'])}")
    print("LEFT/RIGHT=previous/next  PAGEUP/PAGEDOWN=jump  Q/ESC=quit")

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    data.qpos[:HAND_QPOS_DIM] = target
    mujoco.mj_forward(model, data)

    def keyboard_thread() -> None:
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[C":
                state["idx"] = min(qpos24.shape[0] - 1, state["idx"] + args.step)
                state["dirty"] = True
            elif ch == "\x1b[D":
                state["idx"] = max(0, state["idx"] - args.step)
                state["dirty"] = True
            elif ch == "\x1b[5":
                _ = getch()
                state["idx"] = min(qpos24.shape[0] - 1, state["idx"] + 100)
                state["dirty"] = True
            elif ch == "\x1b[6":
                _ = getch()
                state["idx"] = max(0, state["idx"] - 100)
                state["dirty"] = True

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print_status(state["idx"], qpos24, meta)
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                target = qpos24[state["idx"]].copy()
                print_status(state["idx"], qpos24, meta)
                state["dirty"] = False

            hand_qpos = data.qpos[:HAND_QPOS_DIM]
            hand_qpos[:] = hand_qpos + POSE_ALPHA * (target - hand_qpos)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    print()


if __name__ == "__main__":
    main()
