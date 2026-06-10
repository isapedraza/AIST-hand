"""
Browse valid_robot_poses NPZ in MuJoCo. Works for Shadow and Allegro.

Controls (terminal, no need to press Enter):
  LEFT / RIGHT     previous / next pose
  PAGEUP / PAGEDOWN  jump ±100
  R                random pose
  Q / ESC          quit

Run from AIST-hand/:
    python models/grasp-intent-classification/scripts/mujoco_valid_poses_viewer.py --robot allegro
    python models/grasp-intent-classification/scripts/mujoco_valid_poses_viewer.py --robot shadow
    python models/grasp-intent-classification/scripts/mujoco_valid_poses_viewer.py --robot allegro --npz path/to/poses.npz
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
_MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"

POSE_ALPHA = 0.18


@dataclass
class RobotConfig:
    name: str
    hand_dir: Path
    orig_body_tag: str
    new_body_tag: str
    qpos_dim: int
    default_npz: Path


_SHADOW = RobotConfig(
    name="shadow",
    hand_dir=_MENAGERIE / "shadow_hand",
    orig_body_tag='<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
    new_body_tag='<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
    qpos_dim=24,
    default_npz=ROOT / "robot/hands/shadow_hand/datasets/processed/valid_robot_poses_eigengrasp_dong.npz",
)

_ALLEGRO = RobotConfig(
    name="allegro",
    hand_dir=_MENAGERIE / "wonik_allegro",
    orig_body_tag='<body name="palm" quat="0 1 0 1" childclass="allegro_right">',
    new_body_tag='<body name="palm" pos="0 0 0.05" quat="1 0 0 0" childclass="allegro_right">',
    qpos_dim=16,
    default_npz=ROOT / "robot/hands/allegro_hand/datasets/processed/valid_robot_poses_dong.npz",
)

ROBOTS: dict[str, RobotConfig] = {"shadow": _SHADOW, "allegro": _ALLEGRO}


def build_scene(cfg: RobotConfig) -> Path:
    right_hand = cfg.hand_dir / "right_hand.xml"
    upright = cfg.hand_dir / ".right_hand_upright_poses.xml"
    scene = cfg.hand_dir / ".scene_valid_poses_viewer.xml"

    upright.write_text(right_hand.read_text().replace(cfg.orig_body_tag, cfg.new_body_tag, 1))
    scene.write_text(
        f"""<mujoco model="valid_poses_viewer_{cfg.name}">
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


def print_status(idx: int, total: int) -> None:
    print(f"\r  pose={idx + 1:>10,}/{total:,}  LEFT/RIGHT=step  PGUP/PGDN=±100  R=random  Q=quit      ",
          end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=list(ROBOTS), default="allegro")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Override NPZ path (default: valid_robot_poses_dong.npz for chosen robot)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ROBOTS[args.robot]
    npz_path = args.npz or cfg.default_npz

    if not npz_path.exists():
        print(f"NPZ not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=False)
    q_all = data["q"].astype(np.float64)          # [N, J]
    N, J = q_all.shape
    print(f"Robot: {cfg.name}  |  Poses: {N:,}  |  Joints: {J}  |  {npz_path.name}")

    if J != cfg.qpos_dim:
        print(f"Warning: npz J={J} != expected qpos_dim={cfg.qpos_dim}. Proceeding anyway.")

    rng = np.random.default_rng(args.seed)
    idx0 = int(np.clip(args.start, 0, N - 1))
    state = {"idx": idx0, "quit": False, "dirty": True}

    def keyboard_thread() -> None:
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[C":
                state["idx"] = min(N - 1, state["idx"] + args.step)
                state["dirty"] = True
            elif ch == "\x1b[D":
                state["idx"] = max(0, state["idx"] - args.step)
                state["dirty"] = True
            elif ch == "\x1b[5":
                _ = getch()
                state["idx"] = min(N - 1, state["idx"] + 100)
                state["dirty"] = True
            elif ch == "\x1b[6":
                _ = getch()
                state["idx"] = max(0, state["idx"] - 100)
                state["dirty"] = True
            elif ch in ("r", "R"):
                state["idx"] = int(rng.integers(0, N))
                state["dirty"] = True

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()

    model = mujoco.MjModel.from_xml_path(str(build_scene(cfg)))
    mjdata = mujoco.MjData(model)
    target = q_all[state["idx"]].copy()
    mjdata.qpos[:cfg.qpos_dim] = target
    mujoco.mj_forward(model, mjdata)
    print_status(state["idx"], N)

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                target = q_all[state["idx"]].copy()
                print_status(state["idx"], N)
                state["dirty"] = False

            hand_qpos = mjdata.qpos[:cfg.qpos_dim]
            hand_qpos[:] += POSE_ALPHA * (target - hand_qpos)
            mjdata.qvel[:] = 0
            mujoco.mj_forward(model, mjdata)
            viewer.sync()
            time.sleep(0.01)

    state["quit"] = True
    print("\nDone.")


if __name__ == "__main__":
    main()
