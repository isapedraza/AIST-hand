from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"
KEYFRAMES_XML = SHADOW_DIR / "keyframes.xml"

HAND_QPOS_DIM = 24
POSE_ALPHA = 0.18

QPOS24_NAMES = [
    "WRJ2",
    "WRJ1",
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
IDX = {name: idx for idx, name in enumerate(QPOS24_NAMES)}

THUMB_SPREAD_MIN = 0.0
THUMB_SPREAD_MAX = 0.6981
FAN_MIN = 0.0
FAN_MAX = 1.0
OPEN_RELAX_MIN = -0.08
OPEN_RELAX_MAX = 0.14

FAN_PATTERN_J = {
    "FFJ4": -0.34,
    "MFJ4": +0.00,
    "RFJ4": -0.23,
    "LFJ4": -0.34,
}


def load_keyframe(name: str) -> np.ndarray:
    root = ET.parse(KEYFRAMES_XML).getroot()
    for key in root.findall(".//key"):
        if key.attrib.get("name") != name:
            continue
        qpos = np.fromstring(key.attrib.get("qpos", ""), sep=" ", dtype=np.float64)
        if qpos.size != HAND_QPOS_DIM:
            raise ValueError(f"Expected {HAND_QPOS_DIM} values for {name!r}, got {qpos.size}")
        return qpos
    raise KeyError(f"Missing keyframe {name!r} in {KEYFRAMES_XML}")


def build_scene() -> Path:
    upright_hand = SHADOW_DIR / ".right_hand_upright.xml"
    scene = SHADOW_DIR / ".scene_synthetic_open_hand_samples.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright_hand.write_text(hand_text)
    scene.write_text(
        f"""<mujoco model="synthetic_open_hand_samples">
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


def open_qpos_from_params(base: np.ndarray, thumb_spread: float, fan: float, relax: float) -> np.ndarray:
    qpos = base.copy()
    qpos[IDX["THJ2"]] = np.clip(thumb_spread, THUMB_SPREAD_MIN, THUMB_SPREAD_MAX)

    fan = np.clip(fan, FAN_MIN, FAN_MAX)
    for joint, coeff in FAN_PATTERN_J.items():
        qpos[IDX[joint]] = coeff * fan

    relax = np.clip(relax, OPEN_RELAX_MIN, OPEN_RELAX_MAX)
    relax_pos = max(relax, 0.0)
    for prefix in ("FF", "MF", "RF", "LF"):
        qpos[IDX[f"{prefix}J3"]] = relax
        qpos[IDX[f"{prefix}J2"]] = 0.75 * relax_pos
        qpos[IDX[f"{prefix}J1"]] = 0.75 * relax_pos
    qpos[IDX["THJ1"]] = 0.5 * relax
    return qpos


def generate_samples(base: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = np.zeros((n, 3), dtype=np.float64)
    samples = np.zeros((n, HAND_QPOS_DIM), dtype=np.float64)

    for i in range(n):
        thumb_spread = rng.uniform(THUMB_SPREAD_MIN, THUMB_SPREAD_MAX)
        fan = rng.uniform(FAN_MIN, FAN_MAX)
        relax = rng.uniform(OPEN_RELAX_MIN, OPEN_RELAX_MAX)
        params[i] = (thumb_spread, fan, relax)
        samples[i] = open_qpos_from_params(base, thumb_spread, fan, relax)

    return samples, params


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> tuple[int, float | None]:
    data.qpos[:HAND_QPOS_DIM] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return 0, None
    return int(data.ncon), min(float(data.contact[i].dist) for i in range(data.ncon))


def print_status(idx: int, samples: np.ndarray, params: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    thumb, fan, relax = params[idx]
    ncon, min_dist = contact_summary(model, data, samples[idx])
    dist_text = "none" if min_dist is None else f"{min_dist:+.6f}"
    print(
        "\r  "
        f"sample={idx + 1:02d}/{samples.shape[0]}  "
        f"thumb_THJ2={thumb:+.3f}  fan={fan:.2f}  relax={relax:+.3f}  "
        f"ncon={ncon} min_dist={dist_text}      ",
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--print-qpos", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    base = load_keyframe("open hand")
    samples, params = generate_samples(base, args.n, args.seed)

    if args.print_qpos:
        for idx, qpos in enumerate(samples):
            thumb, fan, relax = params[idx]
            print(f"sample {idx + 1:02d}: thumb={thumb:+.5f} fan={fan:.5f} relax={relax:+.5f}")
            print("qpos24=" + " ".join(f"{value:+.5f}" for value in qpos))
    if args.no_viewer:
        return

    print("Synthetic open-hand samples.")
    print("LEFT/RIGHT=previous/next  Q/ESC=quit")

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    state = {"idx": 0, "quit": False, "dirty": True}
    target = samples[state["idx"]].copy()
    data.qpos[:HAND_QPOS_DIM] = target
    mujoco.mj_forward(model, data)

    def keyboard_thread() -> None:
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[C":
                state["idx"] = (state["idx"] + 1) % args.n
                state["dirty"] = True
            elif ch == "\x1b[D":
                state["idx"] = (state["idx"] - 1) % args.n
                state["dirty"] = True

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print_status(state["idx"], samples, params, model, data)
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                target = samples[state["idx"]].copy()
                print_status(state["idx"], samples, params, model, data)
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
