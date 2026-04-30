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

THUMB_SPREAD_MAX = 0.6981
OPEN_RELAX_MIN = -0.08
OPEN_RELAX_MAX = 0.14
FAN_PATTERNS = {
    "A": {
        "FFJ4": -0.12,
        "MFJ4": -0.04,
        "RFJ4": +0.04,
        "LFJ4": +0.10,
    },
    "B": {
        "FFJ4": +0.12,
        "MFJ4": +0.04,
        "RFJ4": -0.04,
        "LFJ4": -0.10,
    },
    "C": {
        "FFJ4": -0.10,
        "MFJ4": +0.00,
        "RFJ4": +0.08,
        "LFJ4": +0.12,
    },
    "D": {
        "FFJ4": +0.10,
        "MFJ4": +0.00,
        "RFJ4": -0.08,
        "LFJ4": -0.12,
    },
    "E": {
        "FFJ4": -0.12,
        "MFJ4": +0.00,
        "RFJ4": -0.08,
        "LFJ4": -0.12,
    },
    "F": {
        "FFJ4": +0.12,
        "MFJ4": +0.00,
        "RFJ4": +0.08,
        "LFJ4": +0.12,
    },
    "G": {
        "FFJ4": -0.16,
        "MFJ4": +0.00,
        "RFJ4": -0.10,
        "LFJ4": -0.16,
    },
    "H": {
        "FFJ4": +0.16,
        "MFJ4": +0.00,
        "RFJ4": +0.10,
        "LFJ4": +0.16,
    },
    "I": {
        "FFJ4": -0.24,
        "MFJ4": +0.00,
        "RFJ4": -0.16,
        "LFJ4": -0.24,
    },
    "J": {
        "FFJ4": -0.34,
        "MFJ4": +0.00,
        "RFJ4": -0.23,
        "LFJ4": -0.34,
    },
}
PATTERN_NAMES = tuple(FAN_PATTERNS)


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
    scene = SHADOW_DIR / ".scene_open_hand_fan_calibrator.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright_hand.write_text(hand_text)
    scene.write_text(
        f"""<mujoco model="open_hand_fan_calibrator">
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


def apply_open_params(
    base: np.ndarray,
    thumb_spread: float,
    fan: float,
    open_relax: float,
    pattern_name: str,
) -> np.ndarray:
    qpos = base.copy()
    qpos[IDX["THJ2"]] = np.clip(thumb_spread, 0.0, THUMB_SPREAD_MAX)
    relax = np.clip(open_relax, OPEN_RELAX_MIN, OPEN_RELAX_MAX)
    relax_pos = max(relax, 0.0)
    for prefix in ("FF", "MF", "RF", "LF"):
        qpos[IDX[f"{prefix}J3"]] = relax
        qpos[IDX[f"{prefix}J2"]] = 0.75 * relax_pos
        qpos[IDX[f"{prefix}J1"]] = 0.75 * relax_pos
    qpos[IDX["THJ1"]] = 0.5 * relax
    for joint, coeff in FAN_PATTERNS[pattern_name].items():
        qpos[IDX[joint]] = coeff * fan
    return qpos


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> tuple[int, float | None]:
    data.qpos[:HAND_QPOS_DIM] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return 0, None
    return int(data.ncon), min(float(data.contact[i].dist) for i in range(data.ncon))


def print_status(
    thumb_spread: float,
    fan: float,
    open_relax: float,
    pattern_name: str,
    qpos: np.ndarray,
    ncon: int,
    min_dist: float | None,
) -> None:
    values = ", ".join(f"{joint}={qpos[IDX[joint]]:+.3f}" for joint in FAN_PATTERNS[pattern_name])
    dist_text = "none" if min_dist is None else f"{min_dist:+.6f}"
    print(
        "\r  "
        f"pattern={pattern_name}  thumb_THJ2={thumb_spread:+.3f}  fan={fan:+.2f}  "
        f"relax_J3={open_relax:+.3f}  "
        f"ncon={ncon} min_dist={dist_text}  {values}      ",
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thumb", type=float, default=THUMB_SPREAD_MAX)
    parser.add_argument("--fan", type=float, default=0.0)
    parser.add_argument("--relax", type=float, default=0.0)
    parser.add_argument("--thumb-step", type=float, default=0.04)
    parser.add_argument("--fan-step", type=float, default=0.10)
    parser.add_argument("--relax-step", type=float, default=0.02)
    parser.add_argument("--pattern", choices=PATTERN_NAMES, default="A")
    args = parser.parse_args()

    base = load_keyframe("open hand")
    state = {
        "thumb": float(np.clip(args.thumb, 0.0, THUMB_SPREAD_MAX)),
        "fan": float(np.clip(args.fan, 0.0, 1.0)),
        "relax": float(np.clip(args.relax, OPEN_RELAX_MIN, OPEN_RELAX_MAX)),
        "pattern_idx": PATTERN_NAMES.index(args.pattern),
        "dirty": True,
        "quit": False,
    }

    print("Open-hand fan calibrator")
    print("UP/DOWN=thumb spread  LEFT/RIGHT=finger fan  W/S=open relax  P=pattern  0=reset fan/relax  Q/ESC=quit")
    print(f"patterns={PATTERN_NAMES}")

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    pattern = PATTERN_NAMES[state["pattern_idx"]]
    target = apply_open_params(base, state["thumb"], state["fan"], state["relax"], pattern)
    data.qpos[:HAND_QPOS_DIM] = target
    mujoco.mj_forward(model, data)

    def keyboard_thread() -> None:
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[A":
                state["thumb"] = min(THUMB_SPREAD_MAX, state["thumb"] + args.thumb_step)
                state["dirty"] = True
            elif ch == "\x1b[B":
                state["thumb"] = max(0.0, state["thumb"] - args.thumb_step)
                state["dirty"] = True
            elif ch == "\x1b[C":
                state["fan"] = min(1.0, state["fan"] + args.fan_step)
                state["dirty"] = True
            elif ch == "\x1b[D":
                state["fan"] = max(0.0, state["fan"] - args.fan_step)
                state["dirty"] = True
            elif ch in ("w", "W"):
                state["relax"] = min(OPEN_RELAX_MAX, state["relax"] + args.relax_step)
                state["dirty"] = True
            elif ch in ("s", "S"):
                state["relax"] = max(OPEN_RELAX_MIN, state["relax"] - args.relax_step)
                state["dirty"] = True
            elif ch in ("p", "P"):
                state["pattern_idx"] = (state["pattern_idx"] + 1) % len(PATTERN_NAMES)
                state["dirty"] = True
            elif ch == "0":
                state["fan"] = 0.0
                state["relax"] = 0.0
                state["dirty"] = True

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        ncon, min_dist = contact_summary(model, data, target)
        print_status(state["thumb"], state["fan"], state["relax"], pattern, target, ncon, min_dist)
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                pattern = PATTERN_NAMES[state["pattern_idx"]]
                target = apply_open_params(base, state["thumb"], state["fan"], state["relax"], pattern)
                ncon, min_dist = contact_summary(model, data, target)
                print_status(state["thumb"], state["fan"], state["relax"], pattern, target, ncon, min_dist)
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
