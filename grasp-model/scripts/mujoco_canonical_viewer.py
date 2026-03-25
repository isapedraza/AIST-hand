"""
Interactive viewer for canonical Shadow Hand poses from Dexonomy anchors.

Controls (terminal, no need to press Enter):
  LEFT / RIGHT   : previous / next grasp class
  UP             : increase apertura (+0.05)
  DOWN           : decrease apertura (-0.05)
  Q / ESC        : quit

Run from AIST-hand/:
    python grasp-model/scripts/mujoco_canonical_viewer.py
"""

from __future__ import annotations

import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

ROOT       = Path(__file__).resolve().parents[2]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"
CANONICAL  = ROOT / "grasp-robot" / "grasp_configs" / "shadow_hand_canonical_v5_grasp.yaml"

HAND_QPOS_DIM = 24
POSE_ALPHA    = 0.15
APERTURA_STEP = 0.05


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_canonical() -> list[dict]:
    with open(CANONICAL) as f:
        raw = yaml.safe_load(f)
    classes = []
    for key, val in sorted(raw.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else -1):
        if not str(key).isdigit():
            continue
        classes.append({
            "hog_id":      int(key),
            "name":        val["class_name"],
            "pose_open":   np.array(val["pose_open"],  dtype=np.float64),
            "pose_close":  np.array(val["pose_close"], dtype=np.float64),
            "apertura_max": float(val.get("apertura_max", 1.0)),
        })
    return classes


def qpos_for(entry: dict, apertura: float) -> np.ndarray:
    a = float(np.clip(apertura, 0.0, entry["apertura_max"]))
    return (1.0 - a) * entry["pose_open"] + a * entry["pose_close"]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def build_scene() -> Path:
    upright = SHADOW_DIR / ".right_hand_upright.xml"
    scene   = SHADOW_DIR / ".scene_canonical_viewer.xml"
    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright.write_text(hand_text)
    scene.write_text(f"""<mujoco model="canonical_viewer">
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
""")
    return scene


# ---------------------------------------------------------------------------
# Keyboard (termios, Linux)
# ---------------------------------------------------------------------------

def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":          # escape sequence (arrow keys)
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return f"\x1b{ch2}{ch3}"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def print_status(entry: dict, apertura: float) -> None:
    bar_len = 20
    filled  = round(apertura * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{entry['hog_id']:2d}] {entry['name']:<26s}  "
          f"apertura {apertura:.2f}  [{bar}]     ", end="", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    classes  = load_canonical()
    n        = len(classes)

    state = {"idx": 0, "apertura": 0.5, "quit": False, "dirty": True}

    print(f"Loaded {n} classes  |  LEFT/RIGHT=class  UP/DOWN=apertura  Q=quit\n")

    def keyboard_thread():
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):   # q, Ctrl-C, ESC
                state["quit"] = True
            elif ch == "\x1b[C":   # RIGHT
                state["idx"] = (state["idx"] + 1) % n
                state["dirty"] = True
            elif ch == "\x1b[D":   # LEFT
                state["idx"] = (state["idx"] - 1) % n
                state["dirty"] = True
            elif ch == "\x1b[A":   # UP
                state["apertura"] = min(1.0, state["apertura"] + APERTURA_STEP)
                state["dirty"] = True
            elif ch == "\x1b[B":   # DOWN
                state["apertura"] = max(0.0, state["apertura"] - APERTURA_STEP)
                state["dirty"] = True

    t = threading.Thread(target=keyboard_thread, daemon=True)
    t.start()

    model  = mujoco.MjModel.from_xml_path(str(build_scene()))
    data   = mujoco.MjData(model)
    target = qpos_for(classes[0], 0.5).copy()
    data.qpos[:HAND_QPOS_DIM] = target
    mujoco.mj_forward(model, data)
    print_status(classes[0], 0.5)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                entry  = classes[state["idx"]]
                target = qpos_for(entry, state["apertura"])
                print_status(entry, state["apertura"])
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
