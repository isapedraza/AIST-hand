#!/usr/bin/env python3
"""
Barrett clip-vs-noclip interactive viewer.

Loads the Barrett URDF with WIDENED distal limits so it can show the full distal
flexion from the data. Press 'c' to toggle clipping the distal joints to dex-urdf's
limit (-0.785) vs the full data value. Same grasp, see the fingertip difference live.

  LEFT/RIGHT   previous / next grasp
  PGUP/PGDN    +/- 100 grasps
  c            toggle distal clip (dex-urdf 0.785) vs full data
  q / ESC      quit

Run from AIST-hand/:
  python models/grasp-intent-classification/scripts/mujoco_barrett_clip_compare.py
"""
from __future__ import annotations

import re
import sys
import termios
import threading
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HD = ROOT / "robot/hands/barrett_hand"
NPZ = HD / "datasets/processed/_remapped_real_grasps.npz"
DISTAL_IDX = [2, 5, 7]          # urdf qpos slots for the 3 distal joints
DISTAL_CLIP = -0.785           # dex-urdf distal lower limit


def build_widened_urdf() -> Path:
    txt = (HD / "bhand_model.urdf").read_text()
    txt = txt.replace('lower="-0.785"', 'lower="-1.30"')   # widen distal so full shows
    inj = (f'<mujoco><compiler meshdir="{HD.resolve()}" strippath="false" '
           f'balanceinertia="true" discardvisual="false"/></mujoco>')
    out = HD / ".bhand_clipcompare.urdf"
    out.write_text(re.sub(r'(<robot[^>]*>)', r'\1\n  ' + inj, txt, count=1))
    return out


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            return f"\x1b{sys.stdin.read(1)}{sys.stdin.read(1)}"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    q_all = np.load(NPZ)["q"].astype(np.float64)   # full remapped grasps [N,8]
    N = q_all.shape[0]
    m = mujoco.MjModel.from_xml_path(str(build_widened_urdf()))
    d = mujoco.MjData(m)
    state = {"idx": 0, "clip": False, "quit": False, "dirty": True}

    def apply():
        q = q_all[state["idx"]].copy()
        if state["clip"]:
            for j in DISTAL_IDX:
                q[j] = max(q[j], DISTAL_CLIP)   # clamp distal to dex-urdf limit
        d.qpos[:] = np.clip(q, m.jnt_range[:, 0], m.jnt_range[:, 1])
        mujoco.mj_forward(m, d)

    def status():
        mode = "CLIP (0.785)" if state["clip"] else "FULL (data)"
        print(f"\r  grasp={state['idx']+1:>6}/{N}   distal={mode}   "
              f"[c=toggle  LEFT/RIGHT  PGUP/PGDN  q=quit]     ", end="", flush=True)

    def kb():
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[C":
                state["idx"] = min(N - 1, state["idx"] + 1); state["dirty"] = True
            elif ch == "\x1b[D":
                state["idx"] = max(0, state["idx"] - 1); state["dirty"] = True
            elif ch == "\x1b[5":
                getch(); state["idx"] = min(N - 1, state["idx"] + 100); state["dirty"] = True
            elif ch == "\x1b[6":
                getch(); state["idx"] = max(0, state["idx"] - 100); state["dirty"] = True
            elif ch in ("c", "C"):
                state["clip"] = not state["clip"]; state["dirty"] = True

    threading.Thread(target=kb, daemon=True).start()
    print("Barrett clip-vs-full. 'c' toggles distal clip. Same grasp, watch fingertips.")
    apply(); status()
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running() and not state["quit"]:
            if state["dirty"]:
                apply(); status(); state["dirty"] = False
            v.sync()


if __name__ == "__main__":
    main()
