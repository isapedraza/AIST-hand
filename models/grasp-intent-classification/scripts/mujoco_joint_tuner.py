#!/usr/bin/env python3
"""
Interactive per-joint pose tuner in MuJoCo. Craft a qpos by hand.

Reuses RobotConfig/build_scene from mujoco_valid_poses_viewer (shadow/allegro/leap),
so the hand orientation matches the pose viewers.

Controls (terminal, no Enter needed):
  LEFT / RIGHT   select previous / next joint
  UP / DOWN      increase / decrease the selected joint by the current step
  [ / ]          halve / double the step size
  0              zero the selected joint
  p              print the full qpos (space-separated + python list)
  r              reset all joints to the start pose
  q / ESC        quit

Start pose:
  --npz PATH --start N   load row N (0-based) of qpos{dim}/q as the start pose
  (default: all zeros)

Run from AIST-hand/:
  python models/grasp-intent-classification/scripts/mujoco_joint_tuner.py \
      --robot leap \
      --npz robot/hands/leap_hand/datasets/processed/_thumbsweep_leap.npz --start 7
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_valid_poses_viewer import ROBOTS, build_scene  # noqa: E402


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


def load_start(npz: Path | None, start: int, J: int) -> np.ndarray:
    if npz is None:
        return np.zeros(J, dtype=np.float64)
    data = np.load(npz, allow_pickle=False)
    qkey = next((k for k in ("q", f"qpos{J}", "qpos16", "qpos22", "qpos24") if k in data.files), None)
    if qkey is None:
        raise KeyError(f"No pose array in {npz}. Has: {list(data.files)}")
    arr = data[qkey].astype(np.float64)
    row = int(np.clip(start, 0, arr.shape[0] - 1))
    return arr[row].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="leap")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--thumb-only", action="store_true", help="Only allow tuning thumb joints (last 4).")
    ap.add_argument("--joints", type=str, default=None,
                    help="Comma-separated joint indices to allow (overrides --thumb-only).")
    ap.add_argument("--pose", type=str, default=None,
                    help="Comma-separated qpos to use as start pose (overrides --npz).")
    args = ap.parse_args()

    cfg = ROBOTS[args.robot]
    model = mujoco.MjModel.from_xml_path(str(build_scene(cfg)))
    data = mujoco.MjData(model)
    J = cfg.qpos_dim

    names = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        names.append(nm if nm else f"j{i}")
    names = names[:J]
    lo = model.jnt_range[:J, 0].copy()
    hi = model.jnt_range[:J, 1].copy()

    # Finger flexion joints (mcp/pip/dip of the non-thumb fingers), no abduction/thumb.
    FLEX_GROUP = {
        "leap":    [0, 2, 3, 4, 6, 7, 8, 10, 11],   # if/mf/rf: mcp,pip,dip
        "allegro": [1, 2, 3, 5, 6, 7, 9, 10, 11],   # index/middle/ring: mcp,pip,dip
    }
    # sel_items: list of (label, [indices moved together]). Single joint = 1-elem list.
    if args.joints:
        sel_items = [(names[i], [i]) for i in (int(x) for x in args.joints.split(","))]
    elif args.thumb_only:
        sel_items = [(names[i], [i]) for i in range(J - 4, J)]
    elif args.robot in FLEX_GROUP:
        sel_items = [("fingers_flex", FLEX_GROUP[args.robot])] + [(names[i], [i]) for i in range(J - 4, J)]
    else:
        sel_items = [(names[i], [i]) for i in range(J)]

    if args.pose:
        start_q = np.array([float(x) for x in args.pose.split(",")], dtype=np.float64)
        assert start_q.size == J, f"--pose needs {J} values, got {start_q.size}"
    else:
        start_q = load_start(args.npz, args.start, J)
    state = {"q": start_q.copy(), "sel": 0, "step": args.step, "quit": False, "dirty": True}

    def status() -> None:
        label, idxs = sel_items[state["sel"]]
        q = state["q"]
        if len(idxs) == 1:
            s = idxs[0]
            print(f"\r  joint[{s:2d}] {label:<12s} = {q[s]:+.3f}  "
                  f"[{lo[s]:+.2f},{hi[s]:+.2f}]  step={state['step']:.3f}        ", end="", flush=True)
        else:
            print(f"\r  group {label:<12s} n={len(idxs)}  mean={q[idxs].mean():+.3f}  "
                  f"step={state['step']:.3f} (moves all together)        ", end="", flush=True)

    def kb() -> None:
        while not state["quit"]:
            ch = getch()
            if ch in ("q", "Q", "\x03", "\x1b\x1b\x1b"):
                state["quit"] = True
            elif ch == "\x1b[D":
                state["sel"] = (state["sel"] - 1) % len(sel_items); state["dirty"] = True
            elif ch == "\x1b[C":
                state["sel"] = (state["sel"] + 1) % len(sel_items); state["dirty"] = True
            elif ch == "\x1b[A":
                for s in sel_items[state["sel"]][1]:
                    state["q"][s] = min(hi[s], state["q"][s] + state["step"])
                state["dirty"] = True
            elif ch == "\x1b[B":
                for s in sel_items[state["sel"]][1]:
                    state["q"][s] = max(lo[s], state["q"][s] - state["step"])
                state["dirty"] = True
            elif ch == "[":
                state["step"] = max(0.005, state["step"] / 2); state["dirty"] = True
            elif ch == "]":
                state["step"] = min(1.0, state["step"] * 2); state["dirty"] = True
            elif ch == "0":
                for s in sel_items[state["sel"]][1]:
                    state["q"][s] = 0.0
                state["dirty"] = True
            elif ch in ("r", "R"):
                state["q"] = start_q.copy(); state["dirty"] = True
            elif ch in ("p", "P"):
                q = state["q"]
                print("\n  qpos = " + " ".join(f"{v:+.4f}" for v in q))
                print("  list = [" + ", ".join(f"{v:.4f}" for v in q) + "]\n")
                state["dirty"] = True

    threading.Thread(target=kb, daemon=True).start()
    print(f"Robot: {cfg.name}  J={J}  LEFT/RIGHT=joint  UP/DOWN=adjust  [ ]=step  0=zero  p=print  r=reset  q=quit")
    status()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not state["quit"]:
            data.qpos[:J] = state["q"]
            mujoco.mj_forward(model, data)
            if state["dirty"]:
                status()
                state["dirty"] = False
            viewer.sync()


if __name__ == "__main__":
    main()
