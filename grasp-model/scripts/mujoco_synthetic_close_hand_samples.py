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

JOINTS22 = [
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

LOW22 = np.array(
    [
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -1.047,
        0.0,
        -0.209,
        -0.698,
        -0.2618,
    ],
    dtype=np.float64,
)
HIGH22 = np.array(
    [
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.7854,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        1.047,
        1.2217,
        0.209,
        0.698,
        1.5708,
    ],
    dtype=np.float64,
)

LONG_FINGERS = {
    "FF": (1, 2, 3),
    "MF": (5, 6, 7),
    "RF": (9, 10, 11),
    "LF": (14, 15, 16),
}
CURL_WEIGHTS = np.array([0.6, 1.0, 0.8], dtype=np.float64)
ABDUCTION_IDXS = np.array([0, 4, 8, 13, 19], dtype=np.int64)
LFJ5_IDX = 12
THUMB_IDXS = np.array([17, 18, 20, 21], dtype=np.int64)


def load_keyframe(name: str) -> np.ndarray:
    root = ET.parse(KEYFRAMES_XML).getroot()
    for key in root.findall(".//key"):
        if key.attrib.get("name") != name:
            continue
        qpos_text = key.attrib.get("qpos", "")
        qpos = np.fromstring(qpos_text, sep=" ", dtype=np.float64)
        if qpos.size != HAND_QPOS_DIM:
            raise ValueError(f"Expected {HAND_QPOS_DIM} qpos values for {name!r}, got {qpos.size}")
        return qpos
    raise KeyError(f"Missing keyframe {name!r} in {KEYFRAMES_XML}")


def build_scene() -> Path:
    upright_hand = SHADOW_DIR / ".right_hand_upright.xml"
    scene = SHADOW_DIR / ".scene_synthetic_close_hand.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright_hand.write_text(hand_text)
    scene.write_text(
        f"""<mujoco model="synthetic_close_hand_samples">
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


def generate_close_hand_samples(
    base_q22: np.ndarray,
    n: int,
    seed: int,
    curl_sigma: float,
    residual_sigma: float,
    thumb_sigma: float,
    abduction_sigma: float,
    lfj5_sigma: float,
    max_abs_delta: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.repeat(base_q22[None, :], n, axis=0)

    for sample in samples:
        delta = np.zeros_like(base_q22)

        for idxs in LONG_FINGERS.values():
            curl = rng.normal(0.0, curl_sigma)
            delta[list(idxs)] += CURL_WEIGHTS * curl
            delta[list(idxs)] += rng.normal(0.0, residual_sigma, size=3)

        delta[THUMB_IDXS] += rng.normal(0.0, thumb_sigma, size=THUMB_IDXS.size)
        delta[ABDUCTION_IDXS] += rng.normal(0.0, abduction_sigma, size=ABDUCTION_IDXS.size)
        delta[LFJ5_IDX] += rng.normal(0.0, lfj5_sigma)
        delta += rng.normal(0.0, residual_sigma * 0.5, size=base_q22.size)

        delta = np.clip(delta, -max_abs_delta, max_abs_delta)
        sample[:] = np.clip(base_q22 + delta, LOW22, HIGH22)

    return samples


def print_sample_delta(idx: int, base_q22: np.ndarray, q22: np.ndarray) -> None:
    delta = q22 - base_q22
    order = np.argsort(np.abs(delta))[::-1][:6]
    terms = ", ".join(f"{JOINTS22[i]}:{delta[i]:+.4f}" for i in order)
    print(f"\r  sample={idx + 1:02d}  max_delta={np.max(np.abs(delta)):.4f}  {terms}      ", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--curl-sigma", type=float, default=0.02)
    parser.add_argument("--residual-sigma", type=float, default=0.004)
    parser.add_argument("--thumb-sigma", type=float, default=0.015)
    parser.add_argument("--abduction-sigma", type=float, default=0.003)
    parser.add_argument("--lfj5-sigma", type=float, default=0.003)
    parser.add_argument("--max-abs-delta", type=float, default=0.04)
    parser.add_argument("--print-q22", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help="Only generate/print samples; do not open MuJoCo.")
    args = parser.parse_args()

    base_qpos = load_keyframe("close hand")
    base_q22 = base_qpos[2:].copy()
    samples22 = generate_close_hand_samples(
        base_q22=base_q22,
        n=args.n,
        seed=args.seed,
        curl_sigma=args.curl_sigma,
        residual_sigma=args.residual_sigma,
        thumb_sigma=args.thumb_sigma,
        abduction_sigma=args.abduction_sigma,
        lfj5_sigma=args.lfj5_sigma,
        max_abs_delta=args.max_abs_delta,
    )
    samples24 = np.concatenate([np.repeat(base_qpos[None, :2], args.n, axis=0), samples22], axis=1)

    if args.print_q22:
        for idx, q22 in enumerate(samples22):
            print(f"sample {idx + 1:02d}: " + " ".join(f"{value:+.5f}" for value in q22))
    if args.no_viewer:
        for idx, q22 in enumerate(samples22):
            print_sample_delta(idx, base_q22, q22)
            print()
        return

    print("Synthetic close-hand samples from MuJoCo keyframe 'close hand'.")
    print("LEFT/RIGHT=previous/next  Q/ESC=quit")

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    state = {"idx": 0, "quit": False, "dirty": True}
    target = samples24[state["idx"]].copy()
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
        print_sample_delta(state["idx"], base_q22, samples22[state["idx"]])
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                target = samples24[state["idx"]].copy()
                print_sample_delta(state["idx"], base_q22, samples22[state["idx"]])
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
