"""
Interactive hand viewer controlled by PCA eigengrasps.

Supports Shadow Hand (default), Allegro Hand (--robot allegro), and Leap Hand (--robot leap).

Controls (terminal, no need to press Enter):
  LEFT / RIGHT   previous / next eigengrasp knob
  UP / DOWN      increase / decrease selected coefficient
  T              toggle target real pose vs eigengrasp reconstruction  (Shadow only)
  0              reset all coefficients
  1..9           jump to eigengrasp 1..9
  Q / ESC        quit

Run from AIST-hand/:
    python models/grasp-intent-classification/scripts/mujoco_eigengrasp_viewer.py
    python models/grasp-intent-classification/scripts/mujoco_eigengrasp_viewer.py --robot allegro
    python models/grasp-intent-classification/scripts/mujoco_eigengrasp_viewer.py --coeffs +1.0 -0.5 +0.2
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


@dataclass
class RobotConfig:
    name: str
    hand_dir: Path
    root_body: str          # body whose quat/pos we patch to orient hand upright
    orig_body_tag: str      # exact string to find in XML
    new_body_tag: str       # replacement string
    qpos_dim: int           # full MuJoCo nq for this hand
    n_pad: int              # leading zeros before finger joints (wrist DOFs)
    n_joints: int           # eigengrasp joint count
    default_eigen: Path
    default_open_syn: Path
    default_close_syn: Path


_SHADOW = RobotConfig(
    name="shadow",
    hand_dir=_MENAGERIE / "shadow_hand",
    root_body="rh_forearm",
    orig_body_tag='<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
    new_body_tag='<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
    qpos_dim=24,
    n_pad=2,
    n_joints=22,
    default_eigen=(
        ROOT / "robot" / "hands" / "shadow_hand" / "datasets" / "processed"
        / "dexonomy_shadow_eigengrasps_balanced_sample.npz"
    ),
    default_open_syn=(
        ROOT / "robot" / "hands" / "shadow_hand" / "datasets" / "processed"
        / "synthetic_open_hand_shadow_qpos.npz"
    ),
    default_close_syn=(
        ROOT / "robot" / "hands" / "shadow_hand" / "datasets" / "processed"
        / "synthetic_close_hand_shadow_qpos.npz"
    ),
)

_ALLEGRO = RobotConfig(
    name="allegro",
    hand_dir=_MENAGERIE / "wonik_allegro",
    root_body="palm",
    orig_body_tag='<body name="palm" quat="0 1 0 1" childclass="allegro_right">',
    new_body_tag='<body name="palm" pos="0 0 0.05" quat="1 0 0 0" childclass="allegro_right">',
    qpos_dim=16,
    n_pad=0,
    n_joints=16,
    default_eigen=(
        ROOT / "robot" / "hands" / "allegro_hand" / "datasets" / "processed"
        / "eigengrasp_allegro.npz"
    ),
    default_open_syn=(
        ROOT / "robot" / "hands" / "allegro_hand" / "datasets" / "processed"
        / "synthetic_open_allegro.npz"
    ),
    default_close_syn=(
        ROOT / "robot" / "hands" / "allegro_hand" / "datasets" / "processed"
        / "synthetic_close_allegro.npz"
    ),
)

_LEAP = RobotConfig(
    name="leap",
    hand_dir=_MENAGERIE / "leap_hand",
    root_body="palm",
    orig_body_tag='<body name="palm" pos="0 0 0.1" quat="0 1 0 0">',
    new_body_tag='<body name="palm" pos="0 0 0.05" quat="0.707 0 0.707 0">',
    qpos_dim=16,
    n_pad=0,
    n_joints=16,
    default_eigen=(
        ROOT / "robot" / "hands" / "leap_hand" / "datasets" / "processed"
        / "eigengrasp_leap.npz"
    ),
    default_open_syn=None,
    default_close_syn=None,
)

ROBOTS: dict[str, RobotConfig] = {"shadow": _SHADOW, "allegro": _ALLEGRO, "leap": _LEAP}

POSE_ALPHA = 0.20


def build_scene(cfg: RobotConfig) -> Path:
    right_hand = cfg.hand_dir / "right_hand.xml"
    upright = cfg.hand_dir / ".right_hand_upright.xml"
    scene = cfg.hand_dir / ".scene_eigengrasp_viewer.xml"

    hand_text = right_hand.read_text().replace(cfg.orig_body_tag, cfg.new_body_tag, 1)
    upright.write_text(hand_text)
    scene.write_text(
        f"""<mujoco model="eigengrasp_viewer_{cfg.name}">
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
    result = {k: data[k] for k in data.files}
    if "coeff_p01" not in result or "coeff_p99" not in result:
        n = int(result["components_norm"].shape[0])
        result.setdefault("coeff_p01", np.full(n, -np.inf, dtype=np.float32))
        result.setdefault("coeff_p99", np.full(n,  np.inf, dtype=np.float32))
    return result


def reconstruct_qpos(
    eigen: dict[str, np.ndarray],
    coeffs: np.ndarray,
    n_knobs: int,
    n_pad: int,
) -> np.ndarray:
    mean = eigen["mean_norm"].astype(np.float64)
    comps = eigen["components_norm"][:n_knobs].astype(np.float64)
    low = eigen["joint_low"].astype(np.float64)
    high = eigen["joint_high"].astype(np.float64)

    q_norm = mean + coeffs[:n_knobs] @ comps
    q_norm = np.clip(q_norm, 0.0, 1.0)
    q_joints = q_norm * (high - low) + low
    if n_pad > 0:
        return np.concatenate([np.zeros(n_pad, dtype=np.float64), q_joints])
    return q_joints


def project_joints(eigen: dict[str, np.ndarray], q_joints: np.ndarray, n_knobs: int) -> np.ndarray:
    low = eigen["joint_low"].astype(np.float64)
    high = eigen["joint_high"].astype(np.float64)
    mean = eigen["mean_norm"].astype(np.float64)
    comps = eigen["components_norm"][:n_knobs].astype(np.float64)
    q_norm = (q_joints.astype(np.float64) - low) / (high - low)
    return (q_norm - mean) @ comps.T


def load_synthetic_pose(path: Path, n_joints: int, *, use_mean: bool = False) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    key = f"base_qpos{n_joints}"
    if not use_mean and key in data:
        qpos = data[key].astype(np.float64)
        if qpos.shape != (n_joints,):
            raise ValueError(f"Expected {key} shape [{n_joints}] in {path}, got {qpos.shape}")
        return qpos
    qkey = f"qpos{n_joints}"
    if qkey not in data:
        raise KeyError(f"Missing {qkey} in {path}")
    qpos = data[qkey].astype(np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != n_joints:
        raise ValueError(f"Expected {qkey} shape [N,{n_joints}] in {path}, got {qpos.shape}")
    return qpos.mean(axis=0)


def start_pose_coeffs(
    eigen: dict[str, np.ndarray],
    start_pose: str,
    n_knobs: int,
    n_joints: int,
    open_synthetic: Path,
    close_synthetic: Path,
) -> np.ndarray:
    if start_pose == "mean":
        return np.zeros(n_knobs, dtype=np.float64)
    if start_pose == "open":
        q = load_synthetic_pose(open_synthetic, n_joints)
    elif start_pose == "close":
        q = load_synthetic_pose(close_synthetic, n_joints)
    else:
        raise ValueError(f"Unknown start pose: {start_pose}")
    return project_joints(eigen, q, n_knobs)


def load_target_qpos_shadow(path: Path, row: int, qpos_key: str, n_pad: int) -> np.ndarray:
    """Load a Dexonomy .npy target pose (Shadow only — qpos shape [N,29])."""
    raw = np.load(path, allow_pickle=True).item()
    if qpos_key not in raw:
        available = ", ".join(sorted(str(key) for key in raw.keys()))
        raise KeyError(f"Missing {qpos_key}. Available keys: {available}")
    qpos = raw[qpos_key]
    if len(qpos.shape) != 2 or qpos.shape[1] != 29:
        raise ValueError(f"Expected {qpos_key} shape [N,29], got {qpos.shape}")
    if row < 0 or row >= qpos.shape[0]:
        raise IndexError(f"row={row} out of bounds for {qpos_key} with {qpos.shape[0]} rows")
    return np.concatenate([np.zeros(n_pad, dtype=np.float64), qpos[row, 7:].astype(np.float64)])


def print_status(
    idx: int,
    coeffs: np.ndarray,
    explained: np.ndarray,
    n_knobs: int,
    step: float,
    target_mode: bool,
    coeff_p01: np.ndarray,
    coeff_p99: np.ndarray,
) -> None:
    active = coeffs[:n_knobs]
    coeff_txt = " ".join(f"{i + 1}:{active[i]:+.2f}" for i in range(n_knobs))
    mode = "target-real" if target_mode else "reconstruction"
    lo, hi = coeff_p01[idx], coeff_p99[idx]
    print(
        f"\r  mode={mode:<14s} knob={idx + 1:02d}/{n_knobs}  step={step:.2f}  "
        f"PC{idx + 1} var={explained[idx] * 100:.1f}%  range=[{lo:+.2f},{hi:+.2f}]  [{coeff_txt}]      ",
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=list(ROBOTS), default="shadow",
                        help="Which hand to visualize (default: shadow)")
    parser.add_argument("--eigengrasps", type=Path, default=None,
                        help="Override eigengrasp .npz path")
    parser.add_argument("--n-knobs", type=int, default=9)
    parser.add_argument("--step", type=float, default=0.08)
    parser.add_argument("--start-pose", choices=("mean", "open", "close"), default="mean")
    parser.add_argument("--open-synthetic", type=Path, default=None)
    parser.add_argument("--close-synthetic", type=Path, default=None)
    parser.add_argument(
        "--coeffs",
        type=float,
        nargs="+",
        help="Initial absolute eigengrasp coefficients. Overrides --start-pose.",
    )
    # Shadow-only: load a real Dexonomy pose as target
    parser.add_argument("--target-npy", type=Path)
    parser.add_argument("--target-row", type=int, default=0)
    parser.add_argument("--target-qpos-key", default="grasp_qpos")
    args = parser.parse_args()

    cfg = ROBOTS[args.robot]
    eigen_path = args.eigengrasps or cfg.default_eigen
    open_syn = args.open_synthetic or cfg.default_open_syn
    close_syn = args.close_synthetic or cfg.default_close_syn

    if not eigen_path.exists():
        print(f"Eigengrasps not found: {eigen_path}")
        if args.robot == "allegro":
            print("Run build_allegro_eigengrasps.py first.")
        sys.exit(1)

    eigen = load_eigengrasps(eigen_path)
    n_available = int(eigen["components_norm"].shape[0])
    n_knobs = int(np.clip(args.n_knobs, 1, n_available))

    coeffs = start_pose_coeffs(
        eigen, args.start_pose, n_knobs, cfg.n_joints, open_syn, close_syn,
    )
    explained = eigen["explained_ratio"].astype(np.float64)

    if args.coeffs:
        n_given = min(len(args.coeffs), n_knobs)
        coeffs[:] = 0.0
        coeffs[:n_given] = np.asarray(args.coeffs[:n_given], dtype=np.float64)
        if len(args.coeffs) > n_knobs:
            print(f"Ignoring {len(args.coeffs) - n_knobs} coeffs beyond --n-knobs={n_knobs}.")

    target_qpos = None
    if args.target_npy:
        if args.robot != "shadow":
            print("--target-npy only supported for Shadow (Dexonomy format). Ignored.")
        else:
            target_qpos = load_target_qpos_shadow(
                args.target_npy, args.target_row, args.target_qpos_key, cfg.n_pad,
            )
            coeffs[:] = project_joints(eigen, target_qpos[cfg.n_pad:], n_knobs)
            print("Projected target coefficients:")
            print(" ".join(f"{i + 1}:{v:+.4f}" for i, v in enumerate(coeffs)))
    initial_coeffs = coeffs.copy()

    coeff_p01 = eigen["coeff_p01"].astype(np.float64)
    coeff_p99 = eigen["coeff_p99"].astype(np.float64)

    state = {"idx": 0, "quit": False, "dirty": True, "target_mode": False}

    print(f"Robot: {cfg.name}  |  Eigengrasps: {eigen_path}")
    print(f"Start pose: {args.start_pose}")
    print(
        f"Using {n_knobs} knobs. LEFT/RIGHT=select  UP/DOWN=change  "
        + ("T=target/recon  " if target_qpos is not None else "")
        + "0=reset  1..9=jump  Q=quit\n"
    )

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
                i = state["idx"]
                coeffs[i] = min(coeffs[i] + args.step, coeff_p99[i])
                state["dirty"] = True
            elif ch == "\x1b[B":
                i = state["idx"]
                coeffs[i] = max(coeffs[i] - args.step, coeff_p01[i])
                state["dirty"] = True
            elif ch in ("t", "T") and target_qpos is not None:
                state["target_mode"] = not state["target_mode"]
                state["dirty"] = True
            elif ch == "0":
                coeffs[:] = initial_coeffs
                state["dirty"] = True
            elif ch.isdigit() and ch != "0":
                target = int(ch) - 1
                if target < n_knobs:
                    state["idx"] = target
                    state["dirty"] = True

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()

    model = mujoco.MjModel.from_xml_path(str(build_scene(cfg)))
    data = mujoco.MjData(model)
    current_target = reconstruct_qpos(eigen, coeffs, n_knobs, cfg.n_pad)
    data.qpos[:cfg.qpos_dim] = current_target
    mujoco.mj_forward(model, data)
    print_status(state["idx"], coeffs, explained, n_knobs, args.step, state["target_mode"], coeff_p01, coeff_p99)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not state["quit"]:
            if state["dirty"]:
                if state["target_mode"] and target_qpos is not None:
                    current_target = target_qpos
                else:
                    current_target = reconstruct_qpos(eigen, coeffs, n_knobs, cfg.n_pad)
                print_status(state["idx"], coeffs, explained, n_knobs, args.step, state["target_mode"], coeff_p01, coeff_p99)
                state["dirty"] = False

            hand_qpos = data.qpos[:cfg.qpos_dim]
            hand_qpos[:] += POSE_ALPHA * (current_target - hand_qpos)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    state["quit"] = True
    print("\nDone.")


if __name__ == "__main__":
    main()
