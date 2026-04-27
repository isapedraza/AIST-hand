"""
Visualize human-to-robot retargeting in MuJoCo.

Loads real HOGraspNet Dong quaternions, retargets to Shadow Hand via the
Stage 1 checkpoint, and displays the result cycling through frames.

Usage:
    python viz_retarget.py \\
        --ckpt  checkpoints/stage1_cam_5000.pt \\
        --csv   data/processed/hograspnet_dong.csv \\
        --grasp 0 \\
        --n     16 \\
        --fps   2
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
ROOT    = SCRIPTS.parent
sys.path.insert(0, str(ROOT / "src/cross_emb"))

from human_modules  import HumanEncoder_E_h
from shared_modules import SharedDecoder_D_X
from robot_modules  import RobotDecoder_D_r

SHADOW_DIR   = Path(__file__).resolve().parents[2] / "third_party/mujoco_menagerie/shadow_hand"
RIGHT_HAND   = SHADOW_DIR / "right_hand.xml"
HAND_QPOS_DIM = 24

JOINT_LOWER = np.array([
    -0.524, -0.698,
    -0.349, -0.262, 0.000, 0.000,
    -0.349, -0.262, 0.000, 0.000,
    -0.349, -0.262, 0.000, 0.000,
     0.000, -0.349, -0.262, 0.000, 0.000,
    -1.047,  0.000, -0.209, -0.698, -0.262,
], dtype=np.float32)

JOINT_UPPER = np.array([
     0.175,  0.489,
     0.349,  1.571, 1.571, 1.571,
     0.349,  1.571, 1.571, 1.571,
     0.349,  1.571, 1.571, 1.571,
     0.785,  0.349,  1.571, 1.571, 1.571,
     1.047,  1.222,  0.209,  0.698,  1.571,
], dtype=np.float32)


def build_scene() -> Path:
    upright_hand = SHADOW_DIR / ".right_hand_upright.xml"
    scene        = SHADOW_DIR / ".scene_viz_retarget.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright_hand.write_text(hand_text)

    scene.write_text(
        f"""<mujoco model="retarget_viz">
  <include file="{upright_hand.name}"/>
  <statistic extent="0.3" center="0 0 0.2"/>
  <visual>
    <global azimuth="145" elevation="-18"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5"/>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
  </worldbody>
</mujoco>"""
    )
    return scene


def load_models(ckpt_path: Path):
    ck     = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    z_dim  = ck["E_h"]["proj.weight"].shape[0]
    n_j    = ck["D_r"]["fc.weight"].shape[0]

    E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=z_dim).eval()
    D_X = SharedDecoder_D_X(z_dim=z_dim, shared_dim=1024).eval()
    D_r = RobotDecoder_D_r(n_joints=n_j, shared_dim=1024).eval()

    E_h.load_state_dict(ck["E_h"])
    D_X.load_state_dict(ck["D_X"])
    D_r.load_state_dict(ck["D_r"])
    print(f"Checkpoint step={ck['step']}  z_dim={z_dim}  n_joints={n_j}")
    return E_h, D_X, D_r


def load_quats(csv_path: Path, grasp_type: int, n: int) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    available = sorted(df["grasp_type"].unique().tolist())
    df = df[df["grasp_type"] == grasp_type].head(n)
    if len(df) == 0:
        raise ValueError(f"grasp_type={grasp_type} not found. Available: {available}")
    q_cols = [f"q{i}_{c}" for i in range(1, 21) for c in ("w", "x", "y", "z")]
    quats  = torch.tensor(df[q_cols].values, dtype=torch.float32).reshape(-1, 20, 4)
    print(f"Loaded {len(quats)} frames  grasp_type={grasp_type}")
    return quats


@torch.no_grad()
def retarget(quats_h: torch.Tensor, E_h, D_X, D_r) -> np.ndarray:
    q = D_r(D_X(E_h(quats_h))).numpy()             # [N, 24]
    return np.clip(q, JOINT_LOWER, JOINT_UPPER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default="checkpoints/stage1_cam_5000.pt")
    parser.add_argument("--csv",   default="data/processed/hograspnet_dong.csv")
    parser.add_argument("--grasp", type=int, default=14)
    parser.add_argument("--n",     type=int, default=16)
    parser.add_argument("--fps",   type=float, default=2.0)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    E_h, D_X, D_r = load_models(ckpt_path)
    quats_h        = load_quats(csv_path, args.grasp, args.n)
    poses          = retarget(quats_h, E_h, D_X, D_r)   # [N, 24]

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data  = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = 0
        while viewer.is_running():
            data.qpos[:HAND_QPOS_DIM] = poses[i % len(poses)]
            mujoco.mj_forward(model, data)
            viewer.sync()
            i += 1
            time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
