"""
Postural Control for Shadow Hand using abl04 GNN classifier.

Analog of Segil et al. (2014) C3 postural controller:
  - EMG → 2D PC domain → JAT → joint angles   (Segil)
  - XYZ → GNN → 28 probs → top-2 interpolation → joint angles  (this work)

Usage:
    python postural_control.py --xyz "0.1,0.2,...,0.3"   # 63 comma-separated XYZ values
    python postural_control.py --demo                      # runs on random test frames
"""

import argparse
import torch
import numpy as np
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "grasp-model" / "src"))

from grasp_gcn.network.gcn_network import GCN_CAM_8_8_16_16_32
from grasp_gcn.dataset.grasps import GRASP_CLASS_NAMES
from grasp_gcn.transforms.tograph import ToGraph

MODEL_PATH  = Path(__file__).parent.parent / "grasp-app" / "models" / "best_model_run_abl04_xyz_ahg_flex.pth"
YAML_PATH   = Path(__file__).parent / "grasp_configs" / "shadow_hand_canonical_v5_grasp.yaml"
N_CLASSES   = 28
IN_DIM      = 24   # xyz + AHG + flex

# Map local class id (0-27) → class name
LOCAL_TO_NAME = GRASP_CLASS_NAMES  # {0: "Large Diameter", ...}
NAME_TO_LOCAL = {name: local_id for local_id, name in LOCAL_TO_NAME.items()}


def load_canonical_poses(yaml_path: Path) -> dict:
    """
    Returns {local_class_id: np.array [24]} using pose_close as canonical grasp pose.
    Missing classes fall back to hand-flat (zeros).
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Build name → pose_close mapping from YAML
    name_to_qpos = {}
    for k, v in data.items():
        if k == '_meta':
            continue
        name_to_qpos[v['class_name']] = np.array(v['pose_close'], dtype=np.float32)

    # Map local ids → qpos
    hand_flat = np.zeros(24, dtype=np.float32)
    canonical = {}
    for local_id, name in LOCAL_TO_NAME.items():
        if name in name_to_qpos:
            canonical[local_id] = name_to_qpos[name]
        else:
            canonical[local_id] = hand_flat
            print(f"  WARNING: no canonical pose for class {local_id} '{name}', using hand-flat")
    return canonical


def load_model(model_path: Path, device: torch.device):
    model = GCN_CAM_8_8_16_16_32(numFeatures=IN_DIM, numClasses=N_CLASSES)
    ckpt  = torch.load(model_path, map_location=device, weights_only=True)
    # handle bare state_dict or wrapped checkpoint
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def xyz_to_qpos(
    xyz: np.ndarray,          # [21, 3] or [63]
    model: torch.nn.Module,
    canonical: dict,
    to_graph: ToGraph,
    device: torch.device,
    top_k: int = 2,
) -> tuple[np.ndarray, list]:
    """
    xyz → GNN → top-k probabilities → interpolated qpos.

    Returns:
        qpos   : [24] joint angles for Shadow Hand
        top2   : [(class_id, class_name, prob), ...]
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(21, 3)

    # Build sample dict expected by ToGraph
    JOINT_KEYS = [
        'WRIST',
        'THUMB_CMC','THUMB_MCP','THUMB_IP','THUMB_TIP',
        'INDEX_FINGER_MCP','INDEX_FINGER_PIP','INDEX_FINGER_DIP','INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP','MIDDLE_FINGER_PIP','MIDDLE_FINGER_DIP','MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP','RING_FINGER_PIP','RING_FINGER_DIP','RING_FINGER_TIP',
        'PINKY_MCP','PINKY_PIP','PINKY_DIP','PINKY_TIP',
    ]
    sample = {key: xyz[i].tolist() for i, key in enumerate(JOINT_KEYS)}
    sample['handedness'] = 0   # right hand
    sample['grasp_type'] = 0   # placeholder, not used for features

    # Build PyG graph (same pipeline as training)
    graph = to_graph(sample)
    graph = graph.to(device)

    with torch.no_grad():
        log_probs = model(graph)      # model takes PyG Data object
        probs = log_probs.exp().squeeze(0)   # [28]

    top_vals, top_idx = probs.topk(top_k)
    top_vals = top_vals.cpu().numpy()
    top_idx  = top_idx.cpu().numpy()

    # Normalize top-k weights
    w = top_vals / top_vals.sum()

    # Weighted interpolation of canonical poses (Segil JAT equivalent)
    qpos = sum(w[i] * canonical[int(top_idx[i])] for i in range(top_k))

    top_info = [(int(top_idx[i]), LOCAL_TO_NAME[int(top_idx[i])], float(top_vals[i]))
                for i in range(top_k)]
    return qpos.astype(np.float32), top_info


class PosturalController:
    """
    Postural controller with sliding-window probability smoothing.

    Each frame: model outputs probs [28]. Window keeps last `window_size`
    distributions and averages them before computing top-k → qpos.
    Separates perception (per-frame) from control (smoothed).

    Usage:
        pc = PosturalController(window_size=8)
        qpos, info = pc(xyz_21x3)
        pc.reset()   # call when hand is lost
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        yaml_path=YAML_PATH,
        device=None,
        window_size: int = 8,
        blocked_class_names: set[str] | None = None,
    ):
        self.device      = device or torch.device("cpu")
        self.window_size = window_size
        self._window     = []   # deque of probs np.array [28]
        blocked_class_names = blocked_class_names or set()
        self.blocked_class_ids = np.array(
            sorted(
                NAME_TO_LOCAL[name]
                for name in blocked_class_names
                if name in NAME_TO_LOCAL
            ),
            dtype=np.int64,
        )
        self.blocked_class_names = {
            LOCAL_TO_NAME[idx] for idx in self.blocked_class_ids.tolist()
        }

        print("Loading canonical poses...")
        self.canonical = load_canonical_poses(yaml_path)
        print("Loading abl04 model...")
        self.model     = load_model(model_path, self.device)
        self.to_graph  = ToGraph(
            add_joint_angles=True,
            add_ahg_angles=True,
            add_ahg_distances=True,
        )
        print("PosturalController ready.")

    def reset(self):
        self._window.clear()

    def __call__(self, xyz: np.ndarray, top_k: int = 2):
        xyz = np.asarray(xyz, dtype=np.float32).reshape(21, 3)

        # Build graph and get per-frame probs
        JOINT_KEYS = [
            'WRIST',
            'THUMB_CMC','THUMB_MCP','THUMB_IP','THUMB_TIP',
            'INDEX_FINGER_MCP','INDEX_FINGER_PIP','INDEX_FINGER_DIP','INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP','MIDDLE_FINGER_PIP','MIDDLE_FINGER_DIP','MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP','RING_FINGER_PIP','RING_FINGER_DIP','RING_FINGER_TIP',
            'PINKY_MCP','PINKY_PIP','PINKY_DIP','PINKY_TIP',
        ]
        sample = {key: xyz[i].tolist() for i, key in enumerate(JOINT_KEYS)}
        sample['handedness'] = 0
        sample['grasp_type'] = 0

        graph = self.to_graph(sample).to(self.device)
        with torch.no_grad():
            log_probs = self.model(graph)
            probs = log_probs.exp().squeeze(0).cpu().numpy()   # [28]

        # Sliding window: add new frame, drop oldest if full
        self._window.append(probs)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        # Average over window
        avg_probs = np.mean(self._window, axis=0)   # [28]

        # Remove blocked classes before selecting top-k.
        effective_probs = avg_probs.copy()
        if self.blocked_class_ids.size > 0:
            effective_probs[self.blocked_class_ids] = 0.0
        total = float(effective_probs.sum())
        if total <= 0.0:
            return np.zeros(24, dtype=np.float32), []
        effective_probs /= total

        # Top-k from filtered/smoothed distribution
        top_idx  = np.argsort(effective_probs)[::-1][:top_k]
        top_vals = effective_probs[top_idx]
        w        = top_vals / top_vals.sum()

        qpos     = sum(w[i] * self.canonical[int(top_idx[i])] for i in range(top_k))
        top_info = [(int(top_idx[i]), LOCAL_TO_NAME[int(top_idx[i])], float(top_vals[i]))
                    for i in range(top_k)]
        return qpos.astype(np.float32), top_info


def demo(n_frames: int = 5):
    """Run on random frames from test cache to verify pipeline."""
    import torch
    from torch_geometric.data import Data

    pc = PosturalController()

    cache = torch.load(
        Path(__file__).parents[1] / "grasp-model/data/processed/hograspnet_test_c28_cmc_nocmc_ahga_ahgd.pt",
        weights_only=False
    )
    data, slices = cache
    N = data.y.shape[0]
    x_all = data.x.view(N, 21, IN_DIM)
    y_all = data.y

    indices = torch.randperm(N)[:n_frames].tolist()
    print(f"\n{'='*60}")
    print(f"Demo: {n_frames} random test frames")
    print(f"{'='*60}")

    for i in indices:
        features = x_all[i].numpy()   # [21, 24] — includes xyz as first 3 cols
        xyz = features[:, :3]          # [21, 3] raw xyz
        true_label = int(y_all[i])

        qpos, top2 = pc(xyz)

        print(f"\nFrame {i} | True: {LOCAL_TO_NAME[true_label]}")
        for rank, (cid, cname, prob) in enumerate(top2):
            print(f"  Top-{rank+1}: [{cid:2d}] {cname:<25} p={prob:.3f}")
        print(f"  qpos (first 6): {qpos[:6].round(3)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Run on random test frames")
    p.add_argument("--xyz",  type=str, default=None, help="63 comma-separated XYZ values")
    p.add_argument("--n",    type=int, default=5,    help="Number of demo frames")
    args = p.parse_args()

    if args.demo:
        demo(args.n)
    elif args.xyz:
        xyz = np.array([float(v) for v in args.xyz.split(",")]).reshape(21, 3)
        pc  = PosturalController()
        qpos, top2 = pc(xyz)
        print("Top-2 predictions:")
        for cid, cname, prob in top2:
            print(f"  [{cid}] {cname}: {prob:.3f}")
        print(f"qpos: {qpos.round(4)}")
    else:
        demo()
