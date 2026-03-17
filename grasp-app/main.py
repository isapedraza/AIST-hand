"""
GraphGrasp — grasp-app entry point.

Runs the real-time inference loop:
  PerceptionBackend → ToGraph → GCN → VotingWindow → GraspToken → Robot

To use a different perception backend, swap MediaPipeBackend for your own
implementation of PerceptionBackend (see perception/mediapipe_backend.py).
"""

import argparse
import os
from pathlib import Path

import torch
from grasp_gcn import get_network, ToGraph, GraspToken, VotingWindow

from inference_runtime import OpenHandLatch, parse_model_output, to_probs
from model_variant import resolve_model_spec
from perception.mediapipe_backend import MediaPipeBackend

# TODO: import robot adapter from grasp-robot
# from grasp_robot import YAMLRobotAdapter

N_VOTES = 5
NUM_FEATURES = 4  # [x, y, z, theta_flex]
NETWORK_TYPE = "GCN_CAM_8_8_16_16_32"


def _resolve_camera_source() -> int | str:
    raw = os.getenv("GRAPHGRASP_CAMERA_INDEX", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GraphGrasp perception app.")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera index (0,1,2,...) or stream URL. Overrides GRAPHGRASP_CAMERA_INDEX.",
    )
    return parser.parse_args()


def _get_camera_source(args: argparse.Namespace) -> int | str:
    if args.camera is None:
        return _resolve_camera_source()
    try:
        return int(args.camera)
    except ValueError:
        return args.camera

def main():
    args = _parse_args()
    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = resolve_model_spec()
    model_path = Path(spec["model_path"])
    num_classes = spec["num_classes"]
    class_names = spec["class_names"]
    model_lines = [
        f"Model: {spec['variant']}",
        f"Classes: {num_classes}",
    ]
    camera_source = _get_camera_source(args)
    model_lines.append(f"Camera: {camera_source}")
    model = None
    try:
        model = get_network(
            NETWORK_TYPE,
            NUM_FEATURES,
            num_classes,
            use_cmc_angle=True,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"[main] Inference model loaded: variant={spec['variant']} path={model_path}")
    except Exception as exc:
        model = None
        print(f"[main] Perception-only mode (model unavailable): {exc}")

    to_graph = ToGraph(
        features="xyz",
        make_undirected=True,
        add_joint_angles=True,
        add_cmc_angle=True,
    )
    window = VotingWindow(n=N_VOTES)
    open_hand_latch = OpenHandLatch()
    backend = MediaPipeBackend(camera_index=camera_source)
    # adapter = YAMLRobotAdapter("../grasp-robot/grasp_configs/shadow_hand.yaml")

    # --- Loop ---
    while backend.is_ready():
        landmarks = backend.get_landmarks()
        status_text = backend.camera_status() or "Perception: waiting for hand"
        token = None
        if landmarks is not None:
            if model is None:
                status_text = "Inference: no model loaded"
            elif open_hand_latch.update(landmarks):
                window.reset()
                token = GraspToken(
                    class_id=None,
                    class_name="Open Hand",
                    confidence=1.0,
                    confirmed=True,
                    observed_handedness=landmarks.get("handedness"),
                )
                status_text = "Inference: open hand override"
            else:
                data = to_graph(landmarks).to(device)
                with torch.no_grad():
                    output = model(data)
                head_a, head_b = parse_model_output(output)

                probs = to_probs(head_a)
                class_id = probs.argmax().item()
                confidence = probs.max().item()
                class_name = class_names.get(class_id, str(class_id))

                confirmed = window.update(class_id, confidence)
                synergy_coeffs = []
                if head_b is not None:
                    if isinstance(head_b, torch.Tensor):
                        synergy_coeffs = head_b.detach().view(head_b.size(0), -1)[0].cpu().tolist()
                    else:
                        synergy_coeffs = list(head_b)

                token = GraspToken(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    synergy_coeffs=synergy_coeffs,
                    confirmed=confirmed,
                    observed_handedness=landmarks.get("handedness"),
                )
                # adapter.execute(token)
                if not confirmed:
                    status_text = "Inference: waiting for stable vote"
                else:
                    status_text = "Inference: confirmed"
        elif model is None:
            status_text = "Inference: no model loaded"

        if not backend.render(token, status_text=status_text, model_lines=model_lines):
            break

    backend.release()


if __name__ == "__main__":
    main()
