"""
GraphGrasp — grasp-app entry point.

Runs the real-time inference loop:
  PerceptionBackend → ToGraph → GCN → VotingWindow → GraspToken → Robot

To use a different perception backend, swap MediaPipeBackend for your own
implementation of PerceptionBackend (see perception/mediapipe_backend.py).
"""

import torch
from grasp_gcn import get_network, ToGraph, GraspToken, VotingWindow

from perception.mediapipe_backend import MediaPipeBackend

# TODO: import robot adapter from grasp-robot
# from grasp_robot import YAMLRobotAdapter

MODEL_PATH = "models/best_model.pth"
NUM_CLASSES = 28
N_VOTES = 5
NUM_FEATURES = 4  # [x, y, z, theta_flex]
NETWORK_TYPE = "GCN_CAM_8_8_16_16_32"


def _to_probs(head_a: torch.Tensor) -> torch.Tensor:
    """Support both log-softmax outputs and raw logits."""
    lse = torch.logsumexp(head_a, dim=1)
    if torch.allclose(lse, torch.zeros_like(lse), atol=1e-4):
        return head_a.exp()
    return torch.softmax(head_a, dim=1)


def _parse_model_output(output):
    """Return (head_a, head_b_or_none) from single-head or multi-head models."""
    if isinstance(output, torch.Tensor):
        return output, None

    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]

    if isinstance(output, dict):
        head_a = (
            output.get("head_a")
            or output.get("logits")
            or output.get("log_probs")
        )
        head_b = (
            output.get("head_b")
            or output.get("synergy")
            or output.get("synergy_coeffs")
        )
        if head_a is None:
            raise ValueError("Model dict output missing Head A logits/log_probs.")
        return head_a, head_b

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def main():
    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        model = get_network(
            NETWORK_TYPE,
            NUM_FEATURES,
            NUM_CLASSES,
            use_cmc_angle=True,
        ).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"[main] Inference model loaded: {MODEL_PATH}")
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
    backend = MediaPipeBackend()
    # adapter = YAMLRobotAdapter("../grasp-robot/grasp_configs/shadow_hand.yaml")

    # --- Loop ---
    while backend.is_ready():
        landmarks = backend.get_landmarks()
        status_text = "Perception: waiting for hand"
        token = None
        if landmarks is not None:
            if model is None:
                status_text = "Inference: no model loaded"
            else:
                data = to_graph(landmarks).to(device)
                with torch.no_grad():
                    output = model(data)
                head_a, head_b = _parse_model_output(output)

                probs = _to_probs(head_a)
                class_id = probs.argmax().item()
                confidence = probs.max().item()
                class_name = str(class_id)  # TODO: map to GRASP taxonomy name

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

        if not backend.render(token, status_text=status_text):
            break

    backend.release()


if __name__ == "__main__":
    main()
