"""
GraphGrasp — grasp-app entry point.

Runs the real-time inference loop:
  PerceptionBackend → ToGraph → GCN → VotingWindow → AperturaCalculator → GraspToken → Robot

To use a different perception backend, swap MediaPipeBackend for your own
implementation of PerceptionBackend (see perception/mediapipe_backend.py).
"""

import torch
from grasp_gcn import get_network, ToGraph, GraspToken, VotingWindow

from perception.mediapipe_backend import MediaPipeBackend
from perception.apertura import AperturaCalculator

# TODO: import robot adapter from grasp-robot
# from grasp_robot import YAMLRobotAdapter

MODEL_PATH = "models/best_model.pth"
NUM_CLASSES = 28
N_VOTES = 5


def main():
    # --- Setup ---
    model = get_network("GCN_8_8_16_16_32", num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    to_graph = ToGraph()
    window = VotingWindow(n=N_VOTES)
    backend = MediaPipeBackend()
    # adapter = YAMLRobotAdapter("../grasp-robot/grasp_configs/shadow_hand.yaml")

    locked_class_id = None
    locked_class_name = None

    # --- Loop ---
    while backend.is_ready():
        landmarks = backend.get_landmarks()
        if landmarks is None:
            continue

        data = to_graph(landmarks)

        with torch.no_grad():
            logits = model(data)

        probs = logits.exp()
        class_id = probs.argmax().item()
        confidence = probs.max().item()
        class_name = str(class_id)  # TODO: map to GRASP taxonomy name

        confirmed = window.update(class_id, confidence)

        if confirmed:
            locked_class_id = class_id
            locked_class_name = class_name

        token = None
        if locked_class_id is not None:
            apertura = AperturaCalculator.compute(landmarks, locked_class_id)
            token = GraspToken(
                class_id=locked_class_id,
                class_name=locked_class_name,
                confidence=confidence,
                apertura=apertura,
            )
            # adapter.execute(token)

        backend.render(token)  # each backend decides how to visualize, or does nothing

    backend.release()


if __name__ == "__main__":
    main()
