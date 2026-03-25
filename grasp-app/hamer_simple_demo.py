"""
Demo simple: HaMeR + GCN.

Muestra camara con landmarks y al lado la prediccion de clase de agarre.
Sin MuJoCo, sin voting window, sin filtros.

Uso:
  python hamer_simple_demo.py --url https://xxxx.trycloudflare.com --camera 1
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch


from grasp_gcn import ToGraph, get_network
from inference_runtime import parse_model_output, to_probs
from model_variant import resolve_model_spec
from perception.hamer_backend import HaMeRBackend


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec        = resolve_model_spec()
    num_classes = spec["num_classes"]
    class_names = spec["class_names"]

    model = get_network("GCN_CAM_8_8_16_16_32", spec["num_node_features"], num_classes, use_cmc_angle=spec.get("use_cmc_angle", True)).to(device)
    model.load_state_dict(torch.load(spec["model_path"], map_location=device, weights_only=True))
    model.eval()
    print(f"Modelo cargado: {spec['variant']} | {num_classes} clases | {spec['num_node_features']} features/nodo")

    to_graph = ToGraph(make_undirected=True, **spec["tograph_kwargs"])

    backend = HaMeRBackend(url=args.url, camera_index=args.camera)
    if backend.startup_error():
        print(backend.startup_error())
        return

    WINDOW = "HaMeR + GCN"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    class_name = "esperando..."
    confidence = 0.0

    while backend.is_ready():
        landmarks = backend.get_landmarks()

        if landmarks is not None:
            try:
                data = to_graph(landmarks).to(device)
                with torch.no_grad():
                    output = model(data)
                head_a, _ = parse_model_output(output)
                probs      = to_probs(head_a)
                class_id   = probs.argmax().item()
                confidence = probs.max().item()
                class_name = class_names.get(class_id, str(class_id))
            except Exception as e:
                print(f"Inference error: {e}")

        # Frame con landmarks
        frame = backend._frame_bgr.copy() if backend._frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        if backend.mirror_display:
            frame = cv2.flip(frame, 1)

        if backend._last_result and backend._last_result.multi_hand_landmarks:
            hand_lm = backend._last_result.multi_hand_landmarks[0]
            if backend.mirror_display:
                hand_lm = type(hand_lm)()
                hand_lm.CopyFrom(backend._last_result.multi_hand_landmarks[0])
                for lm in hand_lm.landmark:
                    lm.x = 1.0 - lm.x
            backend._mp_draw.draw_landmarks(frame, hand_lm, backend._mp_hands.HAND_CONNECTIONS)

        if backend._last_bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = backend._last_bbox
            bx1 = int((1.0 - x2) * w) if backend.mirror_display else int(x1 * w)
            bx2 = int((1.0 - x1) * w) if backend.mirror_display else int(x2 * w)
            color = (0, 255, 0) if backend._last_handedness == "Right" else (255, 0, 0)
            cv2.rectangle(frame, (bx1, int(y1 * h)), (bx2, int(y2 * h)), color, 2)

        # Panel lateral
        h, w   = frame.shape[:2]
        panel  = np.zeros((h, 320, 3), dtype=np.uint8)
        lines  = [
            "HaMeR + GCN",
            f"Hand: {backend._last_handedness}",
            "",
            "Clase:",
            class_name,
            "",
            f"Conf: {confidence:.3f}",
        ]
        y = 40
        for i, line in enumerate(lines):
            color = (235, 235, 235)
            scale = 0.65
            thick = 1
            if i == 0:
                color = (100, 200, 255)
                scale = 0.85
                thick = 2
            elif i == 4:
                color = (80, 220, 120)
                scale = 0.75
                thick = 2
            cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y += 38 if i == 0 else 32

        canvas = np.hstack([frame, panel])
        cv2.imshow(WINDOW, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    backend.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
