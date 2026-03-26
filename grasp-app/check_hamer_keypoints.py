"""
Verifica que los keypoints de HaMeR llegan correctamente normalizados.

Toma un frame de la webcam, lo manda a HaMeR, e imprime:
- Keypoints raw (metros, root-relative)
- Keypoints normalizados (distancia WRIST->INDEX_MCP = 0.1)
- Distancia WRIST->INDEX_MCP antes y despues

Uso:
  python check_hamer_keypoints.py --url https://xxxx.trycloudflare.com --camera 1
"""

import argparse
import cv2
import numpy as np
from perception.hamer_backend import HaMeRBackend

JOINTS = HaMeRBackend.JOINTS

parser = argparse.ArgumentParser()
parser.add_argument("--url",    type=str, required=True)
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--frames", type=int, default=5, help="Cuantos frames capturar (default: 5)")
args = parser.parse_args()

backend = HaMeRBackend(url=args.url, camera_index=args.camera)
if backend.startup_error():
    print(backend.startup_error())
    exit(1)

print("Esperando mano... (presiona Q para salir)\n")

captured = 0
while captured < args.frames and backend.is_ready():
    # get_landmarks ya aplica normalizacion internamente
    # Para ver raw necesitamos interceptar antes de _normalize_geometric
    ok, frame_bgr = backend._cap.read()
    if not ok:
        continue

    backend._frame_bgr = frame_bgr
    import mediapipe as mp
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = backend._hands.process(frame_rgb)

    cv2.imshow("check - presiona Q para salir", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if not result.multi_hand_landmarks:
        continue

    hand = result.multi_hand_landmarks[0]
    if result.multi_handedness:
        label = result.multi_handedness[0].classification[0].label
        handedness = "Right" if label == "Left" else "Left"
    else:
        handedness = "Unknown"

    is_right = handedness == "Right"
    bbox = backend._get_bbox(hand)
    crop = backend._crop_hand(frame_bgr, bbox)
    if crop.size == 0:
        continue

    # Raw desde HaMeR (sin normalizar)
    pts_raw = backend._send_crop(crop, is_right)
    if pts_raw is None:
        print("HaMeR no respondio, reintentando...")
        continue

    # Normalizado
    pts_norm = HaMeRBackend._normalize_geometric(pts_raw)
    if handedness == "Left":
        pts_norm = pts_norm.copy()
        pts_norm[:, 0] *= -1.0

    # Calcular distancias
    dist_raw  = np.linalg.norm(pts_raw[HaMeRBackend.INDEX_MCP_IDX] - pts_raw[HaMeRBackend.WRIST_IDX])
    dist_norm = np.linalg.norm(pts_norm[HaMeRBackend.INDEX_MCP_IDX])

    captured += 1
    print(f"{'='*60}")
    print(f"Frame {captured} | Handedness: {handedness}")
    print(f"\n[RAW - metros desde HaMeR]")
    print(f"  WRIST:          {pts_raw[0]}")
    print(f"  INDEX_MCP:      {pts_raw[5]}")
    print(f"  THUMB_TIP:      {pts_raw[4]}")
    print(f"  MIDDLE_MCP:     {pts_raw[9]}")
    print(f"  dist WRIST->INDEX_MCP: {dist_raw:.4f} m")

    print(f"\n[NORMALIZADO - root-relative + escala]")
    print(f"  WRIST:          {pts_norm[0]}  <- debe ser [0,0,0]")
    print(f"  INDEX_MCP:      {pts_norm[5]}")
    print(f"  THUMB_TIP:      {pts_norm[4]}")
    print(f"  MIDDLE_MCP:     {pts_norm[9]}")
    print(f"  dist WRIST->INDEX_MCP: {dist_norm:.4f}  <- debe ser 0.1000")

    print(f"\n[TODOS LOS JOINTS NORMALIZADOS]")
    for i, name in enumerate(JOINTS):
        print(f"  {i:2d} {name:<25} {pts_norm[i]}")

backend.release()
cv2.destroyAllWindows()
