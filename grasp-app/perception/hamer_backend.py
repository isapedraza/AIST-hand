"""
HaMeR remote inference backend for GraphGrasp.

Pipeline:
  Webcam -> MediaPipe (bbox + landmarks) -> crop 256x256 -> POST a Colab (hilo separado)
  -> keypoints 3D (21, 3) -> root-relative + scale normalization -> dict compatible con ToGraph

La webcam y MediaPipe corren sin bloquearse. Los requests a Colab van en un hilo
separado — get_landmarks() retorna los ultimos keypoints recibidos sin esperar.

Drop-in replacement para MediaPipeBackend.
"""

import os
import threading

import cv2
import mediapipe as mp
import numpy as np
import requests

from grasp_gcn import PerceptionBackend


class HaMeRBackend(PerceptionBackend):
    """HaMeR remote inference implementation of PerceptionBackend.

    Preprocessing contract (identica a MediaPipeBackend):
    1) Root-relative coordinates (subtract WRIST).
    2) Scale normalization: dist(WRIST, INDEX_FINGER_MCP) = 0.1.
    3) Optional left-hand mirroring to right-hand reference frame (x -> -x).
    """

    JOINTS = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    WRIST_IDX     = 0
    INDEX_MCP_IDX = 5
    TARGET_DIST   = 0.1
    CROP_SIZE     = 256
    FALLBACK_FRAME_SHAPE = (480, 640, 3)

    def __init__(
        self,
        url: str,
        camera_index: int | str = 0,
        padding: float = 0.3,
        jpeg_quality: int = 85,
        mirror_left_hand: bool = True,
        mirror_display: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        request_timeout: float = 5.0,
        window_name: str = "GraphGrasp - HaMeR",
    ):
        self.url = url.rstrip("/")
        self.camera_index = camera_index
        self.padding = padding
        self.jpeg_quality = jpeg_quality
        self.mirror_left_hand = mirror_left_hand
        self.mirror_display = mirror_display
        self.request_timeout = request_timeout
        self.window_name = window_name

        # MediaPipe para bbox + landmarks
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Camara
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if os.name == "posix" else [cv2.CAP_ANY]
        self._cap = None
        for b in backends:
            try:
                cap = cv2.VideoCapture(camera_index, b)
            except TypeError:
                cap = cv2.VideoCapture(camera_index)
            if cap and cap.isOpened():
                self._cap = cap
                break
            if cap:
                cap.release()

        self._frame_bgr      = None
        self._last_result    = None   # ultimo resultado de MediaPipe (para dibujo)
        self._last_handedness = "Unknown"
        self._window_open    = True
        self._window_initialized = False
        self._read_failures  = 0

        # Estado del hilo de inferencia
        self._infer_busy     = False
        self._latest_sample  = None   # ultimo dict de keypoints normalizados recibido
        self._last_bbox      = None   # ultimo bbox para dibujar

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_geometric(pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        pts -= pts[HaMeRBackend.WRIST_IDX]
        d = np.linalg.norm(pts[HaMeRBackend.INDEX_MCP_IDX])
        if d > 1e-6:
            pts *= (HaMeRBackend.TARGET_DIST / d)
        return pts

    def _get_bbox(self, landmarks) -> tuple:
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        x1 = max(0.0, min(xs) - self.padding)
        y1 = max(0.0, min(ys) - self.padding)
        x2 = min(1.0, max(xs) + self.padding)
        y2 = min(1.0, max(ys) + self.padding)
        return x1, y1, x2, y2

    def _crop_hand(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        crop = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        if crop.size == 0:
            return crop
        return cv2.resize(crop, (self.CROP_SIZE, self.CROP_SIZE))

    def _send_crop(self, crop: np.ndarray, is_right: bool) -> np.ndarray | None:
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        files = {"crop": ("crop.jpg", buf.tobytes(), "image/jpeg")}
        data  = {"is_right": int(is_right)}
        try:
            resp = requests.post(
                f"{self.url}/infer",
                files=files,
                data=data,
                timeout=self.request_timeout,
            )
            kp = resp.json().get("keypoints")
            if kp is not None:
                return np.array(kp, dtype=np.float32)  # (21, 3)
        except Exception as e:
            print(f"[HaMeRBackend] Request error: {e}")
        return None

    def _infer_async(self, crop: np.ndarray, is_right: bool, handedness: str) -> None:
        """Corre en hilo separado: manda crop, recibe keypoints, actualiza _latest_sample."""
        try:
            pts_raw = self._send_crop(crop, is_right)
            if pts_raw is None:
                return

            pts = self._normalize_geometric(pts_raw)
            if self.mirror_left_hand and handedness == "Left":
                pts = pts.copy()
                pts[:, 0] *= -1.0

            sample = {name: pts[i] for i, name in enumerate(self.JOINTS)}
            sample["handedness"] = handedness
            sample["grasp_type"] = 0
            self._latest_sample  = sample
        finally:
            self._infer_busy = False

    # ── PerceptionBackend interface ───────────────────────────────────────────

    def startup_error(self) -> str | None:
        if self._cap and self._cap.isOpened():
            return None
        source = self.camera_index
        source_desc = f"/dev/video{source}" if isinstance(source, int) else str(source)
        return (
            f"Camera source {source!r} could not be opened. "
            f"Try another index (0, 1, 2, ...). "
            f"On Linux verify device permissions for {source_desc}."
        )

    def get_landmarks(self) -> dict | None:
        """Lee un frame, corre MediaPipe y dispara inferencia HaMeR en background.
        Retorna los ultimos keypoints recibidos (no espera el request actual).
        """
        if not self._cap or not self._cap.isOpened():
            return None

        ok, frame_bgr = self._cap.read()
        if not ok:
            self._read_failures += 1
            return None
        self._read_failures = 0
        self._frame_bgr = frame_bgr

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result    = self._hands.process(frame_rgb)
        self._last_result = result

        if not result.multi_hand_landmarks:
            self._last_handedness = "Unknown"
            self._last_bbox       = None
            return self._latest_sample  # devuelve ultimo conocido o None

        hand = result.multi_hand_landmarks[0]

        # Handedness (compensar selfie mode)
        if result.multi_handedness:
            label     = result.multi_handedness[0].classification[0].label
            handedness = "Right" if label == "Left" else "Left"
        else:
            handedness = "Unknown"
        self._last_handedness = handedness
        is_right = handedness == "Right"

        # Bbox y crop
        bbox = self._get_bbox(hand)
        self._last_bbox = bbox
        crop = self._crop_hand(frame_bgr, bbox)

        # Disparar request a Colab en background si no hay uno en vuelo
        if not self._infer_busy and crop.size > 0:
            self._infer_busy = True
            t = threading.Thread(
                target=self._infer_async,
                args=(crop.copy(), is_right, handedness),
                daemon=True,
            )
            t.start()

        return self._latest_sample

    def is_ready(self) -> bool:
        return bool(self._cap and self._cap.isOpened() and self._window_open)

    def render(self, token=None, status_text: str = None, model_lines: list[str] | None = None) -> bool:
        if not self._window_open:
            return False
        if not self._window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_initialized = True

        frame = (
            self._frame_bgr.copy()
            if self._frame_bgr is not None
            else np.zeros(self.FALLBACK_FRAME_SHAPE, dtype=np.uint8)
        )
        if self.mirror_display and self._frame_bgr is not None:
            frame = cv2.flip(frame, 1)

        # Dibujar landmarks de MediaPipe
        if self._last_result and self._last_result.multi_hand_landmarks:
            hand_lm = self._last_result.multi_hand_landmarks[0]
            if self.mirror_display:
                hand_lm = type(hand_lm)()
                hand_lm.CopyFrom(self._last_result.multi_hand_landmarks[0])
                for lm in hand_lm.landmark:
                    lm.x = 1.0 - lm.x
            self._mp_draw.draw_landmarks(frame, hand_lm, self._mp_hands.HAND_CONNECTIONS)

        # Dibujar bbox
        if self._last_bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self._last_bbox
            bx1 = int((1.0 - x2) * w) if self.mirror_display else int(x1 * w)
            bx2 = int((1.0 - x1) * w) if self.mirror_display else int(x2 * w)
            color = (0, 255, 0) if self._last_handedness == "Right" else (255, 0, 0)
            cv2.rectangle(frame, (bx1, int(y1 * h)), (bx2, int(y2 * h)), color, 2)

        h, w = frame.shape[:2]
        panel_w = max(320, int(w * 0.45))
        canvas  = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        cv2.line(canvas, (w, 0), (w, h), (70, 70, 70), 2)

        lines = ["GraphGrasp [HaMeR]", f"Handedness: {self._last_handedness}"]
        if model_lines:
            lines.extend(model_lines)
        if token is not None:
            lines.append(f"Class ID: {token.class_id}")
            lines.append(f"Class: {token.class_name}")
            lines.append(f"Confidence: {token.confidence:.3f}")
            if getattr(token, "synergy_coeffs", None):
                lines.append("Head B (synergy):")
                for i, val in enumerate(token.synergy_coeffs[:6]):
                    lines.append(f"  s{i}: {val:.4f}")
        else:
            lines.append(status_text or "Inference: no data")

        x0 = w + 18
        y  = 34
        for i, line in enumerate(lines):
            color = (235, 235, 235)
            scale = 0.62
            thick = 1
            if i == 0:
                color = (100, 200, 255)
                scale = 0.82
                thick = 2
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y += 30 if i == 0 else 26

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            self._window_open = False
            cv2.destroyWindow(self.window_name)
            return False
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible < 1:
            self._window_open = False
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            return False
        return True

    def release(self) -> None:
        self._window_open = False
        self._window_initialized = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        cv2.destroyAllWindows()
