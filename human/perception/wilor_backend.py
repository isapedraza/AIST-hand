"""
WiLoR remote inference backend for GraphGrasp.

Pipeline:
  Webcam -> MediaPipe (bbox) -> crop -> POST to Colab (separate thread)
  -> WiLoR detects + reconstructs in the crop -> 21 keypoints 3D + is_right

WiLoR-mini (warmshao/WiLoR-mini) reconstructs MANO and returns pred_keypoints_3d
already in OpenPose/MediaPipe 21-joint order (mano_to_openpose remap), so the rows
map directly onto the standard JOINTS used by Dong -- no reordering needed.

Like the HaMeR backend, a client-side MediaPipe bbox is used to send a small CROP
(~256x256) instead of the full frame: this cuts the upload ~5x (a 640x480 jpg is
~200 KB vs ~42 KB for a 256 crop) and lets WiLoR's own detector work on a small
image -- the dominant per-request cost over a tunnel. The webcam and rendering
never block; get_landmarks() returns the last keypoints received without waiting.

Drop-in shape-compatible with the HaMeR path: get_landmarks() returns a dict
{JOINT_NAME: xyz, ..., "is_right": int}.
"""

import os
import threading

import cv2
import mediapipe as mp
import numpy as np
import requests

# Standard MediaPipe/OpenPose 21-joint order (matches WiLoR's mano_to_openpose output).
JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class WiLoRBackend:
    """WiLoR remote inference. Sends hand crops to a Colab server, returns latest keypoints."""

    CROP_SIZE = 256
    FALLBACK_FRAME_SHAPE = (480, 640, 3)

    def __init__(
        self,
        url: str,
        camera_index: int | str = 0,
        padding: float = 0.3,
        crop_size: int = 256,
        jpeg_quality: int = 80,
        request_timeout: float = 5.0,
        mirror_display: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        window_name: str = "GraphGrasp - WiLoR",
    ):
        self.url = url.rstrip("/")
        self.camera_index = camera_index
        self.padding = padding
        self.crop_size = crop_size
        self.jpeg_quality = jpeg_quality
        self.request_timeout = request_timeout
        self.mirror_display = mirror_display
        self.window_name = window_name

        # MediaPipe only for the bbox (and display); WiLoR does the actual hand pose.
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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

        self._frame_bgr = None
        self._last_result = None
        self._last_bbox = None
        self._window_open = True
        self._window_initialized = False

        self._infer_busy = False
        self._latest_sample = None

        # Reuse one TCP+TLS connection across frames (keep-alive). Over a tunnel
        # this saves the per-request handshake (~140 ms/frame measured).
        self._session = requests.Session()

    # ── helpers ───────────────────────────────────────────────────────────────

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
        return cv2.resize(crop, (self.crop_size, self.crop_size))

    # ── inference ─────────────────────────────────────────────────────────────

    def _infer_async(self, crop: np.ndarray) -> None:
        """Separate thread: POST crop, store keypoints + is_right."""
        try:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return
            files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            resp = self._session.post(f"{self.url}/infer", files=files, timeout=self.request_timeout)
            body = resp.json()
            kp = body.get("keypoints")
            if kp is None:
                return
            pts = np.array(kp, dtype=np.float32)  # (21, 3)
            if pts.shape != (21, 3):
                return
            sample = {name: pts[i] for i, name in enumerate(JOINTS)}
            sample["is_right"] = int(body.get("is_right", 1))
            self._latest_sample = sample
        except Exception as e:
            print(f"[WiLoRBackend] Request error: {e}")
        finally:
            self._infer_busy = False

    # ── source interface (matches HaMeRBackend) ───────────────────────────────

    def get_landmarks(self) -> dict | None:
        """Read a frame, find a bbox with MediaPipe, fire WiLoR inference on the crop."""
        if not self._cap or not self._cap.isOpened():
            return None
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        self._frame_bgr = frame_bgr

        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._last_result = result

        if not result.multi_hand_landmarks:
            self._last_bbox = None
            return self._latest_sample

        bbox = self._get_bbox(result.multi_hand_landmarks[0])
        self._last_bbox = bbox
        crop = self._crop_hand(frame_bgr, bbox)

        if not self._infer_busy and crop.size > 0:
            self._infer_busy = True
            t = threading.Thread(target=self._infer_async, args=(crop.copy(),), daemon=True)
            t.start()

        return self._latest_sample

    def is_ready(self) -> bool:
        return bool(self._cap and self._cap.isOpened() and self._window_open)

    def render(self, status_text: str | None = None) -> bool:
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

        if self._last_bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self._last_bbox
            bx1 = int((1.0 - x2) * w) if self.mirror_display else int(x1 * w)
            bx2 = int((1.0 - x1) * w) if self.mirror_display else int(x2 * w)
            cv2.rectangle(frame, (bx1, int(y1 * h)), (bx2, int(y2 * h)), (0, 255, 0), 2)

        msg = status_text or "Running (WiLoR)"
        if self._latest_sample is not None:
            side = "Right" if self._latest_sample.get("is_right", 1) else "Left"
            msg = f"{msg} | {side}"
        cv2.putText(frame, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        cv2.imshow(self.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
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
