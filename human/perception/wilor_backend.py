"""
WiLoR remote inference backend for GraphGrasp.

Pipeline:
  Webcam -> downscaled full frame -> POST to Colab (separate thread)
  -> WiLoR runs its own detection + reconstruction -> 21 keypoints 3D + is_right

WiLoR-mini (warmshao/WiLoR-mini) reconstructs MANO and returns pred_keypoints_3d
already in OpenPose/MediaPipe 21-joint order (mano_to_openpose remap), so the rows
map directly onto the standard JOINTS used by Dong -- no reordering needed.

WiLoR does its own hand detection, so unlike the HaMeR backend there is no
client-side MediaPipe bbox/crop: we send a (downscaled) full frame and let the
server detect. The webcam and rendering never block; get_landmarks() returns the
last keypoints received without waiting on the in-flight request.

Drop-in shape-compatible with the HaMeR path: get_landmarks() returns a dict
{JOINT_NAME: xyz, ..., "is_right": int}.
"""

import os
import threading

import cv2
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
    """WiLoR remote inference. Sends frames to a Colab server, returns latest keypoints."""

    FALLBACK_FRAME_SHAPE = (480, 640, 3)

    def __init__(
        self,
        url: str,
        camera_index: int | str = 0,
        send_width: int = 640,
        jpeg_quality: int = 80,
        request_timeout: float = 5.0,
        mirror_display: bool = True,
        window_name: str = "GraphGrasp - WiLoR",
    ):
        self.url = url.rstrip("/")
        self.camera_index = camera_index
        self.send_width = send_width
        self.jpeg_quality = jpeg_quality
        self.request_timeout = request_timeout
        self.mirror_display = mirror_display
        self.window_name = window_name

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
        self._window_open = True
        self._window_initialized = False

        self._infer_busy = False
        self._latest_sample = None

    # ── inference ─────────────────────────────────────────────────────────────

    def _infer_async(self, frame_bgr: np.ndarray) -> None:
        """Separate thread: downscale, POST frame, store keypoints + is_right."""
        try:
            h, w = frame_bgr.shape[:2]
            if w > self.send_width:
                s = self.send_width / float(w)
                frame_bgr = cv2.resize(frame_bgr, (self.send_width, int(round(h * s))))

            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return
            files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            resp = requests.post(f"{self.url}/infer", files=files, timeout=self.request_timeout)
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
        """Read a frame, fire WiLoR inference in background, return last keypoints."""
        if not self._cap or not self._cap.isOpened():
            return None
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        self._frame_bgr = frame_bgr

        if not self._infer_busy:
            self._infer_busy = True
            t = threading.Thread(target=self._infer_async, args=(frame_bgr.copy(),), daemon=True)
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
        cv2.destroyAllWindows()
