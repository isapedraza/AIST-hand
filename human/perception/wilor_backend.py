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
import time

import cv2
import mediapipe as mp
import numpy as np
import requests

try:
    from pynput import keyboard as _kb
    _PYNPUT_OK = True
except ImportError:
    _PYNPUT_OK = False

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
        record_directory=None,
        record_metadata=None,
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

        # A str camera_index is a network stream URL (phone-as-webcam: an MJPEG
        # app like "IP Webcam" over USB tethering -> http://PHONE_IP:8080/video).
        # Force FFMPEG (V4L2 chokes on URLs) and a 1-frame buffer so cv2 never
        # queues stale frames -> no creeping lag on top of WiLoR's own latency.
        is_url = isinstance(camera_index, str) and "://" in camera_index
        if is_url:
            backends = [cv2.CAP_FFMPEG]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if os.name == "posix" else [cv2.CAP_ANY]
        self._cap = None
        for b in backends:
            try:
                cap = cv2.VideoCapture(camera_index, b)
            except TypeError:
                cap = cv2.VideoCapture(camera_index)
            if cap and cap.isOpened():
                if is_url:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        self._capture_requested = False
        self._capture_lock = threading.Lock()

        # Global spacebar listener: fires regardless of OS window focus, unlike
        # cv2.waitKey below (only sees keys while the cv2 window itself is
        # focused -- confirmed live 2026-08-26, silently dropped most presses
        # during a DexJoCo capture session where the MuJoCo viewer had focus).
        self._kb_listener = None
        if _PYNPUT_OK:
            self._kb_listener = _kb.Listener(on_press=self._on_key_press)
            self._kb_listener.start()
        else:
            print("WARNING: pynput not installed -- space only captures with the cv2 window focused. "
                  "pip install pynput for global capture.")

        # Reuse one TCP+TLS connection across frames (keep-alive). Over a tunnel
        # this saves the per-request handshake (~140 ms/frame measured).
        self._session = requests.Session()
        self._recorder = None
        if record_directory is not None:
            from human.perception.timed_recording import TimedCameraRecorder
            try:
                self._recorder = TimedCameraRecorder(record_directory, record_metadata)
            except BaseException:
                self.release()
                raise

    def _on_key_press(self, key):
        if key == _kb.Key.space:
            with self._capture_lock:
                self._capture_requested = True

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

    def _infer_async(self, crop: np.ndarray, bbox_px: float,
                      bbox_cx: float, bbox_cy: float, img_w: int, img_h: int) -> None:
        """Separate thread: POST crop, store keypoints + is_right.

        bbox_px = hand bbox width in ORIGINAL (uncropped) frame pixels. The crop
        itself is always resized to a fixed crop_size, so it erases the
        apparent-size-shrinks-with-distance cue WiLoR's depth/translation
        (cam_t.x/y/z) need; bbox_px + bbox_cx/cy + img_w/h (all measured in the
        ORIGINAL frame, before cropping) are sent alongside so the server can
        recompute cam_crop_to_full() with real geometry instead of WiLoR's own
        bbox detected inside the re-centered crop (see
        docs/estado_wrist_txty_dead_2026-07-04.md).
        """
        try:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return
            files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            data = {
                "bbox_px": str(bbox_px),
                "bbox_cx": str(bbox_cx),
                "bbox_cy": str(bbox_cy),
                "img_w": str(img_w),
                "img_h": str(img_h),
            }
            resp = self._session.post(f"{self.url}/infer", files=files, data=data, timeout=self.request_timeout)
            body = resp.json()
            kp = body.get("keypoints")
            if kp is None:
                return
            pts = np.array(kp, dtype=np.float32)  # (21, 3)
            if pts.shape != (21, 3):
                return
            sample = {name: pts[i] for i, name in enumerate(JOINTS)}
            sample["is_right"] = int(body.get("is_right", 1))
            # Wrist 6D extras (optional; absent on old servers). cam_t = global
            # camera-frame wrist translation, global_orient = MANO native global
            # rotation (axis-angle). Consumed by WiLoRSource.wrist_pose() for the
            # sim teleop wrist channel; the finger path ignores them.
            ct = body.get("cam_t")
            if ct is not None:
                sample["cam_t"] = np.array(ct, dtype=np.float32).reshape(-1)[:3]
            go = body.get("global_orient")
            if go is not None:
                sample["global_orient"] = np.array(go, dtype=np.float32).reshape(-1)[:3]
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
        acquired_ns = time.monotonic_ns()
        if not ok:
            return None
        if self._recorder is not None:
            self._recorder.submit(frame_bgr, acquired_ns)
        self._frame_bgr = frame_bgr

        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._last_result = result

        if not result.multi_hand_landmarks:
            self._last_bbox = None
            return self._latest_sample

        bbox = self._get_bbox(result.multi_hand_landmarks[0])
        self._last_bbox = bbox
        crop = self._crop_hand(frame_bgr, bbox)
        img_h, img_w = frame_bgr.shape[:2]
        bbox_px = (bbox[2] - bbox[0]) * img_w  # width in original-frame px
        bbox_cx = (bbox[0] + bbox[2]) / 2.0 * img_w  # bbox center, original-frame px
        bbox_cy = (bbox[1] + bbox[3]) / 2.0 * img_h

        if not self._infer_busy and crop.size > 0:
            self._infer_busy = True
            t = threading.Thread(
                target=self._infer_async,
                args=(crop.copy(), bbox_px, bbox_cx, bbox_cy, img_w, img_h),
                daemon=True,
            )
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

        side = "Right" if (self._latest_sample or {}).get("is_right", 1) else "Left"

        # MediaPipe skeleton (the bbox detector's landmarks, same as the HaMeR
        # window -- this is the client-side detector, not WiLoR's own keypoints).
        if self._last_result and self._last_result.multi_hand_landmarks:
            hand_lm = self._last_result.multi_hand_landmarks[0]
            if self.mirror_display:
                mirrored = type(hand_lm)()
                mirrored.CopyFrom(hand_lm)
                for lm in mirrored.landmark:
                    lm.x = 1.0 - lm.x
                hand_lm = mirrored
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lm, self._mp_hands.HAND_CONNECTIONS
            )

        # bbox (color by handedness, matching the HaMeR window)
        if self._last_bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self._last_bbox
            bx1 = int((1.0 - x2) * w) if self.mirror_display else int(x1 * w)
            bx2 = int((1.0 - x1) * w) if self.mirror_display else int(x2 * w)
            color = (0, 255, 0) if side == "Right" else (255, 0, 0)
            cv2.rectangle(frame, (bx1, int(y1 * h)), (bx2, int(y2 * h)), color, 2)

        # Side panel (matches HaMeRBackend.render).
        h, w = frame.shape[:2]
        panel_w = max(320, int(w * 0.45))
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        cv2.line(canvas, (w, 0), (w, h), (70, 70, 70), 2)

        lines = ["GraphGrasp [WiLoR]", f"Handedness: {side}", status_text or "Running (WiLoR)"]
        x0 = w + 18
        y = 34
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            with self._capture_lock:
                self._capture_requested = True
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

    def pop_capture_request(self) -> bool:
        """True once per spacebar press (consumes the flag)."""
        with self._capture_lock:
            r = self._capture_requested
            self._capture_requested = False
        return r

    def current_frame(self):
        """Copy of the latest raw webcam frame (BGR), or None before the first frame."""
        return None if self._frame_bgr is None else self._frame_bgr.copy()

    def current_frame_with_skeleton(self):
        """Latest webcam frame (BGR) with the MediaPipe hand skeleton drawn on
        top -- not mirrored, so it matches the physical hand orientation (the
        live preview window mirrors for the user's convenience; a saved figure
        should not)."""
        if self._frame_bgr is None:
            return None
        frame = self._frame_bgr.copy()
        if self._last_result and self._last_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, self._last_result.multi_hand_landmarks[0], self._mp_hands.HAND_CONNECTIONS
            )
        return frame

    def release(self) -> None:
        self._window_open = False
        self._window_initialized = False
        if self._kb_listener is not None:
            self._kb_listener.stop()
            self._kb_listener = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        cv2.destroyAllWindows()
        if self._recorder is not None:
            self._recorder.close()
