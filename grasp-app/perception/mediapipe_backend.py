import cv2
import mediapipe as mp
import numpy as np

from grasp_gcn import PerceptionBackend


class MediaPipeBackend(PerceptionBackend):
    """MediaPipe implementation of PerceptionBackend with train-aligned preprocessing.

    Preprocessing contract for model compatibility:
    1) Root-relative coordinates (subtract WRIST).
    2) Scale normalization to target dist(WRIST, INDEX_FINGER_MCP) = 0.1.
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
    WRIST_IDX = 0
    INDEX_MCP_IDX = 5
    TARGET_DIST = 0.1

    def __init__(
        self,
        camera_index: int = 0,
        mirror_left_hand: bool = True,
        selfie_mode: bool = True,
        mirror_display: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        window_name: str = "GraphGrasp - MediaPipe",
    ):
        self.mirror_left_hand = mirror_left_hand
        self.selfie_mode = selfie_mode
        self.mirror_display = mirror_display
        self.window_name = window_name

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._cap = cv2.VideoCapture(camera_index)
        self._frame_bgr = None
        self._last_result = None
        self._last_handedness = "Unknown"
        self._window_open = True
        self._window_initialized = False

    @staticmethod
    def _normalize_geometric(pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        pts -= pts[MediaPipeBackend.WRIST_IDX]
        d = np.linalg.norm(pts[MediaPipeBackend.INDEX_MCP_IDX])
        if d > 1e-6:
            pts *= (MediaPipeBackend.TARGET_DIST / d)
        return pts

    @staticmethod
    def _extract_handedness(result, idx: int) -> str:
        if not result.multi_handedness or idx >= len(result.multi_handedness):
            return "Unknown"
        cls = result.multi_handedness[idx].classification
        if not cls:
            return "Unknown"
        label = cls[0].label
        return label if label in ("Left", "Right") else "Unknown"

    def _normalize_handedness(self, handedness: str) -> str:
        if handedness not in ("Left", "Right"):
            return "Unknown"
        if not self.selfie_mode:
            return handedness
        return "Right" if handedness == "Left" else "Left"

    def get_landmarks(self) -> dict:
        if not self._cap or not self._cap.isOpened():
            return None

        ok, frame_bgr = self._cap.read()
        if not ok:
            return None

        self._frame_bgr = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        self._last_result = result

        if not result.multi_hand_landmarks:
            self._last_handedness = "Unknown"
            return None

        hand = result.multi_hand_landmarks[0]
        handedness = self._normalize_handedness(
            self._extract_handedness(result, 0)
        )
        self._last_handedness = handedness

        pts = np.array(
            [(lm.x, lm.y, lm.z) for lm in hand.landmark],
            dtype=np.float32,
        )  # [21,3]

        pts = self._normalize_geometric(pts)
        if self.mirror_left_hand and handedness == "Left":
            pts[:, 0] *= -1.0

        sample = {name: pts[i] for i, name in enumerate(self.JOINTS)}
        sample["handedness"] = handedness
        sample["grasp_type"] = 0  # placeholder required by current ToGraph
        return sample

    def is_ready(self) -> bool:
        return bool(self._cap and self._cap.isOpened() and self._window_open)

    def render(self, token=None, status_text: str = None) -> bool:
        if not self._window_open or self._frame_bgr is None:
            return False
        if not self._window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_initialized = True

        frame = self._frame_bgr.copy()
        if self.mirror_display:
            frame = cv2.flip(frame, 1)
        if self._last_result and self._last_result.multi_hand_landmarks:
            hand_landmarks = self._last_result.multi_hand_landmarks[0]
            if self.mirror_display:
                hand_landmarks = type(hand_landmarks)()
                hand_landmarks.CopyFrom(self._last_result.multi_hand_landmarks[0])
                for landmark in hand_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x
            self._mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )

        h, w = frame.shape[:2]
        panel_w = max(320, int(w * 0.45))
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        # Divider between camera and info panel
        cv2.line(canvas, (w, 0), (w, h), (70, 70, 70), 2)

        lines = [
            "GraphGrasp",
            f"Handedness: {self._last_handedness}",
        ]

        if token is not None:
            lines.append(f"Class ID: {token.class_id}")
            lines.append(f"Confidence: {token.confidence:.3f}")
            if getattr(token, "synergy_coeffs", None):
                lines.append("Head B (synergy):")
                for i, val in enumerate(token.synergy_coeffs[:6]):
                    lines.append(f"  s{i}: {val:.4f}")
        else:
            lines.append(status_text or "Inference: no data")

        x0 = w + 18
        y = 34
        for i, line in enumerate(lines):
            color = (235, 235, 235)
            scale = 0.62
            thick = 1
            if i == 0:
                color = (80, 220, 120)
                scale = 0.82
                thick = 2
            cv2.putText(
                canvas,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )
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
