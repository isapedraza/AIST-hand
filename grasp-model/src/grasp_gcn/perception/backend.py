from abc import ABC, abstractmethod


class PerceptionBackend(ABC):
    """Contract between any perception source and the GraphGrasp pipeline.

    Implement this interface to feed landmarks from any sensor
    (RGB camera + MediaPipe, depth camera, motion capture, haptic glove, etc.)
    into the GCN model.

    The only requirement is that `get_landmarks()` returns a dict with the
    21 hand joints in ToGraph-compatible format:

        {
            'WRIST':              [x, y, z],
            'THUMB_CMC':          [x, y, z],
            'THUMB_MCP':          [x, y, z],
            'THUMB_IP':           [x, y, z],
            'THUMB_TIP':          [x, y, z],
            'INDEX_FINGER_MCP':   [x, y, z],
            'INDEX_FINGER_PIP':   [x, y, z],
            'INDEX_FINGER_DIP':   [x, y, z],
            'INDEX_FINGER_TIP':   [x, y, z],
            'MIDDLE_FINGER_MCP':  [x, y, z],
            'MIDDLE_FINGER_PIP':  [x, y, z],
            'MIDDLE_FINGER_DIP':  [x, y, z],
            'MIDDLE_FINGER_TIP':  [x, y, z],
            'RING_FINGER_MCP':    [x, y, z],
            'RING_FINGER_PIP':    [x, y, z],
            'RING_FINGER_DIP':    [x, y, z],
            'RING_FINGER_TIP':    [x, y, z],
            'PINKY_MCP':          [x, y, z],
            'PINKY_PIP':          [x, y, z],
            'PINKY_DIP':          [x, y, z],
            'PINKY_TIP':          [x, y, z],
            'grasp_type':         int,   # optional at inference time
            'handedness':         str,   # optional: 'Right', 'Left', 'Unknown'
        }

    Example
    -------
    >>> class MyBackend(PerceptionBackend):
    ...     def get_landmarks(self):
    ...         # your sensor logic here
    ...         return {...}
    ...     def is_ready(self):
    ...         return True
    """

    @abstractmethod
    def get_landmarks(self) -> dict:
        """Return 21 hand joint coordinates in ToGraph-compatible format.

        Returns
        -------
        dict
            Joint name → [x, y, z] coordinates.
            Missing joints may be omitted; ToGraph will handle them as masked nodes.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the backend is initialized and ready to provide landmarks."""

    def render(self, token=None) -> None:
        """Optional: display current sensor state and inference results.

        This method exists to decouple visualization from the inference loop.
        `main.py` calls `backend.render()` every frame without knowing what
        the backend will show — or whether it will show anything at all.

        This design means:
        - The inference loop (main.py) stays clean and sensor-agnostic.
        - Each backend owns its own visualization logic.
        - A MediaPipe backend can show an OpenCV window with landmarks.
        - A Gradio backend can update a web UI.
        - A haptic glove backend can skip visualization entirely.
        - No visualization code leaks into main.py.

        Parameters
        ----------
        token : GraspToken or None
            Current token to display (class, confidence, aperture).
            None if no class has been confirmed yet.

        Override this method to add visualization. Default: no-op.
        """

    def release(self) -> None:
        """Release any resources held by the backend (camera, socket, etc.).
        Override if needed.
        """
