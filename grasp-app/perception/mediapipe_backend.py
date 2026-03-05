from grasp_gcn import PerceptionBackend


class MediaPipeBackend(PerceptionBackend):
    """Reference implementation of PerceptionBackend using MediaPipe.

    Captures frames from an RGB camera and extracts 21 hand landmarks
    using Google MediaPipe Hands.

    To use a different sensor, create a new file in this folder and
    implement PerceptionBackend. Then swap this class in main.py.
    """

    def __init__(self):
        # TODO: initialize MediaPipe and camera
        pass

    def get_landmarks(self) -> dict:
        # TODO: capture frame, run MediaPipe, return 21 joints in ToGraph format
        raise NotImplementedError

    def is_ready(self) -> bool:
        # TODO: return True when camera and MediaPipe are initialized
        raise NotImplementedError

    def release(self) -> None:
        # TODO: release camera
        pass
