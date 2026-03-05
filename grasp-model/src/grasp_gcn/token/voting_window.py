from collections import deque
from typing import Optional


class VotingWindow:
    """Temporal smoothing filter for grasp classification.

    Returns True when the same class is predicted for `n` consecutive frames,
    False otherwise. Does NOT construct GraspToken — that is the caller's
    responsibility, so that aperture can be included at construction time.

    Example
    -------
    >>> window = VotingWindow(n=5)
    >>> confirmed = window.update(class_id=0, confidence=0.92)
    >>> if confirmed:
    ...     apertura = AperturaCalculator.compute(landmarks, class_id=0)
    ...     token = GraspToken(class_id=0, class_name="Large diameter",
    ...                        confidence=0.92, apertura=apertura)
    """

    def __init__(self, n: int = 5):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._buffer: deque[int] = deque(maxlen=n)

    def update(self, class_id: int, confidence: float) -> bool:
        """Push a new prediction. Returns True if consensus is reached."""
        self._buffer.append(class_id)
        return len(self._buffer) == self.n and len(set(self._buffer)) == 1

    def reset(self) -> None:
        self._buffer.clear()
