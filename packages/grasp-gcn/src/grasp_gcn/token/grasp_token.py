from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass(frozen=True)
class GraspToken:
    """Continuous grasp-inference message emitted by the app runtime.

    The model operates in a canonical right-hand frame. If a left hand is
    observed, the perception backend may mirror it before graph construction;
    the original detector output can still be preserved in
    ``observed_handedness`` for logging or downstream routing.
    """

    timestamp: float = field(default_factory=time.time)
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: float = 0.0
    synergy_coeffs: list[float] = field(default_factory=list)
    confirmed: bool = False
    observed_handedness: Optional[str] = None
    source: str = "grasp-app"
