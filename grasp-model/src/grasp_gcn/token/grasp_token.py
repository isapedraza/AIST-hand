from dataclasses import dataclass


@dataclass
class GraspToken:
    """Grasp intent signal passed from perception to the robot adapter.

    class_id   : int   — grasp type index (Feix taxonomy)
    class_name : str   — human-readable label (e.g. "Tripod")
    confidence : float — model confidence [0, 1]
    apertura   : float — normalized hand openness [0, 1]
    """
    class_id:   int
    class_name: str
    confidence: float
    apertura:   float
