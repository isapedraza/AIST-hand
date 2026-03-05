from dataclasses import dataclass


@dataclass
class GraspToken:
    """Complete grasp intent signal — contract between perception and robot.

    Carries both the classified grasp type (from the GCN) and the hand
    aperture (from geometry). Both are required: a token without aperture
    is incomplete and cannot be executed by a RobotAdapter.

    Fields
    ------
    class_id   : int   — grasp type index (GRASP taxonomy)
    class_name : str   — human-readable name (e.g. "Tripod")
    confidence : float — model confidence [0, 1]
    apertura   : float — normalized hand openness [0, 1] for this grasp class
                         (0 = as closed as possible, 1 = fully open)
    """
    class_id:   int
    class_name: str
    confidence: float
    apertura:   float
