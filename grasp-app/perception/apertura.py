class AperturaCalculator:
    """Computes normalized hand aperture [0, 1] from raw landmarks.

    Aperture is a continuous signal measuring how open the human hand is
    for a given grasp class. It is independent of the robot morphology —
    the robot interprets the scalar within its own physical limits.

    Aperture depends on grasp class: different grasp types measure
    openness differently (e.g. Pinch uses thumb-index distance,
    Large Diameter uses palm-fingertip spread).

    Must be called after VotingWindow confirms a class.
    """

    @staticmethod
    def compute(landmarks: dict, class_id: int) -> float:
        """Compute aperture for the given grasp class from landmarks.

        Parameters
        ----------
        landmarks : dict
            21 joints in ToGraph-compatible format (from PerceptionBackend).
        class_id : int
            Confirmed grasp class from VotingWindow.

        Returns
        -------
        float
            Normalized aperture in [0, 1].
            0 = as closed as possible for this grasp.
            1 = fully open.
        """
        # TODO: implement per-class aperture measurement
        raise NotImplementedError
