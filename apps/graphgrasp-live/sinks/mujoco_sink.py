"""
MuJoCo sink for retargeting pipeline.

Receives qpos from Retargeter and displays the robot hand in an interactive
MuJoCo viewer (mouse to rotate/zoom). Robot selected by name (default shadow).

Shadow Hand (24 DOF):
  [0-1]   WRJ2, WRJ1
  [2-5]   FFJ4, FFJ3, FFJ2, FFJ1  (index)
  [6-9]   MFJ4, MFJ3, MFJ2, MFJ1  (middle)
  [10-13] RFJ4, RFJ3, RFJ2, RFJ1  (ring)
  [14-18] LFJ5, LFJ4, LFJ3, LFJ2, LFJ1  (little)
  [19-23] THJ5, THJ4, THJ3, THJ2, THJ1  (thumb)

Allegro Hand (16 DOF): ff[j0-3], mf[j0-3], rf[j0-3], th[j0-3] (no wrist DOFs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_MENAGERIE = Path(__file__).resolve().parents[3] / "third_party" / "mujoco_menagerie"
POSE_ALPHA = 0.25


@dataclass(frozen=True)
class _RobotSink:
    hand_dir: Path
    qpos_dim: int
    orig_body_tag: str   # exact tag to find in right_hand.xml
    new_body_tag: str    # replacement (orients hand upright, lifts off floor)


_ROBOTS: dict[str, _RobotSink] = {
    "shadow": _RobotSink(
        hand_dir=_MENAGERIE / "shadow_hand",
        qpos_dim=24,
        orig_body_tag='<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        new_body_tag='<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
    ),
    "allegro": _RobotSink(
        hand_dir=_MENAGERIE / "wonik_allegro",
        qpos_dim=16,
        orig_body_tag='<body name="palm" quat="0 1 0 1" childclass="allegro_right">',
        new_body_tag='<body name="palm" pos="0 0 0.05" quat="1 0 0 0" childclass="allegro_right">',
    ),
}


def _build_scene(cfg: _RobotSink) -> Path:
    right_hand = cfg.hand_dir / "right_hand.xml"
    upright    = cfg.hand_dir / ".right_hand_upright.xml"
    scene      = cfg.hand_dir / ".scene_retarget_sink.xml"

    hand_text = right_hand.read_text().replace(cfg.orig_body_tag, cfg.new_body_tag, 1)
    upright.write_text(hand_text)
    scene.write_text(f"""<mujoco model="retarget_sink">
  <include file="{upright.name}"/>
  <statistic extent="0.3" center="0 0 0.2"/>
  <visual>
    <global azimuth="145" elevation="-18"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5"/>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
  </worldbody>
</mujoco>""")
    return scene


class MuJocoSink:
    """
    Display a robot hand pose from retargeter output.

    Opens an interactive MuJoCo viewer — mouse to rotate/zoom.
    robot: "shadow" (24 DOF) or "allegro" (16 DOF).
    Input: np.ndarray [1, J] or [J]  qpos in MuJoCo Menagerie order.
    """

    def __init__(self, robot: str = "shadow"):
        if robot not in _ROBOTS:
            raise ValueError(f"Unsupported robot '{robot}' (have {list(_ROBOTS)})")
        cfg = _ROBOTS[robot]
        self._qpos_dim = cfg.qpos_dim
        scene_path  = _build_scene(cfg)
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data  = mujoco.MjData(self._model)
        self._target = np.zeros(self._qpos_dim, dtype=np.float64)
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def update(self, qpos: np.ndarray) -> None:
        self._target = qpos.reshape(self._qpos_dim).astype(np.float64)

        current = self._data.qpos[:self._qpos_dim]
        current[:] += POSE_ALPHA * (self._target - current)
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()

    def release(self) -> None:
        self._viewer.close()
