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

_REPO      = Path(__file__).resolve().parents[3]
_MENAGERIE = _REPO / "third_party" / "mujoco_menagerie"
POSE_ALPHA = 0.25


@dataclass(frozen=True)
class _RobotSink:
    model_path: Path          # right_hand.xml (menagerie) or full .mjcf (barrett)
    qpos_dim: int
    # upright=(orig_tag,new_tag): rewrite the root body to stand the hand upright
    # off the floor (shadow). None -> include the model file as-is (leap/barrett).
    upright: tuple[str, str] | None = None


_ROBOTS: dict[str, _RobotSink] = {
    "shadow": _RobotSink(
        model_path=_MENAGERIE / "shadow_hand" / "right_hand.xml",
        qpos_dim=24,
        upright=(
            '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
            '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        ),
    ),
    "allegro": _RobotSink(
        model_path=_MENAGERIE / "wonik_allegro" / "right_hand.xml",
        qpos_dim=16,
        upright=(
            '<body name="palm" quat="0 1 0 1" childclass="allegro_right">',
            '<body name="palm" pos="0 0 0.05" quat="1 0 0 0" childclass="allegro_right">',
        ),
    ),
    "leap": _RobotSink(
        model_path=_MENAGERIE / "leap_hand" / "right_hand.xml",
        qpos_dim=16,
    ),
    "barrett": _RobotSink(
        model_path=_REPO / "robot" / "hands" / "barrett_hand" / "barrett.mjcf",
        qpos_dim=8,
    ),
}


def _build_scene(cfg: _RobotSink) -> Path:
    hand_dir = cfg.model_path.parent
    scene    = hand_dir / ".scene_retarget_sink.xml"

    if cfg.upright is not None:
        orig, new = cfg.upright
        upright   = hand_dir / ".right_hand_upright.xml"
        upright.write_text(cfg.model_path.read_text().replace(orig, new, 1))
        include_name = upright.name
    else:
        include_name = cfg.model_path.name

    scene.write_text(f"""<mujoco model="retarget_sink">
  <include file="{include_name}"/>
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
