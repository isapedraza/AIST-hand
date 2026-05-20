"""
MuJoCo sink for retargeting pipeline.

Receives qpos [1, 24] from Retargeter and displays the Shadow Hand
in an interactive MuJoCo viewer (mouse to rotate/zoom).

Joint ordering (MuJoCo Menagerie 24 DOF):
  [0-1]   WRJ2, WRJ1
  [2-5]   FFJ4, FFJ3, FFJ2, FFJ1  (index)
  [6-9]   MFJ4, MFJ3, MFJ2, MFJ1  (middle)
  [10-13] RFJ4, RFJ3, RFJ2, RFJ1  (ring)
  [14-18] LFJ5, LFJ4, LFJ3, LFJ2, LFJ1  (little)
  [19-23] THJ5, THJ4, THJ3, THJ2, THJ1  (thumb)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SHADOW_DIR    = Path(__file__).resolve().parents[5] / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND    = SHADOW_DIR / "right_hand.xml"
HAND_QPOS_DIM = 24
POSE_ALPHA    = 0.25


def _build_scene() -> Path:
    upright = SHADOW_DIR / ".right_hand_upright.xml"
    scene   = SHADOW_DIR / ".scene_retarget_sink.xml"

    hand_text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
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
    Display Shadow Hand pose from retargeter output.

    Opens an interactive MuJoCo viewer — mouse to rotate/zoom.
    Input: np.ndarray [1, 24] or [24]  qpos in MuJoCo Menagerie order.
    """

    def __init__(self):
        scene_path  = _build_scene()
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data  = mujoco.MjData(self._model)
        self._target = np.zeros(HAND_QPOS_DIM, dtype=np.float64)
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def update(self, qpos: np.ndarray) -> None:
        self._target = qpos.reshape(HAND_QPOS_DIM).astype(np.float64)

        current = self._data.qpos[:HAND_QPOS_DIM]
        current[:] += POSE_ALPHA * (self._target - current)
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()

    def release(self) -> None:
        self._viewer.close()
