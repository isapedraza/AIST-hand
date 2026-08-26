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

# Camara calibrada y aprobada 2026-08-26 (palma al frente, mano completa en
# cuadro) -- ver models/grasp-intent-classification/scripts/render_fig41_pair.py.
_CAPTURE_CAM: dict[str, dict] = {
    "shadow":  dict(azimuth=180.0, elevation=0.0, distance=0.80, lookat=np.array([0.0, -0.01, 0.20])),
    "allegro": dict(azimuth=180.0, elevation=0.0, distance=0.46, lookat=np.array([0.02, 0.02, 0.05])),
}
_CAPTURE_CAM_DEFAULT = dict(azimuth=180.0, elevation=0.0, distance=0.40, lookat=np.array([0.0, 0.0, 0.2]))

# Fingertip = distal body's own local +Z offset (fixed joint in the URDF,
# robot/hands/shadow_hand/shadow_hand_right.urdf: FFtip/MFtip/RFtip/LFtip
# origin xyz="0 0 0.026", THtip origin xyz="0 0 0.0275"). Ported 1:1, same
# offsets already used for fig 3.19 (render_signals_sk.py).
_SHADOW_TIP_OFFSETS = {
    "thumb":  ("rh_thdistal", 0.0275),
    "index":  ("rh_ffdistal", 0.026),
    "middle": ("rh_mfdistal", 0.026),
    "ring":   ("rh_rfdistal", 0.026),
    "little": ("rh_lfdistal", 0.026),
}

_SCENE_BASE_XML = """<mujoco model="retarget_sink">
  <statistic extent="0.45" center="0 0 0.15"/>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global azimuth="145" elevation="-18"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.48 0.62" rgb2="0.04 0.05 0.06" width="512" height="3072"/>
    <texture type="2d" name="retarget_groundplane_tex" builtin="checker" mark="edge"
      rgb1="0.22 0.28 0.31" rgb2="0.10 0.13 0.15" markrgb="0.75 0.78 0.80" width="300" height="300"/>
    <material name="retarget_groundplane" texture="retarget_groundplane_tex" texuniform="true" texrepeat="5 5" reflectance="0.18"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5"/>
    <light pos="0.35 -0.2 1.5" dir="0 0 -1" directional="true"/>
    <geom name="retarget_floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" material="retarget_groundplane"/>
  </worldbody>
</mujoco>"""


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
  <statistic extent="0.45" center="0 0 0.15"/>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global azimuth="145" elevation="-18"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.48 0.62" rgb2="0.04 0.05 0.06" width="512" height="3072"/>
    <texture type="2d" name="retarget_groundplane_tex" builtin="checker" mark="edge"
      rgb1="0.22 0.28 0.31" rgb2="0.10 0.13 0.15" markrgb="0.75 0.78 0.80" width="300" height="300"/>
    <material name="retarget_groundplane" texture="retarget_groundplane_tex" texuniform="true" texrepeat="5 5" reflectance="0.18"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5"/>
    <light pos="0.35 -0.2 1.5" dir="0 0 -1" directional="true"/>
    <geom name="retarget_floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" material="retarget_groundplane"/>
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
        self._robot_name = robot
        cfg = _ROBOTS[robot]
        self._qpos_dim = cfg.qpos_dim
        scene_path  = _build_scene(cfg)
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data  = mujoco.MjData(self._model)
        self._target = np.zeros(self._qpos_dim, dtype=np.float64)
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._offscreen: mujoco.Renderer | None = None  # built lazily (only if capture() is used)

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def update(self, qpos: np.ndarray) -> None:
        self._target = qpos.reshape(self._qpos_dim).astype(np.float64)

        current = self._data.qpos[:self._qpos_dim]
        current[:] += POSE_ALPHA * (self._target - current)
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()

    def capture(self, path: Path) -> None:
        """Save a clean offscreen render (no window chrome/UI) of the current
        pose to `path`. Independent of the interactive viewer window."""
        if self._offscreen is None:
            self._model.vis.global_.offwidth = 900
            self._model.vis.global_.offheight = 900
            self._offscreen = mujoco.Renderer(self._model, height=900, width=900)
        cam_cfg = _CAPTURE_CAM.get(self._robot_name, _CAPTURE_CAM_DEFAULT)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance, cam.lookat = (
            cam_cfg["azimuth"], cam_cfg["elevation"], cam_cfg["distance"], cam_cfg["lookat"],
        )
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self._offscreen.update_scene(self._data, camera=cam, scene_option=opt)
        img = self._offscreen.render()
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(path))

    def tip_positions(self) -> dict[str, np.ndarray] | None:
        """World-frame fingertip positions {thumb,index,middle,ring,little} ->
        [3], via FK on the current data (distal body xpos + local +Z tip
        offset from the URDF). Shadow only -- returns None for other robots."""
        if self._robot_name != "shadow":
            return None
        out = {}
        for finger, (body_name, offset) in _SHADOW_TIP_OFFSETS.items():
            body = self._data.body(body_name)
            R = body.xmat.reshape(3, 3)
            out[finger] = body.xpos + R @ np.array([0.0, 0.0, offset])
        return out

    def wrist_position(self) -> np.ndarray:
        return self._data.body("rh_wrist").xpos.copy()

    def mcp_frame_points(self) -> dict[str, np.ndarray] | None:
        """{wrist, index_mcp, middle_mcp, ring_mcp} world positions -- same
        3 anatomical landmarks Dong Block 1 uses to build the wrist frame
        (robot/hand-configs/shadow.yaml: frame_index_mcp=ffknuckle, etc).
        Shadow only."""
        if self._robot_name != "shadow":
            return None
        return {
            "wrist":      self._data.body("rh_wrist").xpos.copy(),
            "index_mcp":  self._data.body("rh_ffknuckle").xpos.copy(),
            "middle_mcp": self._data.body("rh_mfknuckle").xpos.copy(),
            "ring_mcp":   self._data.body("rh_rfknuckle").xpos.copy(),
        }

    def reference_length(self) -> float | None:
        """Wrist-to-middle-knuckle distance, the same hand-length reference
        used on the human side (Dong's wrist frame, Metodos.tex Sec. 3.7).
        Shadow only."""
        if self._robot_name != "shadow":
            return None
        wrist = self._data.body("rh_wrist").xpos
        mf_mcp = self._data.body("rh_mfknuckle").xpos
        return float(np.linalg.norm(mf_mcp - wrist))

    def release(self) -> None:
        self._viewer.close()


def _load_spec(path: Path):
    """MjSpec loader. from_file resolves relative meshdir (menagerie); the .mjcf
    extension has no file decoder, so fall back to from_string (barrett uses an
    absolute meshdir, so string loading still resolves its meshes)."""
    p = str(path)
    try:
        return mujoco.MjSpec.from_file(p)
    except ValueError:
        return mujoco.MjSpec.from_string(path.read_text())


class MergedMuJocoSink:
    """Several robot hands in ONE MuJoCo window, side by side.

    Avoids opening multiple passive viewers in one process (which segfaults).
    The hands are merged into a single model via MjSpec.attach (per-robot name
    prefixes resolve all collisions) and spaced along X. One viewer, one data;
    each robot's qpos is written to its contiguous slice (attach order).

    update() takes {robot_name: qpos}. Robots must be keys of _ROBOTS.
    """

    def __init__(self, robots: list[str], spacing: float = 0.35):
        unknown = [r for r in robots if r not in _ROBOTS]
        if unknown:
            raise ValueError(f"Unsupported robots {unknown} (have {list(_ROBOTS)})")
        parent = mujoco.MjSpec.from_string(_SCENE_BASE_XML)
        self._slices: dict[str, tuple[int, int]] = {}
        off = 0
        for i, name in enumerate(robots):
            cfg   = _ROBOTS[name]
            child = _load_spec(cfg.model_path)
            frame = parent.worldbody.add_frame(pos=[i * spacing, 0.0, 0.05])
            parent.attach(child, prefix=f"{name}_", frame=frame)
            self._slices[name] = (off, cfg.qpos_dim)
            off += cfg.qpos_dim
        self._nq     = off
        self._model  = parent.compile()
        self._data   = mujoco.MjData(self._model)
        self._target = np.zeros(self._nq, dtype=np.float64)
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def update(self, qpos_by_robot: dict[str, np.ndarray]) -> None:
        for name, (o, d) in self._slices.items():
            if name in qpos_by_robot:
                self._target[o:o + d] = qpos_by_robot[name].reshape(d).astype(np.float64)
        cur = self._data.qpos[:self._nq]
        cur[:] += POSE_ALPHA * (self._target - cur)
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()

    def release(self) -> None:
        self._viewer.close()
