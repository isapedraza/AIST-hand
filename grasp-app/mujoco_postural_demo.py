"""
GraphGrasp -- Postural Control Demo with MuJoCo Shadow Hand.

Analog of Segil et al. (2014) C3 postural controller:
  XYZ (MediaPipe) -> abl04 GNN -> top-2 probs -> interpolated qpos -> Shadow Hand

All 27 Feix classes active (vs 8 in mujoco_canonical_demo.py).
Continuous interpolation -- no discrete state switching.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/liberation2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grasp-model" / "src"))
sys.path.insert(0, str(ROOT / "grasp-robot"))
sys.path.insert(0, str(Path(__file__).parent))

from postural_control import PosturalController
from inference_runtime import OpenHandLatch
from perception.mediapipe_backend import MediaPipeBackend

SHADOW_DIR    = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND    = SHADOW_DIR / "right_hand.xml"
HAND_QPOS_DIM = 24
POSE_ALPHA    = 0.18          # smoothing factor (same as original demo)
CONFIDENCE_THRESHOLD = 0.45   # below this → hand flat (uncertain prediction)
NO_HAND_RESET_FRAMES = 12
SIM_W = 800;  SIM_H = 640
CAM_W = 400;  CAM_H = 320
INFO_H = SIM_H - CAM_H
WINDOW = "GraphGrasp - Postural Control"
CLOSE_REQUESTED = -2
CAMERA_RESET = {
    "azimuth": 152.0, "elevation": -16.0,
    "distance": 0.62,
    "lookat": np.array([0.0, 0.0, 0.12], dtype=np.float64),
}

HAND_FLAT = np.zeros(HAND_QPOS_DIM, dtype=np.float64)   # open hand fallback


def _build_scene() -> Path:
    upright = SHADOW_DIR / ".right_hand_upright.xml"
    scene   = SHADOW_DIR / ".scene_postural.xml"
    text = RIGHT_HAND.read_text().replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright.write_text(text)
    scene.write_text(f"""<mujoco model="graphgrasp postural demo">
  <include file="{upright.name}"/>
  <statistic extent="0.3" center="0 0 0.2"/>
  <visual>
    <global azimuth="145" elevation="-18" offwidth="{SIM_W}" offheight="{SIM_H}"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1"/>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>""")
    return scene


def _render_composite(
    sim_rgb: np.ndarray,
    backend: MediaPipeBackend,
    top2: list | None,
    status: str,
) -> int:
    frame = backend._frame_bgr.copy() if backend._frame_bgr is not None \
            else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    if backend._last_result and backend._last_result.multi_hand_landmarks:
        lm = backend._last_result.multi_hand_landmarks[0]
        backend._mp_draw.draw_landmarks(frame, lm, backend._mp_hands.HAND_CONNECTIONS)
    if backend.mirror_display:
        frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

    info = np.zeros((INFO_H, CAM_W, 3), dtype=np.uint8)
    lines = ["GraphGrasp -- Postural Control", status]
    if top2:
        for rank, (_, name, prob) in enumerate(top2):
            lines.append(f"  Top-{rank+1}: {name}  ({prob:.2f})")
    lines += ["", "J/L yaw  W/S tilt  U/O zoom  R reset  Q quit"]

    y = 36
    for i, line in enumerate(lines):
        color = (80, 220, 120) if i == 0 else (235, 235, 235)
        scale = 0.85 if i == 0 else 0.60
        thick = 2 if i == 0 else 1
        cv2.putText(info, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)
        y += 34 if i == 0 else 26

    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
    sim_bgr = cv2.resize(sim_bgr, (SIM_W, SIM_H), interpolation=cv2.INTER_AREA)
    canvas  = np.hstack([sim_bgr, np.vstack([frame, info])])
    cv2.imshow(WINDOW, canvas)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyWindow(WINDOW)
        return CLOSE_REQUESTED
    return key


def _handle_camera_key(cam: mujoco.MjvCamera, key: int) -> None:
    if key < 0 or key == 255:
        return
    key = key & 0xFF
    if   key == ord("r"): cam.azimuth = CAMERA_RESET["azimuth"]; cam.elevation = CAMERA_RESET["elevation"]; cam.distance = CAMERA_RESET["distance"]; cam.lookat[:] = CAMERA_RESET["lookat"]
    elif key == ord("j"): cam.azimuth   -= 4.0
    elif key == ord("l"): cam.azimuth   += 4.0
    elif key == ord("w"): cam.elevation += 2.0
    elif key == ord("s"): cam.elevation -= 2.0
    elif key == ord("u"): cam.distance   = max(0.2, cam.distance - 0.03)
    elif key == ord("o"): cam.distance  += 0.03
    elif key == ord("i"): cam.lookat[2] += 0.02
    elif key == ord("k"): cam.lookat[2] -= 0.02


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=str, default="0")
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cpu")

    print("Loading PosturalController...")
    pc = PosturalController(device=device)

    scene_path   = _build_scene()
    mj_model     = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data      = mujoco.MjData(mj_model)
    renderer     = mujoco.Renderer(mj_model, height=SIM_H, width=SIM_W)
    cam          = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth   = CAMERA_RESET["azimuth"]
    cam.elevation = CAMERA_RESET["elevation"]
    cam.distance  = CAMERA_RESET["distance"]
    cam.lookat[:] = CAMERA_RESET["lookat"]

    target_qpos = HAND_FLAT.copy()
    mj_data.qpos[:HAND_QPOS_DIM] = target_qpos
    mujoco.mj_forward(mj_model, mj_data)

    try:
        camera_idx = int(args.camera)
    except ValueError:
        camera_idx = args.camera

    backend = MediaPipeBackend(
        camera_index=camera_idx,
        window_name=WINDOW,
    )
    if backend.startup_error():
        raise RuntimeError(backend.startup_error())

    open_hand_latch    = OpenHandLatch()
    frames_without_hand = 0
    top2               = None
    status             = "Waiting for hand..."

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    while backend.is_ready():
        landmarks = backend.get_landmarks()

        if landmarks is not None:
            frames_without_hand = 0

            if open_hand_latch.update(landmarks):
                # Hand flat override
                target_qpos = HAND_FLAT.copy()
                top2        = None
                status      = "Open hand detected"
            else:
                # Postural control: extract xyz [21,3] from landmarks dict
                xyz = np.array(
                    [landmarks[k] for k in backend.JOINTS],
                    dtype=np.float32,
                )  # [21,3]

                qpos_new, top2 = pc(xyz, top_k=2)
                top1_conf = top2[0][2]
                if top1_conf >= CONFIDENCE_THRESHOLD:
                    target_qpos = qpos_new.astype(np.float64)
                    status      = "Postural control active"
                else:
                    target_qpos = HAND_FLAT.copy()
                    status      = f"Low confidence ({top1_conf:.2f}) -- open hand"
        else:
            frames_without_hand += 1
            if frames_without_hand >= NO_HAND_RESET_FRAMES:
                target_qpos = HAND_FLAT.copy()
                top2        = None
                status      = "Hand lost -- open hand"
                open_hand_latch.reset()
                    pc.reset()

        # Smooth towards target (Segil-style continuous morph)
        hand_qpos = mj_data.qpos[:HAND_QPOS_DIM]
        hand_qpos[:] += POSE_ALPHA * (target_qpos - hand_qpos)
        mj_data.qvel[:] = 0
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        sim_rgb = renderer.render()

        key = _render_composite(sim_rgb, backend, top2, status)
        if key == CLOSE_REQUESTED:
            break
        _handle_camera_key(cam, key)

    backend.release()
    renderer.close()


if __name__ == "__main__":
    main()
