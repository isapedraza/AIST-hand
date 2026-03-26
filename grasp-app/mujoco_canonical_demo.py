"""
GraphGrasp semantic demo with MuJoCo Shadow Hand.

Runs the current perception + GCN pipeline and updates the Shadow Hand only
when the prediction is temporally stable and sufficiently confident, mapping a
small subset of classes to existing MuJoCo Menagerie keyframes.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/liberation2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import mujoco
import numpy as np
import torch

from grasp_gcn import GraspToken, ToGraph, VotingWindow, get_network

from inference_runtime import OpenHandLatch, parse_model_output, to_probs
from model_variant import resolve_model_spec
from perception.mediapipe_backend import MediaPipeBackend
from perception.hamer_backend import HaMeRBackend


ROOT = Path(__file__).resolve().parents[1]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"
KEYFRAMES_XML = SHADOW_DIR / "keyframes.xml"

NUM_FEATURES = 4
NETWORK_TYPE = "GCN_CAM_8_8_16_16_32"
N_VOTES = 3
CONFIDENCE_THRESHOLD = 0.55
POSE_SWITCH_VOTES = 4
HAND_QPOS_DIM = 24
POSE_ALPHA = 0.18
NO_HAND_RESET_FRAMES = 12
SIM_W = 800
SIM_H = 640
CAM_W = 400
CAM_H = 320
INFO_H = SIM_H - CAM_H
COMPOSITE_WINDOW = "GraphGrasp - MuJoCo"
CLOSE_REQUESTED = -2
CAMERA_RESET = {
    "azimuth": 152.0,
    "elevation": -16.0,
    "distance": 0.62,
    "lookat": np.array([0.0, 0.0, 0.12], dtype=np.float64),
}

# Only map classes with an acceptably direct semantic correspondence to the
# keyframes Menagerie already provides.
POSE_MAP_C28 = {
    "Large Diameter": "grasp hard",
    "Index Finger Extension": "point",
    "Parallel Extension": "parallel extension",
    "Palmar": "grasp hard",
    "Power Disk": "grasp hard",
    "Palmar Pinch": "two finger pinch",
    "Sphere 4-Finger": "grasp sphere",
    "Tripod": "three finger pinch",
}

POSE_MAP_TAXONOMY_V1 = {
    "Tripod_cluster": "three finger pinch",
    "Pinch_cluster": "two finger pinch",
    "Power_Wrap_cluster": "grasp hard",
    "Index_Finger_Ext": "point",
    "Parallel_Ext": "parallel extension",
    "Sphere_4_Finger": "grasp sphere",
}


def _resolve_camera_source() -> int | str:
    raw = os.getenv("GRAPHGRASP_CAMERA_INDEX", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MuJoCo semantic demo.")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera index (0,1,2,...) or stream URL. Overrides GRAPHGRASP_CAMERA_INDEX.",
    )
    return parser.parse_args()


def _get_camera_source(args: argparse.Namespace) -> int | str:
    if args.camera is None:
        return _resolve_camera_source()
    try:
        return int(args.camera)
    except ValueError:
        return args.camera

def _load_keyframes() -> dict[str, np.ndarray]:
    root = ET.parse(KEYFRAMES_XML).getroot()
    keyframes: dict[str, np.ndarray] = {}
    for key in root.findall(".//key"):
        name = key.attrib.get("name")
        qpos_text = key.attrib.get("qpos")
        if not name or not qpos_text:
            continue
        qpos = np.fromstring(qpos_text, sep=" ", dtype=np.float64)
        if qpos.size != HAND_QPOS_DIM:
            continue
        keyframes[name] = qpos
    if "open hand" not in keyframes:
        raise RuntimeError("Expected 'open hand' keyframe not found.")
    return keyframes


def _build_upright_scene_no_object() -> Path:
    upright_hand = SHADOW_DIR / ".right_hand_upright.xml"
    scene = SHADOW_DIR / ".scene_right_upright_no_object.xml"

    hand_text = RIGHT_HAND.read_text()
    hand_text = hand_text.replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )
    upright_hand.write_text(hand_text)

    scene.write_text(
        f"""<mujoco model="right_shadow_hand semantic demo">
  <include file="{upright_hand.name}"/>
  <statistic extent="0.3" center="0 0 0.2"/>
  <visual>
    <global azimuth="145" elevation="-18" offwidth="{SIM_W}" offheight="{SIM_H}"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1"/>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""
    )
    return scene


def _render_composite(
    sim_rgb: np.ndarray,
    backend: MediaPipeBackend,
    token: GraspToken | None,
    status_text: str,
    active_pose_name: str,
    model_lines: list[str],
) -> int:
    frame = backend._frame_bgr.copy() if backend._frame_bgr is not None else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    if backend.mirror_display:
        frame = cv2.flip(frame, 1)
    if backend._last_result and backend._last_result.multi_hand_landmarks:
        hand_landmarks = backend._last_result.multi_hand_landmarks[0]
        if backend.mirror_display:
            hand_landmarks = type(hand_landmarks)()
            hand_landmarks.CopyFrom(backend._last_result.multi_hand_landmarks[0])
            for landmark in hand_landmarks.landmark:
                landmark.x = 1.0 - landmark.x
        backend._mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            backend._mp_hands.HAND_CONNECTIONS,
        )
    frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

    info_panel = np.zeros((INFO_H, CAM_W, 3), dtype=np.uint8)
    lines = [
        "GraphGrasp",
        f"Handedness: {backend._last_handedness}",
        f"Pose: {active_pose_name}",
        status_text,
        *model_lines,
        "Camera: I/K up-down  J/L yaw",
        "U/O zoom  W/S tilt  R reset",
    ]
    if token is not None:
        lines.insert(2, f"Class: {token.class_name}")
        lines.insert(3, f"Confidence: {token.confidence:.3f}")

    y = 36
    for i, line in enumerate(lines):
        color = (235, 235, 235)
        scale = 0.65
        thick = 1
        if i == 0:
            color = (80, 220, 120)
            scale = 0.92
            thick = 2
        cv2.putText(
            info_panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )
        y += 34 if i == 0 else 28

    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
    sim_bgr = cv2.resize(sim_bgr, (SIM_W, SIM_H), interpolation=cv2.INTER_AREA)
    right_panel = np.vstack([frame, info_panel])
    canvas = np.hstack([sim_bgr, right_panel])

    cv2.imshow(COMPOSITE_WINDOW, canvas)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyWindow(COMPOSITE_WINDOW)
        return CLOSE_REQUESTED
    return key


def _handle_camera_key(camera: mujoco.MjvCamera, key: int) -> None:
    if key < 0 or key == 255:
        return

    key = key & 0xFF
    if key == ord("r"):
        camera.azimuth = CAMERA_RESET["azimuth"]
        camera.elevation = CAMERA_RESET["elevation"]
        camera.distance = CAMERA_RESET["distance"]
        camera.lookat[:] = CAMERA_RESET["lookat"]
    elif key == ord("j"):
        camera.azimuth -= 4.0
    elif key == ord("l"):
        camera.azimuth += 4.0
    elif key == ord("w"):
        camera.elevation += 2.0
    elif key == ord("s"):
        camera.elevation -= 2.0
    elif key == ord("u"):
        camera.distance = max(0.2, camera.distance - 0.03)
    elif key == ord("o"):
        camera.distance += 0.03
    elif key == ord("i"):
        camera.lookat[2] += 0.02
    elif key == ord("k"):
        camera.lookat[2] -= 0.02


def main():
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    camera_source = _get_camera_source(args)
    spec = resolve_model_spec()
    model_path = Path(spec["model_path"])
    num_classes = spec["num_classes"]
    class_names = spec["class_names"]
    model_lines = [
        f"Model: {spec['variant']}",
        f"Classes: {num_classes}",
        f"Camera: {camera_source}",
    ]
    if spec["variant"] == "taxonomy_v1":
        pose_map = POSE_MAP_TAXONOMY_V1
    else:
        pose_map = POSE_MAP_C28
    mapped_classes = set(pose_map)
    keyframes = _load_keyframes()
    scene_path = _build_upright_scene_no_object()

    model = get_network(
        NETWORK_TYPE,
        NUM_FEATURES,
        num_classes,
        use_cmc_angle=True,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[mujoco] Inference model loaded: variant={spec['variant']} path={model_path}")

    mujoco_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mujoco_data = mujoco.MjData(mujoco_model)
    renderer = mujoco.Renderer(mujoco_model, height=SIM_H, width=SIM_W)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = CAMERA_RESET["azimuth"]
    camera.elevation = CAMERA_RESET["elevation"]
    camera.distance = CAMERA_RESET["distance"]
    camera.lookat[:] = CAMERA_RESET["lookat"]
    target_qpos = keyframes["open hand"].copy()
    mujoco_data.qpos[:HAND_QPOS_DIM] = target_qpos
    mujoco_data.qvel[:] = 0
    mujoco.mj_forward(mujoco_model, mujoco_data)

    backend = HaMeRBackend(
        url="https://village-learning-pill-lip.trycloudflare.com",
        camera_index=camera_source,
        window_name="GraphGrasp - HaMeR Demo",
    )
    startup_error = backend.startup_error()
    if startup_error:
        backend.release()
        renderer.close()
        raise RuntimeError(startup_error)
    to_graph = ToGraph(
        features="xyz",
        make_undirected=True,
        add_joint_angles=True,
        add_cmc_angle=True,
    )
    window = VotingWindow(n=N_VOTES)
    open_hand_latch = OpenHandLatch()
    active_pose_name = "open hand"
    pending_pose_name: str | None = None
    pending_pose_votes = 0
    frames_without_hand = 0

    cv2.namedWindow(COMPOSITE_WINDOW, cv2.WINDOW_AUTOSIZE)
    loop_count = 0
    while backend.is_ready():
        loop_count += 1
        landmarks = backend.get_landmarks()
        status_text = "Perception: waiting for hand"
        token = None

        if landmarks is not None:
            frames_without_hand = 0
            if open_hand_latch.update(landmarks):
                window.reset()
                pending_pose_name = None
                pending_pose_votes = 0
                target_qpos = keyframes["open hand"].copy()
                active_pose_name = "open hand"
                token = GraspToken(
                    class_id=None,
                    class_name="Open Hand",
                    confidence=1.0,
                    confirmed=True,
                    observed_handedness=landmarks.get("handedness"),
                    source="grasp-app+mujoco",
                )
                status_text = "MuJoCo: open hand override"
            else:
                data = to_graph(landmarks).to(device)
                with torch.no_grad():
                    output = model(data)
                head_a, head_b = parse_model_output(output)
                probs = to_probs(head_a)

                class_id = probs.argmax().item()
                confidence = probs.max().item()
                class_name = class_names.get(class_id, str(class_id))
                if class_name not in mapped_classes:
                    window.reset()
                    open_hand_latch.reset()
                    pending_pose_name = None
                    pending_pose_votes = 0
                    status_text = "MuJoCo: class filtered"
                else:
                    confirmed = window.update(class_id, confidence)

                    synergy_coeffs = []
                    if head_b is not None:
                        if isinstance(head_b, torch.Tensor):
                            synergy_coeffs = head_b.detach().view(head_b.size(0), -1)[0].cpu().tolist()
                        else:
                            synergy_coeffs = list(head_b)

                    token = GraspToken(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        synergy_coeffs=synergy_coeffs,
                        confirmed=confirmed,
                        observed_handedness=landmarks.get("handedness"),
                        source="grasp-app+mujoco",
                    )

                    pose_name = pose_map[class_name]
                    if confirmed and confidence >= CONFIDENCE_THRESHOLD and pose_name in keyframes:
                        if pose_name == active_pose_name:
                            pending_pose_name = None
                            pending_pose_votes = 0
                            status_text = f"MuJoCo: {pose_name}"
                        else:
                            if pose_name == pending_pose_name:
                                pending_pose_votes += 1
                            else:
                                pending_pose_name = pose_name
                                pending_pose_votes = 1

                            if pending_pose_votes >= POSE_SWITCH_VOTES:
                                target_qpos = keyframes[pose_name].copy()
                                active_pose_name = pose_name
                                pending_pose_name = None
                                pending_pose_votes = 0
                                status_text = f"MuJoCo: {pose_name}"
                            else:
                                status_text = (
                                    f"MuJoCo: hold {pose_name} "
                                    f"({pending_pose_votes}/{POSE_SWITCH_VOTES})"
                                )
                    elif confirmed:
                        pending_pose_name = None
                        pending_pose_votes = 0
                        status_text = "MuJoCo: confidence too low"
                    else:
                        pending_pose_name = None
                        pending_pose_votes = 0
                        status_text = "Inference: waiting for stable vote"
        else:
            frames_without_hand += 1
            if frames_without_hand >= NO_HAND_RESET_FRAMES:
                target_qpos = keyframes["open hand"].copy()
                active_pose_name = "open hand"
                pending_pose_name = None
                pending_pose_votes = 0
                window.reset()
                open_hand_latch.reset()
                status_text = "Perception: hand lost -> open hand"

        hand_qpos = mujoco_data.qpos[:HAND_QPOS_DIM]
        hand_qpos[:] = hand_qpos + POSE_ALPHA * (target_qpos - hand_qpos)
        mujoco_data.qvel[:] = 0
        mujoco.mj_forward(mujoco_model, mujoco_data)
        renderer.update_scene(mujoco_data, camera=camera)
        sim_rgb = renderer.render()

        key = _render_composite(sim_rgb, backend, token, status_text, active_pose_name, model_lines)
        if key == CLOSE_REQUESTED:
            break
        _handle_camera_key(camera, key)

    backend.release()
    renderer.close()


if __name__ == "__main__":
    main()
