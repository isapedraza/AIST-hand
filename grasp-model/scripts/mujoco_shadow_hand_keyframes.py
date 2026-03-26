from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"
KEYFRAMES_XML = SHADOW_DIR / "keyframes.xml"
HAND_QPOS_DIM = 24
POSE_ALPHA = 0.12


def load_keyframes() -> dict[str, np.ndarray]:
    root = ET.parse(KEYFRAMES_XML).getroot()
    out: dict[str, np.ndarray] = {}
    for key in root.findall(".//key"):
        name = key.attrib.get("name")
        qpos_text = key.attrib.get("qpos")
        if not name or not qpos_text:
            continue
        qpos = np.fromstring(qpos_text, sep=" ", dtype=np.float64)
        if qpos.size == HAND_QPOS_DIM:
            out[name] = qpos
    return out


def build_scene() -> Path:
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
        f"""<mujoco model="right_shadow_hand keyframes">
  <include file="{upright_hand.name}"/>
  <statistic extent="0.3" center="0 0 0.2"/>
  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="8192"/>
    <global azimuth="145" elevation="-18"/>
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", help="Show only one named keyframe.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds per pose in autoplay.")
    args = parser.parse_args()

    keyframes = load_keyframes()
    if not keyframes:
        raise RuntimeError("No keyframes found.")

    if args.pose:
        names = [args.pose]
        if args.pose not in keyframes:
            raise KeyError(f"Unknown pose: {args.pose}")
    else:
        names = list(keyframes.keys())

    print("Keyframes:")
    for name in names:
        print("-", name)

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    target = keyframes[names[0]].copy()
    data.qpos[:HAND_QPOS_DIM] = target
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        pose_idx = 0
        last_switch = time.time()
        while viewer.is_running():
            now = time.time()
            if len(names) > 1 and now - last_switch >= args.seconds:
                pose_idx = (pose_idx + 1) % len(names)
                target = keyframes[names[pose_idx]].copy()
                print(f"Pose -> {names[pose_idx]}")
                last_switch = now

            hand_qpos = data.qpos[:HAND_QPOS_DIM]
            hand_qpos[:] = hand_qpos + POSE_ALPHA * (target - hand_qpos)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
