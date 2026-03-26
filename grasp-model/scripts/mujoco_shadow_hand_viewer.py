import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
SCENE_RIGHT = SHADOW_DIR / "scene_right.xml"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"


def _build_upright_scene() -> Path:
    tmp_scene = SHADOW_DIR / ".scene_right_upright.xml"
    tmp_hand = SHADOW_DIR / ".right_hand_upright.xml"

    scene_text = SCENE_RIGHT.read_text()
    hand_text = RIGHT_HAND.read_text()

    # Rotate the forearm root into a more upright presentation.
    hand_text = hand_text.replace(
        '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
        '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">',
        1,
    )

    # Bring the default camera lower and more frontal for presentation.
    scene_text = scene_text.replace(
        'include file="right_hand.xml"',
        'include file=".right_hand_upright.xml"',
        1,
    )
    scene_text = scene_text.replace(
        '<global azimuth="220" elevation="-30"/>',
        '<global azimuth="145" elevation="-18"/>',
        1,
    )

    tmp_scene.write_text(scene_text)
    tmp_hand.write_text(hand_text)
    return tmp_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Load the original Menagerie scene without reorienting the hand.",
    )
    args = parser.parse_args()

    if not SCENE_RIGHT.exists():
        raise FileNotFoundError(f"Shadow Hand scene not found: {SCENE_RIGHT}")

    scene_path = SCENE_RIGHT if args.raw else _build_upright_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
