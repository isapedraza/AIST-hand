#!/usr/bin/env python3
"""Generate a draft robot.yaml from a URDF.

The generated file contains mechanical facts from the URDF plus conservative
component hints. Semantic hand metadata is filled only for known Shadow-style
joint/link names and should be reviewed by a human.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


HAND_PREFIXES = ("WRJ", "FFJ", "MFJ", "RFJ", "LFJ", "THJ")
ARM_JOINT_NAMES = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
}


def _parse_xyz(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(v) for v in text.split()]


def _infer_kind(urdf_path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    parts = set(urdf_path.parts)
    if "assembly" in parts:
        return "assembly"
    if "hands" in parts:
        return "hand"
    if "arms" in parts:
        return "arm"
    return "robot"


def _component_for_joint(name: str, kind: str) -> str | None:
    if kind == "hand":
        return "hand"
    if kind == "arm":
        return "arm"
    if kind == "assembly":
        if name.startswith(HAND_PREFIXES):
            return "hand"
        if name in ARM_JOINT_NAMES:
            return "arm"
    return None


def _parse_urdf(urdf_path: Path) -> tuple[str, set[str], list[dict[str, Any]]]:
    root = ET.parse(urdf_path).getroot()
    robot_name = root.attrib.get("name", urdf_path.stem)
    links = {elem.attrib["name"] for elem in root.findall("link") if "name" in elem.attrib}

    joints: list[dict[str, Any]] = []
    for elem in root.findall("joint"):
        joint_type = elem.attrib.get("type", "")
        if joint_type == "fixed":
            continue

        name = elem.attrib.get("name")
        if not name:
            continue

        parent = elem.find("parent")
        child = elem.find("child")
        axis = elem.find("axis")
        limit = elem.find("limit")

        lower = upper = effort = velocity = None
        if limit is not None:
            lower = float(limit.attrib["lower"]) if "lower" in limit.attrib else None
            upper = float(limit.attrib["upper"]) if "upper" in limit.attrib else None
            effort = float(limit.attrib["effort"]) if "effort" in limit.attrib else None
            velocity = float(limit.attrib["velocity"]) if "velocity" in limit.attrib else None

        joint: dict[str, Any] = {
            "name": name,
            "index": len(joints),
            "type": joint_type,
            "limits": [lower, upper],
            "parent_link": parent.attrib.get("link") if parent is not None else None,
            "child_link": child.attrib.get("link") if child is not None else None,
        }
        parsed_axis = _parse_xyz(axis.attrib.get("xyz") if axis is not None else None)
        if parsed_axis is not None:
            joint["axis"] = parsed_axis
        if effort is not None:
            joint["effort"] = effort
        if velocity is not None:
            joint["velocity"] = velocity
        joints.append(joint)

    return robot_name, links, joints


def _shadow_hand_metadata(joint_names: set[str], links: set[str]) -> dict[str, Any]:
    has_shadow_joints = {"FFJ4", "MFJ4", "RFJ4", "LFJ4", "THJ5"}.issubset(joint_names)
    has_shadow_links = {"palm", "ffknuckle", "mfknuckle", "rfknuckle"}.issubset(links)
    if not (has_shadow_joints and has_shadow_links):
        return {
            "kinematic_chains": {},
            "representation": {},
            "semantic_roles": {
                "abduction": {"joints": [], "xin_position_weight": 0.5},
                "thumb_rotation": {"joints": [], "xin_position_weight": 0.1},
                "special_constraints": [],
            },
        }

    return {
        "kinematic_chains": {
            "thumb": {
                "joints": ["THJ5", "THJ4", "THJ3", "THJ2", "THJ1"],
                "links": ["thbase", "thmiddle", "thdistal", "thtip"],
            },
            "index": {
                "joints": ["FFJ4", "FFJ3", "FFJ2", "FFJ1"],
                "links": ["ffknuckle", "ffmiddle", "ffdistal", "fftip"],
            },
            "middle": {
                "joints": ["MFJ4", "MFJ3", "MFJ2", "MFJ1"],
                "links": ["mfknuckle", "mfmiddle", "mfdistal", "mftip"],
            },
            "ring": {
                "joints": ["RFJ4", "RFJ3", "RFJ2", "RFJ1"],
                "links": ["rfknuckle", "rfmiddle", "rfdistal", "rftip"],
            },
            "pinky": {
                "joints": ["LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1"],
                "links": ["lfknuckle", "lfmiddle", "lfdistal", "lftip"],
            },
        },
        "representation": {
            "palm_frame": {
                "wrist_link": "palm",
                "index_mcp_link": "ffknuckle",
                "middle_mcp_link": "mfknuckle",
                "ring_mcp_link": "rfknuckle",
            }
        },
        "semantic_roles": {
            "abduction": {
                "joints": ["FFJ4", "MFJ4", "RFJ4", "LFJ4"],
                "xin_position_weight": 0.5,
            },
            "thumb_rotation": {
                "joints": ["THJ5"],
                "xin_position_weight": 0.1,
            },
            "special_constraints": [],
        },
    }


def _build_components(kind: str, joints: list[dict[str, Any]], links: set[str]) -> dict[str, Any]:
    joint_names = {j["name"] for j in joints}
    components: dict[str, Any] = {}

    arm_joints = [j["name"] for j in joints if j.get("component") == "arm"]
    hand_joints = [j["name"] for j in joints if j.get("component") == "hand"]

    if kind == "arm" or arm_joints:
        components["arm"] = {
            "kind": "arm",
            "joints": arm_joints or [j["name"] for j in joints],
            "base_link": "base_link" if "base_link" in links else None,
            "end_effector_link": "ee_link" if "ee_link" in links else None,
        }

    if kind == "hand" or hand_joints:
        hand_meta = _shadow_hand_metadata(joint_names, links)
        components["hand"] = {
            "kind": "hand",
            "joints": hand_joints or [j["name"] for j in joints],
            **hand_meta,
        }

    return components


def generate(urdf_path: Path, out_path: Path, kind: str, asset_path: str | None) -> None:
    urdf_path = urdf_path.expanduser().resolve()
    robot_name, links, joints = _parse_urdf(urdf_path)
    kind = _infer_kind(urdf_path, kind)

    for joint in joints:
        component = _component_for_joint(joint["name"], kind)
        if component:
            joint["component"] = component

    doc: dict[str, Any] = {
        "name": robot_name,
        "kind": kind,
        "dof": len(joints),
        "metadata": {
            "generated_from": asset_path or str(urdf_path),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "component_assignment": "heuristic",
            "review_required": True,
        },
        "assets": {
            "urdf": asset_path or str(urdf_path),
        },
        "joints": joints,
    }

    components = _build_components(kind, joints, links)
    if components:
        doc["components"] = components

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, help="Source URDF path.")
    parser.add_argument("--out", required=True, help="Output robot.yaml path.")
    parser.add_argument(
        "--kind",
        default="auto",
        choices=["auto", "hand", "arm", "assembly", "robot"],
        help="Robot kind. Defaults to path-based inference.",
    )
    parser.add_argument(
        "--asset-path",
        default=None,
        help="Path to write under assets.urdf instead of the absolute source path.",
    )
    args = parser.parse_args()

    generate(Path(args.urdf), Path(args.out), args.kind, args.asset_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
