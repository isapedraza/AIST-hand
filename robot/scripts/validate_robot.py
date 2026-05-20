#!/usr/bin/env python3
"""Validate a robot.yaml draft against its URDF."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


def _load_urdf(urdf_path: Path) -> tuple[set[str], set[str]]:
    root = ET.parse(urdf_path).getroot()
    links = {elem.attrib["name"] for elem in root.findall("link") if "name" in elem.attrib}
    joints = {
        elem.attrib["name"]
        for elem in root.findall("joint")
        if elem.attrib.get("type") != "fixed" and "name" in elem.attrib
    }
    return links, joints


def _iter_component_lists(doc: dict[str, Any]):
    for cname, component in (doc.get("components") or {}).items():
        for joint in component.get("joints") or []:
            yield f"components.{cname}.joints", joint, "joint"

        for chain_name, chain in (component.get("kinematic_chains") or {}).items():
            for joint in chain.get("joints") or []:
                yield f"components.{cname}.kinematic_chains.{chain_name}.joints", joint, "joint"
            for link in chain.get("links") or []:
                yield f"components.{cname}.kinematic_chains.{chain_name}.links", link, "link"

        palm_frame = (component.get("representation") or {}).get("palm_frame") or {}
        for key, link in palm_frame.items():
            yield f"components.{cname}.representation.palm_frame.{key}", link, "link"

        for role_name, role in (component.get("semantic_roles") or {}).items():
            if role_name == "special_constraints":
                for item in role or []:
                    for joint in item.get("joints") or []:
                        yield f"components.{cname}.semantic_roles.special_constraints", joint, "joint"
                continue
            for joint in role.get("joints") or []:
                yield f"components.{cname}.semantic_roles.{role_name}.joints", joint, "joint"


def validate(robot_yaml: Path, urdf_override: Path | None = None) -> list[str]:
    with robot_yaml.open() as fh:
        doc = yaml.safe_load(fh)

    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["robot.yaml must contain a mapping at top level"]

    urdf_path = urdf_override
    if urdf_path is None:
        asset_urdf = (doc.get("assets") or {}).get("urdf")
        if asset_urdf:
            candidate = Path(asset_urdf)
            urdf_path = candidate if candidate.is_absolute() else (robot_yaml.parent / candidate)
    if urdf_path is None or not urdf_path.exists():
        errors.append("URDF not found. Pass --urdf or set assets.urdf to an existing path.")
        urdf_links, urdf_joints = set(), set()
    else:
        urdf_links, urdf_joints = _load_urdf(urdf_path)

    joints = doc.get("joints") or []
    if not isinstance(joints, list):
        errors.append("joints must be a list")
        joints = []

    names = [j.get("name") for j in joints if isinstance(j, dict)]
    indices = [j.get("index") for j in joints if isinstance(j, dict)]
    if len(names) != len(set(names)):
        errors.append("joint names must be unique")
    if sorted(indices) != list(range(len(indices))):
        errors.append("joint indices must be unique and consecutive from 0")

    yaml_joint_names = set(names)
    missing_from_urdf = sorted(yaml_joint_names - urdf_joints)
    if missing_from_urdf:
        errors.append(f"joints missing from URDF: {missing_from_urdf}")

    for joint in joints:
        if not isinstance(joint, dict):
            errors.append("each joint entry must be a mapping")
            continue
        limits = joint.get("limits")
        if not isinstance(limits, list) or len(limits) != 2:
            errors.append(f"{joint.get('name')}: limits must be [lower, upper]")
        elif limits[0] is None or limits[1] is None:
            errors.append(f"{joint.get('name')}: actuated joint limits must not be null")

    for path, value, kind in _iter_component_lists(doc):
        if kind == "joint" and value not in yaml_joint_names:
            errors.append(f"{path}: unknown joint {value!r}")
        if kind == "link" and value not in urdf_links:
            errors.append(f"{path}: unknown link {value!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_yaml", help="Path to robot.yaml")
    parser.add_argument("--urdf", default=None, help="Override URDF path")
    args = parser.parse_args()

    errors = validate(
        Path(args.robot_yaml),
        Path(args.urdf).expanduser().resolve() if args.urdf else None,
    )
    if errors:
        print("Invalid robot spec:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Robot spec valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
