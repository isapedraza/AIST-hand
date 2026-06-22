"""Structural validation for declarative robot -> UDHM adapters."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from cross_emb.udhm_backbone_refactor.core.udhm import UDHM22_SLOTS


def _repo_root(path: Path) -> Path:
    for parent in (path.resolve(), *path.resolve().parents):
        if (parent / ".git").exists():
            return parent
    raise ValueError(f"cannot find repository root from {path}")


def validate_adapter_file(path: str | Path) -> dict[str, int | str]:
    """Validate slots, signs, URDF names, links, and movable-joint coverage."""
    path = Path(path)
    adapter = yaml.safe_load(path.read_text())
    errors: list[str] = []

    if not isinstance(adapter, dict):
        raise ValueError(f"{path}: adapter must be a YAML mapping")
    joints = adapter.get("joints")
    if not isinstance(joints, dict) or not joints:
        errors.append("joints must be a non-empty mapping")
        joints = {}

    slots: list[str] = []
    for joint, entry in joints.items():
        if not isinstance(entry, dict):
            errors.append(f"{joint}: entry must be a mapping")
            continue
        slot, sign = entry.get("slot"), entry.get("sign")
        if slot not in UDHM22_SLOTS:
            errors.append(f"{joint}: unknown UDHM slot {slot!r}")
        else:
            slots.append(slot)
        if sign not in (-1, 1):
            errors.append(f"{joint}: sign must be +1 or -1, got {sign!r}")

    repeated = sorted({slot for slot in slots if slots.count(slot) > 1})
    if repeated:
        errors.append(f"slots mapped more than once: {repeated}")

    urdf_value = adapter.get("urdf")
    if not isinstance(urdf_value, str):
        errors.append("urdf must be a repository-relative path")
        urdf_path = None
    else:
        urdf_path = _repo_root(path) / urdf_value
        if not urdf_path.is_file():
            errors.append(f"URDF does not exist: {urdf_value}")

    movable: set[str] = set()
    if urdf_path is not None and urdf_path.is_file():
        root = ET.parse(urdf_path).getroot()
        movable = {
            str(j.get("name"))
            for j in root.findall("joint")
            if j.get("type") in {"revolute", "continuous", "prismatic"}
        }
        links = {str(link.get("name")) for link in root.findall("link")}
        ignored = set(adapter.get("ignored_joints", []))
        unknown = (set(joints) | ignored) - movable
        missing = movable - set(joints) - ignored
        overlap = set(joints) & ignored
        if unknown:
            errors.append(f"joints absent from URDF: {sorted(unknown)}")
        if missing:
            errors.append(f"movable joints neither mapped nor ignored: {sorted(missing)}")
        if overlap:
            errors.append(f"joints both mapped and ignored: {sorted(overlap)}")

        referenced_links: set[str] = set()
        frame = adapter.get("frame", {})
        if isinstance(frame, dict):
            referenced_links.update(str(value) for value in frame.values())
        fingers = adapter.get("fingers", {})
        if isinstance(fingers, dict):
            for entry in fingers.values():
                if isinstance(entry, dict):
                    referenced_links.update(str(value) for value in entry.values())
        absent_links = referenced_links - links
        if absent_links:
            errors.append(f"links absent from URDF: {sorted(absent_links)}")

    if errors:
        raise ValueError(f"{path}:\n- " + "\n- ".join(errors))
    return {
        "robot": str(adapter.get("robot", "")),
        "mapped_joints": len(joints),
        "present_slots": len(slots),
        "movable_joints": len(movable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.adapters:
        result = validate_adapter_file(path)
        print(f"{path}: OK {result}")


if __name__ == "__main__":
    main()
