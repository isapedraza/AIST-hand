#!/usr/bin/env python3
"""
Generic URDF randomizer + FK sampler.

Goal:
  - Load any URDF.
  - Extract actuated joints and limits automatically.
  - Sample random joint configurations uniformly inside limits.
  - Run FK for sampled configurations.
  - Print a clear summary to stdout.

This script does not modify training code. It is a standalone utility to validate
that random q sampling from URDF limits works across different hands/robots.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import pytorch_kinematics as pk


ACTUATED_TYPES = {"revolute", "prismatic", "continuous"}


@dataclass
class JointSpec:
    name: str
    joint_type: str
    lower: float | None
    upper: float | None
    mimic_parent: str | None
    mimic_multiplier: float
    mimic_offset: float


class URDFRandomizer:
    def __init__(self, urdf_path: str | Path, device: str = "cpu", continuous_range: float = math.pi):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.device = torch.device(device)
        self.continuous_range = float(continuous_range)

        urdf_bytes = self.urdf_path.read_bytes()
        self.chain = pk.build_chain_from_urdf(urdf_bytes)
        self.chain_joint_names = list(self.chain.get_joint_parameter_names())
        self.joint_specs, self.robot_name, self.link_names = self._parse_urdf_metadata(self.urdf_path)

        # Precompute specs in chain order and report missing metadata.
        self.specs_in_chain_order = self._build_specs_in_chain_order()

    def _parse_urdf_metadata(self, urdf_path: Path) -> tuple[dict[str, JointSpec], str, list[str]]:
        root = ET.parse(urdf_path).getroot()
        robot_name = root.attrib.get("name", urdf_path.stem)
        link_names = [e.attrib["name"] for e in root.findall("link") if "name" in e.attrib]

        specs: dict[str, JointSpec] = {}
        for joint in root.findall("joint"):
            name = joint.attrib.get("name")
            jtype = joint.attrib.get("type", "")
            if not name:
                continue

            limit = joint.find("limit")
            lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None
            upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None

            mimic = joint.find("mimic")
            mimic_parent = None
            mimic_multiplier = 1.0
            mimic_offset = 0.0
            if mimic is not None:
                mimic_parent = mimic.attrib.get("joint")
                if "multiplier" in mimic.attrib:
                    mimic_multiplier = float(mimic.attrib["multiplier"])
                if "offset" in mimic.attrib:
                    mimic_offset = float(mimic.attrib["offset"])

            specs[name] = JointSpec(
                name=name,
                joint_type=jtype,
                lower=lower,
                upper=upper,
                mimic_parent=mimic_parent,
                mimic_multiplier=mimic_multiplier,
                mimic_offset=mimic_offset,
            )
        return specs, robot_name, link_names

    def _default_limits_for(self, joint_type: str) -> tuple[float, float]:
        if joint_type == "continuous":
            return -self.continuous_range, self.continuous_range
        # Fallback for missing limits: use the same policy as continuous.
        return -self.continuous_range, self.continuous_range

    def _resolve_limits(self, spec: JointSpec) -> tuple[float, float]:
        if spec.lower is not None and spec.upper is not None:
            lo, hi = float(spec.lower), float(spec.upper)
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        return self._default_limits_for(spec.joint_type)

    def _build_specs_in_chain_order(self) -> list[JointSpec]:
        specs_in_order: list[JointSpec] = []
        for jname in self.chain_joint_names:
            spec = self.joint_specs.get(jname)
            if spec is None:
                # Unknown in XML metadata; assume revolute with fallback range.
                spec = JointSpec(
                    name=jname,
                    joint_type="revolute",
                    lower=None,
                    upper=None,
                    mimic_parent=None,
                    mimic_multiplier=1.0,
                    mimic_offset=0.0,
                )
            specs_in_order.append(spec)
        return specs_in_order

    def sample_q(self, num_samples: int, seed: int | None = None) -> tuple[torch.Tensor, list[str]]:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        if seed is not None:
            torch.manual_seed(int(seed))

        B = int(num_samples)
        values: dict[str, torch.Tensor] = {}
        unresolved_mimic: list[JointSpec] = []

        # First pass: sample non-mimic joints.
        for spec in self.specs_in_chain_order:
            if spec.joint_type not in ACTUATED_TYPES:
                # Chain params are expected to be actuated, but keep guard.
                lo, hi = self._default_limits_for("revolute")
                values[spec.name] = lo + torch.rand(B, device=self.device) * (hi - lo)
                continue

            if spec.mimic_parent:
                unresolved_mimic.append(spec)
                continue

            lo, hi = self._resolve_limits(spec)
            values[spec.name] = lo + torch.rand(B, device=self.device) * (hi - lo)

        # Resolve mimic joints from their parents.
        # If dependency cannot be resolved, fallback to random inside own limits.
        while unresolved_mimic:
            progressed = False
            next_round: list[JointSpec] = []
            for spec in unresolved_mimic:
                parent = spec.mimic_parent
                if parent and parent in values:
                    v = values[parent] * spec.mimic_multiplier + spec.mimic_offset
                    lo, hi = self._resolve_limits(spec)
                    values[spec.name] = torch.clamp(v, min=lo, max=hi)
                    progressed = True
                else:
                    next_round.append(spec)
            if not progressed:
                for spec in next_round:
                    lo, hi = self._resolve_limits(spec)
                    values[spec.name] = lo + torch.rand(B, device=self.device) * (hi - lo)
                break
            unresolved_mimic = next_round

        q = torch.stack([values[name] for name in self.chain_joint_names], dim=1)  # [B, J]
        return q, self.chain_joint_names

    # =========================================================================
    # STAGE 1 — FULL FK ON THE COMPLETE HAND
    # =========================================================================
    def run_fk(self, q_samples: torch.Tensor) -> dict[str, torch.Tensor]:
        if q_samples.ndim != 2:
            raise ValueError(f"q_samples must be [B, J], got {tuple(q_samples.shape)}")
        if q_samples.shape[1] != len(self.chain_joint_names):
            raise ValueError(
                f"q_samples second dim ({q_samples.shape[1]}) does not match J ({len(self.chain_joint_names)})"
            )

        q_samples = q_samples.to(self.device)
        th = {name: q_samples[:, i] for i, name in enumerate(self.chain_joint_names)}
        ret = self.chain.forward_kinematics(th)
        # Returns {link_name: Tensor[B, 4, 4]} — homogeneous transforms in world/root frame.
        # Access position of link l for all samples: fk_out[l][:, :3, 3]  -> [B, 3]
        # Access rotation of link l for all samples: fk_out[l][:, :3, :3] -> [B, 3, 3]
        # Parent-relative rotation between P and C:  R_P.T @ R_C
        return {link: tf.get_matrix() for link, tf in ret.items()}

    # =========================================================================
    # STAGE 2 — DONG-STYLE KINEMATICS FROM FK POSITIONS
    # =========================================================================
    def run_dong_stage2(
        self,
        fk_out: dict[str, torch.Tensor],
        hand_config_path: str | Path,
    ) -> tuple[torch.Tensor, list[str], dict]:
        """
        Apply Dong Block 1 + Block 3 to FK link positions.

        Args:
            fk_out           : output of run_fk()
            hand_config_path : path to YAML config (see hand_configs/*.yaml)

        Returns:
            quats        : Tensor[B, N, 4]  Dong-convention quaternions (wxyz, w>=0)
            joint_labels : list[str]        label for each quaternion slot
            meta         : dict             R_wrist and per-finger rotation matrices
        """
        from dong_stage2 import run_stage2, load_config
        config = load_config(hand_config_path)
        return run_stage2(fk_out, config)

    # =========================================================================
    # STAGE 3 — PROJECT/FILTER TO THE COMPARABLE SUBSPACE FOR HUMAN-ROBOT USE
    #            (PLACEHOLDER, NOT IMPLEMENTED IN THIS VERSION)
    # =========================================================================

    def print_summary(self, num_samples: int, q_samples: torch.Tensor, fk_out: dict[str, torch.Tensor]) -> None:
        n_links = len(self.link_names)
        n_joint_total = len(self.joint_specs)
        n_joint_chain = len(self.chain_joint_names)
        n_actuated_xml = sum(1 for s in self.joint_specs.values() if s.joint_type in ACTUATED_TYPES)
        n_mimic = sum(1 for s in self.specs_in_chain_order if s.mimic_parent is not None)

        print("=== URDF Randomizer Summary ===")
        print(f"urdf_path        : {self.urdf_path}")
        print(f"robot_name       : {self.robot_name}")
        print(f"device           : {self.device}")
        print(f"links (xml)      : {n_links}")
        print(f"joints (xml)     : {n_joint_total}")
        print(f"actuated (xml)   : {n_actuated_xml}")
        print(f"chain joints (J) : {n_joint_chain}")
        print(f"mimic joints     : {n_mimic}")
        print(f"num_samples (B)  : {num_samples}")
        print(f"q_samples shape  : {tuple(q_samples.shape)}")
        print(f"fk links output  : {len(fk_out)}")

        print("\nfirst 10 chain joints + limits:")
        for i, spec in enumerate(self.specs_in_chain_order[:10]):
            lo, hi = self._resolve_limits(spec)
            mimic = f" mimic={spec.mimic_parent}" if spec.mimic_parent else ""
            print(f"  {i:02d} {spec.name:30s} type={spec.joint_type:10s} [{lo:+.5f}, {hi:+.5f}]{mimic}")

        # quick sanity stats
        q_min = q_samples.min(dim=0).values
        q_max = q_samples.max(dim=0).values
        print("\nfirst 10 sampled min/max:")
        for i, name in enumerate(self.chain_joint_names[:10]):
            print(f"  {i:02d} {name:30s} min={q_min[i].item():+.5f} max={q_max[i].item():+.5f}")

        fk_links_preview = list(fk_out.keys())[:10]
        print("\nfirst 10 FK links:")
        for ln in fk_links_preview:
            print(f"  - {ln}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URDF random joint sampler + FK runner (generic).")
    p.add_argument("--urdf", required=True, help="Path to URDF file.")
    p.add_argument("--num-samples", type=int, default=100000, help="How many random q samples to generate.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--continuous-range",
        type=float,
        default=math.pi,
        help="Sampling range for continuous joints: [-range, +range].",
    )
    p.add_argument("--hand-config", default=None, help="Path to hand YAML config for Stage 2.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    rnd = URDFRandomizer(
        urdf_path=args.urdf,
        device=device,
        continuous_range=args.continuous_range,
    )
    # STAGE 1
    q_samples, _ = rnd.sample_q(num_samples=args.num_samples, seed=args.seed)
    # STAGE 1
    fk_out = rnd.run_fk(q_samples)
    rnd.print_summary(num_samples=args.num_samples, q_samples=q_samples, fk_out=fk_out)
    # STAGE 2
    if args.hand_config:
        quats, labels, meta = rnd.run_dong_stage2(fk_out, args.hand_config)
        print(f"\n=== Stage 2 (Dong) ===")
        print(f"quats shape      : {tuple(quats.shape)}")
        print(f"joint labels ({len(labels)}): {labels}")
        print(f"quats[0]         :")
        for i, lbl in enumerate(labels):
            q = quats[0, i]
            print(f"  {lbl:20s}  [{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}]")


if __name__ == "__main__":
    main()
