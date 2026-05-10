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

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
import yaml


ACTUATED_TYPES = {"revolute", "prismatic", "continuous"}
EPS = 1e-8


# =============================================================================
# STAGE 2 — DONG-STYLE KINEMATICS (module-level geometry, robot-agnostic)
# =============================================================================

def _dong_normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=-1, eps=EPS)


def _dong_mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] -> [B, 4] wxyz with w >= 0."""
    q = pk.matrix_to_quaternion(R)
    q = torch.where(q[:, :1] < 0, -q, q)
    return q


def _dong_rotation_y(beta: torch.Tensor) -> torch.Tensor:
    """Ry(beta) for each sample. beta: [B] -> [B, 3, 3]."""
    c = beta.cos()
    s = beta.sin()
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack([ c, z, s], dim=-1)
    row1 = torch.stack([ z, o, z], dim=-1)
    row2 = torch.stack([-s, z, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _dong_block1_wrist_frame(
    wrist_pos: torch.Tensor,
    index_mcp: torch.Tensor,
    middle_mcp: torch.Tensor,
    ring_mcp: torch.Tensor,
) -> torch.Tensor:
    """Dong Eq. 5-7: FK positions -> wrist frame R_wrist [B, 3, 3]."""
    Y0 = _dong_normalize(ring_mcp - index_mcp)
    v  = _dong_normalize(middle_mcp - wrist_pos)
    Z0 = _dong_normalize(torch.linalg.cross(v, Y0))
    X0 = _dong_normalize(torch.linalg.cross(Y0, Z0))
    return torch.stack([X0, Y0, Z0], dim=-1)


def _dong_world_to_local(p: torch.Tensor, wrist_pos: torch.Tensor, R_wrist: torch.Tensor) -> torch.Tensor:
    diff = p - wrist_pos
    return torch.bmm(R_wrist.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)


def _dong_block3_mcp(mcp_local: torch.Tensor, pip_local: torch.Tensor) -> torch.Tensor:
    """Dong Eq. 20-22: MCP frame [B, 3, 3]."""
    Xi = _dong_normalize(pip_local - mcp_local)
    Z0 = torch.zeros_like(Xi); Z0[..., 2] = 1.0
    Yi = _dong_normalize(torch.linalg.cross(Z0, Xi))
    Zi = _dong_normalize(torch.linalg.cross(Xi, Yi))
    return torch.stack([Xi, Yi, Zi], dim=-1)


def _dong_block3_pip_dip(
    mcp_local: torch.Tensor,
    pip_local: torch.Tensor,
    dip_local: torch.Tensor,
    tip_local: torch.Tensor,
    R_mcp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dong Eq. 29-34: PIP and DIP flexion angles. Returns R_pip, R_dip [B,3,3]."""
    R_mcp_T = R_mcp.transpose(-1, -2)

    def to_mcp(p: torch.Tensor) -> torch.Tensor:
        return torch.bmm(R_mcp_T, (p - mcp_local).unsqueeze(-1)).squeeze(-1)

    dj  = to_mcp(pip_local)
    dj1 = to_mcp(dip_local)
    dj2 = to_mcp(tip_local)

    xj   = dj1 - dj
    xjm1 = dj
    cos_pip = (xj * xjm1).sum(-1) / (xj.norm(dim=-1).clamp(min=EPS) * xjm1.norm(dim=-1).clamp(min=EPS))
    beta_pip = torch.arccos(cos_pip.clamp(-1 + EPS, 1 - EPS))

    xj1 = dj2 - dj1
    cos_dip = (xj1 * xj).sum(-1) / (xj1.norm(dim=-1).clamp(min=EPS) * xj.norm(dim=-1).clamp(min=EPS))
    beta_dip = torch.arccos(cos_dip.clamp(-1 + EPS, 1 - EPS))

    return _dong_rotation_y(beta_pip), _dong_rotation_y(beta_dip)


def _dong_block3_pip_only(
    mcp_local: torch.Tensor,
    pip_local: torch.Tensor,
    tip_local: torch.Tensor,
    R_mcp: torch.Tensor,
) -> torch.Tensor:
    """PIP angle for 2-link fingers. Returns R_pip [B,3,3]."""
    R_mcp_T = R_mcp.transpose(-1, -2)

    def to_mcp(p: torch.Tensor) -> torch.Tensor:
        return torch.bmm(R_mcp_T, (p - mcp_local).unsqueeze(-1)).squeeze(-1)

    dj  = to_mcp(pip_local)
    dj1 = to_mcp(tip_local)
    xj   = dj1 - dj
    xjm1 = dj
    cos_pip = (xj * xjm1).sum(-1) / (xj.norm(dim=-1).clamp(min=EPS) * xjm1.norm(dim=-1).clamp(min=EPS))
    beta_pip = torch.arccos(cos_pip.clamp(-1 + EPS, 1 - EPS))
    return _dong_rotation_y(beta_pip)


def _dong_run_stage2(
    fk_out: dict[str, torch.Tensor],
    config: dict,
) -> tuple[torch.Tensor, list[str], dict]:
    """
    Dong Block 1 + Block 3 applied to FK link positions.

    Returns:
        quats        : Tensor[B, N, 4]  wxyz quaternions (w >= 0)
        joint_labels : list[str]        label for each slot
        meta         : dict             R_wrist, per-finger rotations,
                                        tips [B,F,3] (wrist-local, unnormalized),
                                        tip_labels list[str]
    """
    def pos(link: str) -> torch.Tensor:
        return fk_out[link][:, :3, 3]

    wrist_pos  = pos(config["wrist_link"])
    index_mcp  = pos(config["frame_index_mcp"])
    middle_mcp = pos(config["frame_middle_mcp"])
    ring_mcp   = pos(config["frame_ring_mcp"])

    R_wrist = _dong_block1_wrist_frame(wrist_pos, index_mcp, middle_mcp, ring_mcp)

    def local(link: str) -> torch.Tensor:
        return _dong_world_to_local(pos(link), wrist_pos, R_wrist)

    quats_list: list[torch.Tensor] = []
    labels: list[str] = []
    finger_meta: dict = {}
    tips_list: list[torch.Tensor] = []
    tip_labels: list[str] = []
    chain_positions: dict[str, torch.Tensor] = {}

    for finger_name, finger_cfg in config["fingers"].items():
        chain = finger_cfg["chain"]
        n = len(chain)
        if n < 2:
            continue

        chain_local = [local(link) for link in chain]  # list of [B, 3]
        chain_positions[finger_name] = torch.stack(chain_local, dim=1)  # [B, L, 3]

        mcp_l = chain_local[0]
        pip_l = chain_local[1]
        tip_l = chain_local[-1]
        tips_list.append(tip_l)
        tip_labels.append(finger_name)

        R_mcp = _dong_block3_mcp(mcp_l, pip_l)
        q_mcp = _dong_mat_to_quat(R_mcp)
        quats_list.append(q_mcp)
        labels.append(f"{finger_name}_mcp")

        if n == 2:
            finger_meta[finger_name] = {"R_mcp": R_mcp}
            continue

        if n == 3:
            R_pip = _dong_block3_pip_only(mcp_l, pip_l, tip_l, R_mcp)
            q_pip = _dong_mat_to_quat(R_pip)
            quats_list.append(q_pip)
            labels.append(f"{finger_name}_pip")
            finger_meta[finger_name] = {"R_mcp": R_mcp, "R_pip": R_pip}
            continue

        dip_l = local(chain[2])
        R_pip, R_dip = _dong_block3_pip_dip(mcp_l, pip_l, dip_l, tip_l, R_mcp)
        q_pip = _dong_mat_to_quat(R_pip)
        q_dip = _dong_mat_to_quat(R_dip)
        quats_list.append(q_pip)
        labels.append(f"{finger_name}_pip")
        quats_list.append(q_dip)
        labels.append(f"{finger_name}_dip")
        finger_meta[finger_name] = {"R_mcp": R_mcp, "R_pip": R_pip, "R_dip": R_dip}

    quats = torch.stack(quats_list, dim=1)
    tips  = torch.stack(tips_list, dim=1)

    meta = {
        "R_wrist": R_wrist,
        "fingers": finger_meta,
        "wrist_pos": wrist_pos,
        "tips": tips,
        "tip_labels": tip_labels,
        "chain_positions": chain_positions,  # {finger: [B, L, 3]} all chain links in palm frame (unnormalized)
    }
    return quats, labels, meta


def _load_hand_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class JointSpec:
    name: str
    joint_type: str
    lower: float | None
    upper: float | None
    mimic_parent: str | None
    mimic_multiplier: float
    mimic_offset: float


class RobotLoader:
    def __init__(
        self,
        urdf_path: str | Path,
        device: str = "cpu",
        continuous_range: float = math.pi,
        valid_poses_path: str | Path | None = None,
    ):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.device = torch.device(device)
        self.continuous_range = float(continuous_range)

        urdf_bytes = self.urdf_path.read_bytes()
        self.chain = pk.build_chain_from_urdf(urdf_bytes).to(device=self.device)
        self.chain_joint_names = list(self.chain.get_joint_parameter_names())
        self.joint_specs, self.robot_name, self.link_names = self._parse_urdf_metadata(self.urdf_path)

        # Precompute specs in chain order and report missing metadata.
        self.specs_in_chain_order = self._build_specs_in_chain_order()

        # Sampling mode: NPZ pool or random uniform.
        if valid_poses_path is not None:
            p = Path(valid_poses_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(
                    f"valid_poses_path not found: {p}\n"
                    f"Generate it first: python grasp-model/scripts/generate_valid_robot_poses.py"
                )
            data = np.load(p)
            self._valid_poses = torch.from_numpy(data["q"]).to(self.device)
            print(f"[RobotLoader] mode=VALID_NPZ  path={p}  n_poses={len(self._valid_poses):,}")
        else:
            self._valid_poses = None
            print(f"[RobotLoader] mode=RANDOM_UNIFORM  (no collision filtering)")

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

        if self._valid_poses is not None:
            if seed is not None:
                torch.manual_seed(int(seed))
            idx = torch.randint(0, len(self._valid_poses), (num_samples,), device=self.device)
            return self._valid_poses[idx], self.chain_joint_names

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
    def _get_hand_length(self, config: dict) -> float:
        """Path length along middle finger chain at q=0 (pose-invariant). Computed once and cached."""
        if not hasattr(self, "_hand_length"):
            q_zero = torch.zeros(1, len(self.chain_joint_names), device=self.device)
            fk_zero = self.run_fk(q_zero)
            _, _, m = _dong_run_stage2(fk_zero, config)
            # Sum of segment lengths along middle finger chain (palm origin → each chain link)
            # chain_positions["middle"] is [1, L, 3] in palm frame (unnormalized)
            pts = m["chain_positions"]["middle"][0]  # [L, 3]
            # Add palm origin (0,0,0) as first point
            origin = torch.zeros(1, 3, device=pts.device)
            pts_full = torch.cat([origin, pts], dim=0)  # [L+1, 3]
            diffs = pts_full[1:] - pts_full[:-1]        # [L, 3]
            self._hand_length = diffs.norm(dim=-1).sum().item()
        return self._hand_length

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
            meta         : dict             R_wrist, per-finger rotations,
                                            tips [B,F,3] (wrist-local, normalized by hand_length),
                                            tip_labels list[str],
                                            hand_length float
        """
        config = _load_hand_config(hand_config_path)
        quats, labels, meta = _dong_run_stage2(fk_out, config)
        hand_length = self._get_hand_length(config)
        meta["tips"] = meta["tips"] / hand_length
        meta["chain_positions"] = {f: v / hand_length for f, v in meta["chain_positions"].items()}
        meta["hand_length"] = hand_length
        return quats, labels, meta

    def sample_dong(
        self,
        num_samples: int,
        hand_config_path: str | Path,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], dict]:
        """
        Sample random configurations and compute Dong quaternions in one call.

        Returns:
            q            : Tensor[B, J]    sampled joint angles
            quats        : Tensor[B, N, 4] Dong quaternions (wxyz, w>=0)
            joint_labels : list[str]
            meta         : dict (R_wrist, tips [B,F,3] normalized, hand_length, tip_labels)
        """
        q, _ = self.sample_q(num_samples, seed)
        fk_out = self.run_fk(q)
        config = _load_hand_config(hand_config_path)
        quats, labels, meta = _dong_run_stage2(fk_out, config)
        hand_length = self._get_hand_length(config)
        meta["tips"] = meta["tips"] / hand_length
        meta["chain_positions"] = {f: v / hand_length for f, v in meta["chain_positions"].items()}
        meta["hand_length"] = hand_length
        return q, quats, labels, meta

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

    rnd = RobotLoader(
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
