#!/usr/bin/env python3
"""
Generic URDF randomizer + FK sampler.

Loads any URDF, extracts actuated joints and limits, samples random joint
configurations uniformly inside limits, and runs FK.
Dong-style kinematics (Block 1 + 3) are in dong_math.py.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import pytorch_kinematics as pk
import yaml

from .dong_math import dong_run_stage2, _dong_block1_wrist_frame, _dong_world_to_local

# Backward-compat alias used by precompute scripts that imported _dong_run_stage2 directly.
_dong_run_stage2 = dong_run_stage2


def _quat_wxyz_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] wxyz quaternion → [..., 6] rot6d (first two cols of rotation matrix).

    Used to compute rot6 on-the-fly from cached quats, avoiding 3.6 GB of VRAM
    that would otherwise be needed to pre-cache the rot6 field.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    c1 = torch.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], dim=-1)
    c2 = torch.stack([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], dim=-1)
    return torch.cat([c1, c2], dim=-1)

ACTUATED_TYPES = {"revolute", "prismatic", "continuous"}

_PRIM_TYPE_NAMES = ["FLEX", "ABD", "ROT"]


def _parse_joint_to_child(urdf_path: Path) -> dict[str, str]:
    root = ET.parse(urdf_path).getroot()
    result: dict[str, str] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        child = joint.find("child")
        if name and child is not None:
            result[name] = child.attrib.get("link", "")
    return result


def _parse_link_to_parent(urdf_path: Path) -> dict[str, str]:
    """Returns {child_link: parent_link} from URDF joint data."""
    root = ET.parse(urdf_path).getroot()
    result: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el  = joint.find("child")
        if parent_el is not None and child_el is not None:
            result[child_el.attrib.get("link", "")] = parent_el.attrib.get("link", "")
    return result


_HAND_CONFIG_CACHE: dict[str, dict] = {}


def _load_hand_config(path: str | Path) -> dict:
    key = str(Path(path).expanduser().resolve())
    cached = _HAND_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _HAND_CONFIG_CACHE[key] = cfg
    return cfg


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
        primitive_sample: bool = False,
        eigengrasp_path: str | Path | None = None,
        mjcf_path: str | Path | None = None,
        n_knobs: int = 9,
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

        # Sampling mode: DONG_CACHE > PRIMITIVE_SAMPLE > RANDOM_UNIFORM.
        # DONG_CACHE: pre-computed NPZ with Dong fields.
        # PRIMITIVE_SAMPLE: DexGrasp-Zero M_h primitive space; no NPZ needed.
        # RANDOM_UNIFORM: fallback, joints sampled independently.
        self._dong_cache: dict | None = None
        self._primitive_mode: bool = False
        self._primitives_lazy: dict[str, tuple] = {}  # hcp_key -> (primitives, jt_finger)
        self._eigengrasp: dict | None = None
        self._mj_model = None
        self._mj_data  = None
        self._n_knobs  = n_knobs
        if valid_poses_path is not None:
            p = Path(valid_poses_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(
                    f"valid_poses_path not found: {p}\n"
                    f"Generate it first: python models/latent-retargeting/scripts/generate_valid_robot_poses.py"
                )
            data = np.load(p)
            self._valid_poses = torch.from_numpy(data["q"]).to("cpu")
            cache_keys = ("quats", "chain", "tips", "joint_labels", "tip_labels")
            if all(k in data.files for k in cache_keys):
                quats_np  = data["quats"]
                chain_np  = data["chain"]
                tips_np   = data["tips"]
                joint_labels = [str(s) for s in data["joint_labels"]]
                tip_labels   = [str(s) for s in data["tip_labels"]]
                cached_cfg = str(data["hand_config"]) if "hand_config" in data.files else None
                has_rot6_npz = "rot6" in data.files
                self._dong_cache = {
                    "quats": torch.from_numpy(quats_np).to("cpu"),
                    "chain": torch.from_numpy(chain_np).to("cpu"),
                    "tips":  torch.from_numpy(tips_np).to("cpu"),
                    "joint_labels": joint_labels,
                    "tip_labels":   tip_labels,
                    "hand_config":  cached_cfg,
                }
                # rot6 is NOT loaded into VRAM — computed on-the-fly from quats
                # to save ~3.6 GB of device memory.
                print(
                    f"[RobotLoader] mode=DONG_CACHE(cpu-pinned)  path={p}  n_poses={len(self._valid_poses):,}  "
                    f"quats={tuple(quats_np.shape)} "
                    f"rot6={'npz_present(on-the-fly)' if has_rot6_npz else 'missing'} "
                    f"chain={tuple(chain_np.shape)} tips={tuple(tips_np.shape)}"
                )
                if cached_cfg is not None:
                    print(f"[RobotLoader] cached hand_config={cached_cfg}")
            else:
                print(f"[RobotLoader] mode=VALID_NPZ  path={p}  n_poses={len(self._valid_poses):,}")
        else:
            self._valid_poses = None
            if primitive_sample:
                self._primitive_mode = True
                print(f"[RobotLoader] mode=PRIMITIVE_SAMPLE  (DexGrasp-Zero M_h, primitives built on first sample_dong call)")
            else:
                print(f"[RobotLoader] mode=RANDOM_UNIFORM  (no collision filtering)")

        # EIGENGRASP_ONLINE: load PCA basis + MuJoCo model for on-the-fly
        # collision-filtered sampling. Overrides RANDOM_UNIFORM/PRIMITIVE_SAMPLE
        # if both eigengrasp_path and mjcf_path are provided. DONG_CACHE takes
        # precedence over everything (npz already contains collision-free poses).
        if eigengrasp_path is not None and self._dong_cache is None:
            import mujoco
            ep = Path(eigengrasp_path).expanduser().resolve()
            mp = Path(mjcf_path).expanduser().resolve() if mjcf_path else None
            if not ep.exists():
                raise FileNotFoundError(f"eigengrasp_path not found: {ep}")
            if mp is None or not mp.exists():
                raise FileNotFoundError(f"mjcf_path required and not found: {mjcf_path}")
            d = np.load(ep)
            self._eigengrasp = {
                "mean_norm":       d["mean_norm"].astype(np.float32),
                "components_norm": d["components_norm"].astype(np.float32),
                "joint_low":       d["joint_low"].astype(np.float32),
                "joint_high":      d["joint_high"].astype(np.float32),
                "coeff_p01":       d["coeff_p01"].astype(np.float32),
                "coeff_p99":       d["coeff_p99"].astype(np.float32),
            }
            self._mj_model = mujoco.MjModel.from_xml_path(str(mp))
            self._mj_data  = mujoco.MjData(self._mj_model)
            n_eigen_joints = self._eigengrasp["mean_norm"].shape[0]
            self._eigen_n_pad = self._mj_model.nq - n_eigen_joints
            if self._eigen_n_pad < 0:
                raise ValueError(f"MJCF nq ({self._mj_model.nq}) < eigengrasp joints ({n_eigen_joints})")
            self._primitive_mode = False  # disable primitive mode if eigengrasp takes over
            print(
                f"[RobotLoader] mode=EIGENGRASP_ONLINE  eigen={ep.name}  mjcf={mp.name}  "
                f"n_knobs={n_knobs}  n_pad={self._eigen_n_pad}"
            )

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

    def run_dong_tips_only(
        self,
        fk_out: dict[str, torch.Tensor],
        hand_config_path: str | Path,
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Lightweight Dong path for L_temp: compute only fingertip positions in
        wrist-local frame (normalized by hand_length). Skips block3 rotations
        and `_dong_mat_to_quat` calls done by `run_dong_stage2`.

        Math identical to taking `run_dong_stage2(fk_out, ...)` and reading
        `meta["tips"]`. Verified by parity test.

        Returns:
            tips       : Tensor[B, F, 3]
            tip_labels : list[str]   len F
        """
        config = _load_hand_config(hand_config_path)

        def pos(link: str) -> torch.Tensor:
            return fk_out[link][:, :3, 3]

        wrist_pos  = pos(config["wrist_link"])
        index_mcp  = pos(config["frame_index_mcp"])
        middle_mcp = pos(config["frame_middle_mcp"])
        ring_mcp   = pos(config["frame_ring_mcp"])
        R_wrist = _dong_block1_wrist_frame(wrist_pos, index_mcp, middle_mcp, ring_mcp)

        tips_list: list[torch.Tensor] = []
        tip_labels: list[str] = []
        for finger_name, finger_cfg in config["fingers"].items():
            chain = finger_cfg["chain"]
            if len(chain) < 2:
                continue
            tip_world = pos(chain[-1])
            tip_local = _dong_world_to_local(tip_world, wrist_pos, R_wrist)
            tips_list.append(tip_local)
            tip_labels.append(finger_name)

        tips = torch.stack(tips_list, dim=1)            # [B, F, 3] wrist-local
        hand_length = self._get_hand_length(config)
        tips = tips / hand_length
        return tips, tip_labels

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
        rot_repr: str = "quat",
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], dict]:
        """
        Sample random configurations and compute Dong pose in one call.

        Returns:
            q            : Tensor[B, J]    sampled joint angles
            pose         : Tensor[B, N, 4] (quat) or [B, N, 6] (r6)
            joint_labels : list[str]
            meta         : dict (tips [B,F,3] normalized, tip_labels, chain_positions)
        """
        if rot_repr not in {"quat", "r6"}:
            raise ValueError(f"rot_repr must be quat or r6, got {rot_repr!r}")

        if self._dong_cache is not None and self._valid_poses is not None:
            if seed is not None:
                torch.manual_seed(int(seed))
            idx = torch.randint(0, len(self._valid_poses), (num_samples,))  # CPU — tensors are cpu-pinned
            q     = self._valid_poses[idx].to(self.device)
            quats_batch = self._dong_cache["quats"][idx].to(self.device)  # [B, Jk, 4]
            if rot_repr == "quat":
                pose = quats_batch
            else:
                pose = _quat_wxyz_to_rot6d(quats_batch)                  # [B, Jk, 6] on-the-fly
            chain = self._dong_cache["chain"][idx].to(self.device)       # [B, F, 4, 3]
            tips  = self._dong_cache["tips"][idx].to(self.device)        # [B, F, 3]
            tip_labels = self._dong_cache["tip_labels"]
            chain_positions = {f: chain[:, fi] for fi, f in enumerate(tip_labels)}
            meta = {
                "tips":            tips,
                "tip_labels":      tip_labels,
                "chain_positions": chain_positions,
                "rot6":            _quat_wxyz_to_rot6d(quats_batch),     # always available, no VRAM cost
            }
            return q, pose, self._dong_cache["joint_labels"], meta

        if self._eigengrasp is not None:
            q = self._sample_eigengrasp_q(num_samples, seed)
        elif self._primitive_mode:
            q = self._sample_primitive_q(num_samples, hand_config_path, seed)
        else:
            q, _ = self.sample_q(num_samples, seed)
        fk_out = self.run_fk(q)
        config = _load_hand_config(hand_config_path)
        quats, labels, meta = _dong_run_stage2(fk_out, config)
        hand_length = self._get_hand_length(config)
        meta["tips"] = meta["tips"] / hand_length
        meta["chain_positions"] = {f: v / hand_length for f, v in meta["chain_positions"].items()}
        meta["hand_length"] = hand_length
        pose = quats if rot_repr == "quat" else meta["rot6"]
        return q, pose, labels, meta

    # =========================================================================
    # PRIMITIVE SAMPLING — DexGrasp-Zero M_h construction (S1.B)
    # =========================================================================
    def _build_motion_primitives(
        self,
        hand_config_path: str | Path,
        joint_to_finger: dict[str, str | None] | None = None,
    ) -> dict[str, dict]:
        """Classify each joint as FLEX/ABD/ROT via unit excitation in palm frame.

        For each joint j, excites it by a small delta (all others fixed) and
        measures the FINGERTIP displacement in the wrist/palm frame. Dominant
        axis = primitive type (X=FLEX, Y=ABD, Z=ROT in Dong palm frame).
        Pure kinematics — equivalent to DexGrasp-Zero S1.B without physics.
        """
        config = _load_hand_config(hand_config_path)
        J = len(self.chain_joint_names)

        if joint_to_finger is None:
            joint_to_finger = self._build_joint_to_finger(hand_config_path)

        # Fingertip link per finger (last link in chain).
        finger_to_tip = {
            fname: fcfg["chain"][-1]
            for fname, fcfg in config["fingers"].items()
            if fcfg.get("chain")
        }

        # Row 0 = q=0 (baseline), rows 1..J = unit excitation per joint.
        q_batch = torch.zeros(J + 1, J, device=self.device)
        for j, jname in enumerate(self.chain_joint_names):
            spec = self.joint_specs.get(jname)
            lo, hi = self._resolve_limits(spec) if spec else (-math.pi, math.pi)
            dq = 0.05 * (hi - lo) if (hi - lo) > 1e-6 else 0.05
            q_batch[j + 1, j] = dq

        fk_out = self.run_fk(q_batch)

        def _pos(link: str, i: int) -> torch.Tensor:
            return fk_out[link][i, :3, 3]

        wrist_pos  = _pos(config["wrist_link"],       0).unsqueeze(0)
        index_mcp  = _pos(config["frame_index_mcp"],  0).unsqueeze(0)
        middle_mcp = _pos(config["frame_middle_mcp"], 0).unsqueeze(0)
        ring_mcp   = _pos(config["frame_ring_mcp"],   0).unsqueeze(0)
        R_wrist = _dong_block1_wrist_frame(wrist_pos, index_mcp, middle_mcp, ring_mcp)[0]  # [3,3]

        primitives: dict[str, dict] = {}
        for j, jname in enumerate(self.chain_joint_names):
            fname = joint_to_finger.get(jname)
            if fname is None:
                primitives[jname] = {"type": "FLEX", "sign": 1}
                continue

            tip_link = finger_to_tip.get(fname)
            if tip_link is None or tip_link not in fk_out:
                primitives[jname] = {"type": "FLEX", "sign": 1}
                continue

            # Measure FINGERTIP displacement: exciting joint j moves the tip.
            # (Immediate child-link origin = joint pivot → zero displacement there.)
            p_base  = fk_out[tip_link][0,     :3, 3]
            p_delta = fk_out[tip_link][j + 1, :3, 3]
            disp_local = R_wrist.T @ (p_delta - p_base)  # [3] in palm frame

            if disp_local.abs().max().item() < 1e-7:
                primitives[jname] = {"type": "FLEX", "sign": 1}
                continue

            dom = int(disp_local.abs().argmax().item())
            primitives[jname] = {
                "type": _PRIM_TYPE_NAMES[dom],
                "sign": 1 if disp_local[dom].item() >= 0 else -1,
            }

        n_flex = sum(1 for v in primitives.values() if v["type"] == "FLEX")
        n_abd  = sum(1 for v in primitives.values() if v["type"] == "ABD")
        n_rot  = sum(1 for v in primitives.values() if v["type"] == "ROT")
        print(f"[RobotLoader] M_h built: FLEX={n_flex} ABD={n_abd} ROT={n_rot} (J={J})")
        return primitives

    def _build_joint_to_finger(self, hand_config_path: str | Path) -> dict[str, str | None]:
        """Assign each chain joint to a finger by kinematic ancestry.

        A joint belongs to finger f if its child_link is an ancestor of f's
        fingertip (and not shared with other fingers). Joints that influence
        multiple fingers (wrist, palm joints) are left unassigned (None).
        """
        config = _load_hand_config(hand_config_path)
        joint_to_child = _parse_joint_to_child(self.urdf_path)
        link_to_parent = _parse_link_to_parent(self.urdf_path)

        # For each finger, collect all ancestor links from tip up to root.
        finger_ancestor_sets: dict[str, set[str]] = {}
        for fname, fcfg in config["fingers"].items():
            chain = fcfg.get("chain", [])
            if not chain:
                continue
            ancestors: set[str] = set()
            link: str | None = chain[-1]
            while link:
                ancestors.add(link)
                link = link_to_parent.get(link)
            finger_ancestor_sets[fname] = ancestors

        # Joint j → finger f only if j's child is an ancestor of EXACTLY one finger.
        result: dict[str, str | None] = {}
        for jname in self.chain_joint_names:
            child = joint_to_child.get(jname)
            matched = [f for f, anc in finger_ancestor_sets.items() if child and child in anc]
            result[jname] = matched[0] if len(matched) == 1 else None
        return result

    def _sample_eigengrasp_q(self, num_samples: int, seed: int | None = None) -> torch.Tensor:
        """Sample collision-free joint angles from eigengrasp PCA space.

        Equivalent to generate_valid_robot_poses.py inner loop, run online.
        Acceptance rate ~56% for Allegro, ~60% for Shadow — samples ~2x num_samples
        candidates per call on average.
        """
        import mujoco
        rng   = np.random.default_rng(seed)
        eg    = self._eigengrasp
        model = self._mj_model
        data  = self._mj_data
        n_pad = self._eigen_n_pad
        n_k   = min(self._n_knobs, eg["coeff_p01"].shape[0])
        p01   = eg["coeff_p01"][:n_k]
        p99   = eg["coeff_p99"][:n_k]
        mean  = eg["mean_norm"]
        comps = eg["components_norm"][:n_k]          # [n_k, J]
        jlow  = eg["joint_low"]
        jhigh = eg["joint_high"]

        valid: list[np.ndarray] = []
        batch = max(num_samples * 2, 256)            # oversample; 2x covers ~56% acceptance
        while len(valid) < num_samples:
            coeffs = rng.uniform(p01, p99, size=(batch, n_k)).astype(np.float32)
            q_norm = mean + coeffs @ comps           # [batch, J]
            q_joints = q_norm * (jhigh - jlow) + jlow
            q_joints = np.clip(q_joints, jlow, jhigh)
            if n_pad > 0:
                pad = np.zeros((batch, n_pad), dtype=np.float32)
                q_full = np.concatenate([pad, q_joints], axis=1)
            else:
                q_full = q_joints
            for i in range(batch):
                data.qpos[:] = q_full[i]
                mujoco.mj_forward(model, data)
                if data.ncon == 0:
                    valid.append(q_full[i])
                    if len(valid) == num_samples:
                        break
            batch = max(256, (num_samples - len(valid)) * 2)

        q_np = np.stack(valid[:num_samples], axis=0).astype(np.float32)
        return torch.from_numpy(q_np).to(self.device)

    def _sample_primitive_q(
        self,
        num_samples: int,
        hand_config_path: str | Path,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample [B, J] joint angles coupled by finger/primitive type (DexGrasp-Zero)."""
        hcp_key = str(Path(hand_config_path).expanduser().resolve())
        if hcp_key not in self._primitives_lazy:
            jt_fgr = self._build_joint_to_finger(hand_config_path)
            prims  = self._build_motion_primitives(hand_config_path, joint_to_finger=jt_fgr)
            self._primitives_lazy[hcp_key] = (prims, jt_fgr)
        primitives, joint_to_finger = self._primitives_lazy[hcp_key]

        J = len(self.chain_joint_names)
        B = num_samples
        if seed is not None:
            torch.manual_seed(int(seed))

        fingers = sorted(set(f for f in joint_to_finger.values() if f is not None))

        # One activation scalar per (finger, primitive_type).
        # FLEX ∈ [0,1] (uni-directional flexion), ABD/ROT ∈ [-1,1].
        activations: dict[tuple, torch.Tensor] = {}
        for fname in fingers:
            activations[(fname, "FLEX")] = torch.rand(B, device=self.device)
            activations[(fname, "ABD")]  = torch.rand(B, device=self.device) * 2 - 1
            activations[(fname, "ROT")]  = torch.rand(B, device=self.device) * 2 - 1

        q = torch.zeros(B, J, device=self.device)
        mimic_deferred: list[tuple[int, JointSpec]] = []

        for i, jname in enumerate(self.chain_joint_names):
            spec = self.joint_specs.get(jname)
            if spec and spec.mimic_parent:
                mimic_deferred.append((i, spec))
                continue

            lo, hi = self._resolve_limits(spec) if spec else (-math.pi, math.pi)
            fname = joint_to_finger.get(jname)
            prim  = primitives.get(jname, {"type": "FLEX", "sign": 1})

            if fname is None:
                q[:, i] = lo + torch.rand(B, device=self.device) * (hi - lo)
                continue

            alpha = activations.get((fname, prim["type"]))
            if alpha is None:
                q[:, i] = lo + torch.rand(B, device=self.device) * (hi - lo)
                continue

            if prim["type"] == "FLEX":
                # Flexion is directional: map [0,1] into joint range aligned with sign.
                if prim["sign"] > 0:
                    q[:, i] = lo + alpha * (hi - lo)
                else:
                    q[:, i] = hi - alpha * (hi - lo)
            else:
                # ABD/ROT: [-1,1] → [lo, hi], scaled by primitive sign.
                mid  = (lo + hi) * 0.5
                half = (hi - lo) * 0.5
                q[:, i] = mid + prim["sign"] * alpha * half

        # Resolve mimic joints from parents (same logic as sample_q).
        values = {jname: q[:, i] for i, jname in enumerate(self.chain_joint_names)}
        for i, spec in mimic_deferred:
            parent = spec.mimic_parent
            if parent and parent in values:
                v = values[parent] * spec.mimic_multiplier + spec.mimic_offset
                lo, hi = self._resolve_limits(spec)
                q[:, i] = torch.clamp(v, lo, hi)
                values[spec.name] = q[:, i]
            else:
                lo, hi = self._resolve_limits(spec)
                q[:, i] = lo + torch.rand(B, device=self.device) * (hi - lo)

        return q

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
