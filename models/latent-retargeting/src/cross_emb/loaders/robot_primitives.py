"""Function 1 -- build the per-joint primitive table (M_h, DexGrasp-Zero S1.2).

Tells, for a given robot, which qpos column fills which UDHM finger slot, with a
sign. This is the ONLY robot-specific lookup; Function 2 (robot_qpos_to_udhm)
consumes it and is hand-agnostic.

Output:
    {finger: {role: (qpos_col, sign)}}   role in {mcp_flex, mcp_abd, pip, dip}

Paper-faithful (DexGrasp-Zero):
  - Functional partitioning (S1.1): a finger's joints are grouped by position
    along the kinematic chain into semantic nodes proximal/middle/distal.
    Separated root joints (e.g. LEAP MCP flex + abd) share the proximal node
    (S1.1.a) -> they fill mcp_flex and mcp_abd of the same finger.
  - Primitive per joint (S1.2): the type comes from the joint rotation axis vs the
    finger triad (||lateral = FLEX, ||palm-normal = ABD, ||bone = ROT). ROT and
    metacarpal joints have no UDHM finger slot -> dropped.
  Node -> UDHM slot: proximal = mcp, middle = pip, distal = dip.

Sign makes flexion positive (matches the human Dong convention). It only matters
for pip/dip (udhm_run_stage3 reads those with signed atan2; mcp via arccos folds
the sign). Rule: flexion is the joint's larger range-of-motion side, so
``sign = +1 if |hi| >= |lo| else -1`` from the URDF joint limits. Deterministic;
no excitation/pivot issues.

Manual overrides (S1.2 sanctions manual annotation) are used where the automatic
palm frame is invalid (Barrett: 3-finger radial) or a DoF has no human analog.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from cross_emb.loaders.dong_math import _dong_block1_wrist_frame
from cross_emb.loaders.robot_loader import _load_hand_config
from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS

# Manual (finger, role) overrides, keyed by config["robot"]. None excludes a joint.
# Sign is NOT here -- it is always computed from joint limits (one rule for all).
#   role in {mcp_flex, mcp_abd, pip, dip}
_MANUAL: dict[str, dict[str, tuple[str, str] | None]] = {
    # Barrett (annotated with mujoco_primitive_annotator). 3-finger radial palm ->
    # auto axis classifier unreliable. finger_3 = thumb (the opposing finger).
    # Spread (prox of finger_1/2, +-180deg) has no human analog -> excluded.
    "barrett_hand": {
        "finger_1_prox_joint": None,
        "finger_2_prox_joint": None,
        "finger_1_med_joint": ("index", "mcp_flex"),
        "finger_1_dist_joint": ("index", "pip"),
        "finger_2_med_joint": ("middle", "mcp_flex"),
        "finger_2_dist_joint": ("middle", "pip"),
        "finger_3_med_joint": ("thumb", "mcp_flex"),
        "finger_3_dist_joint": ("thumb", "pip"),
    },
    # Inspire hand: thumb yaw misclassified as ROT (dropped) and pitch as ABD.
    # Manual override: yaw=mcp_abd (lateral spread), pitch=mcp_flex.
    # Mimic joints (intermediate, distal) included as pip; non-thumb auto-classify is correct.
    "inspire_hand": {
        "thumb_proximal_yaw_joint":       ("thumb", "mcp_abd"),
        "thumb_proximal_pitch_joint":     ("thumb", "mcp_flex"),
        "thumb_intermediate_joint":       ("thumb", "pip"),
        "thumb_distal_joint":             None,   # mimic, skip (no ip_flex slot in UDHM22)
    },
}

_FLEX_ROLES = ("mcp_flex", "pip", "dip")  # proximal -> distal destinations for FLEX joints


def _parse_joint_axes(urdf_path: str | Path) -> dict[str, tuple[str, np.ndarray]]:
    """joint_name -> (child_link, axis_xyz) for actuated revolute joints."""
    root = ET.parse(str(urdf_path)).getroot()
    out: dict[str, tuple[str, np.ndarray]] = {}
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        child, axis = j.find("child"), j.find("axis")
        if child is None:
            continue
        xyz = [float(v) for v in (axis.get("xyz") if axis is not None else "0 0 1").split()]
        out[j.get("name")] = (child.get("link"), np.asarray(xyz, dtype=float))
    return out


def _flex_sign(loader, joint_name: str) -> float:
    """Flexion = the joint's larger range-of-motion side. Makes flexion positive."""
    spec = loader.joint_specs.get(joint_name)
    lo, hi = loader._resolve_limits(spec) if spec is not None else (-np.pi, np.pi)
    return 1.0 if abs(hi) >= abs(lo) else -1.0


def build_primitives(loader, hand_config_path: str | Path) -> dict[str, dict[str, tuple[int, float]]]:
    """Build the primitive table {finger: {role: (qpos_col, sign)}}. Run once per robot."""
    config = _load_hand_config(hand_config_path)
    cj: list[str] = loader.chain_joint_names
    manual = _MANUAL.get(config.get("robot", ""), {})
    jaxes = _parse_joint_axes(loader.urdf_path)
    jt_finger = loader._build_joint_to_finger(hand_config_path)

    # Palm frame + per-finger bone direction, at the zero pose (for axis classification).
    fk0 = loader.run_fk(torch.zeros(1, len(cj), device=loader.device))
    pos = lambda link: fk0[link][0, :3, 3]
    Rw = _dong_block1_wrist_frame(
        pos(config["wrist_link"])[None], pos(config["frame_index_mcp"])[None],
        pos(config["frame_middle_mcp"])[None], pos(config["frame_ring_mcp"])[None],
    )[0]
    RwT = Rw.t().cpu().numpy()
    wrist_local = RwT @ pos(config["wrist_link"]).cpu().numpy()
    # bone base/tip = each finger's OWN chain ends (chain[0]..chain[-1]). Using a
    # shared mcp dict here was wrong: it lacked pinky/thumb -> their bone was taken
    # from the INDEX knuckle, corrupting their axis classification (e.g. the pinky
    # metacarpal LFJ5 misclassified as FLEX instead of ROT/twist).
    base_link = {f: c["chain"][0] for f, c in config["fingers"].items() if c.get("chain")}
    tip_link = {f: c["chain"][-1] for f, c in config["fingers"].items() if c.get("chain")}

    def bone(finger: str) -> np.ndarray:
        v = RwT @ (pos(tip_link[finger]) - pos(base_link[finger])).cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-9)

    # Abduction sign = "away from the middle finger axis" (DexGrasp S1, Eq. ABD):
    # value > 0 when the finger spreads AWAY from middle. Determined geometrically:
    # excite the joint +, see if the fingertip moves further from the middle finger
    # (same lateral side) -> +; toward it -> -. Lateral = palm-frame Y. The limit
    # rule does NOT work for abduction (symmetric range -> no info).
    Jn = len(cj)
    qexc = torch.zeros(Jn + 1, Jn, device=loader.device)
    for j in range(Jn):
        qexc[j + 1, j] = 0.1
    fke = loader.run_fk(qexc)
    mid_tip_Y = float((RwT @ fke[tip_link["middle"]][0, :3, 3].cpu().numpy())[1])

    def abd_away_sign(col: int, finger: str) -> float:
        tipY0 = float((RwT @ fke[tip_link[finger]][0, :3, 3].cpu().numpy())[1])
        tipY1 = float((RwT @ fke[tip_link[finger]][col + 1, :3, 3].cpu().numpy())[1])
        side = tipY0 - mid_tip_Y          # which side of middle this finger sits
        dY = tipY1 - tipY0                # lateral motion from +excitation
        if abs(side) < 1e-6 or abs(dY) < 1e-9:
            return 1.0
        return 1.0 if (dY > 0) == (side > 0) else -1.0  # +exc moves away -> +

    # Per finger collect: ABD joints, and FLEX joints with chain depth (proximal->distal).
    grouped: dict[str, dict[str, list]] = {}
    for col, jn in enumerate(cj):
        if jn in manual:  # manual (finger, role) override or exclusion
            ent = manual[jn]
            if ent is None:
                continue
            finger, role = ent
            g = grouped.setdefault(finger, {"abd": [], "flex": [], "twist": []})
            sign = _flex_sign(loader, jn)
            (g["abd"] if role == "mcp_abd" else g.setdefault("manual", [])).append(
                (col, sign) if role == "mcp_abd" else (role, col, sign))
            continue

        finger = jt_finger.get(jn)
        if finger is None or jn not in jaxes:
            continue
        child, axis_local = jaxes[jn]
        axis = RwT @ (fk0[child][0, :3, :3].cpu().numpy() @ axis_local)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        # type = primitive the joint axis is most parallel to (S1.2 dominant component)
        proj = [abs(axis @ np.array([0., 1., 0.])), abs(axis @ np.array([0., 0., 1.])),
                abs(axis @ bone(finger))]
        typ = ("FLEX", "ABD", "ROT")[int(np.argmax(proj))]
        g = grouped.setdefault(finger, {"abd": [], "flex": [], "twist": []})
        if typ == "ROT":
            # axial twist -> only the pinky has a UDHM slot (pinky_twist); fill it
            # if present (robot's own morphology, semantic insertion), else drop.
            if f"{finger}_twist" in UDHM22_SLOTS:
                g["twist"].append((col, _flex_sign(loader, jn)))
            continue
        if typ == "ABD":
            g["abd"].append((col, abd_away_sign(col, finger)))  # away from middle = +
        else:
            depth = float(np.linalg.norm(RwT @ pos(child).cpu().numpy() - wrist_local))
            g["flex"].append((depth, col, _flex_sign(loader, jn)))

    # Assemble: ABD -> mcp_abd; FLEX proximal->distal -> mcp_flex/pip/dip; manual roles as given.
    table: dict[str, dict[str, tuple[int, float]]] = {}
    for finger, g in grouped.items():
        roles: dict[str, tuple[int, float]] = {}
        if g["abd"]:
            roles["mcp_abd"] = g["abd"][0]
        if g.get("twist"):
            roles["twist"] = g["twist"][0]
        for role, col, sign in g.get("manual", []):
            roles[role] = (col, sign)
        free = [r for r in _FLEX_ROLES if r not in roles]
        for (_, col, sign), role in zip(sorted(g["flex"]), free):
            roles[role] = (col, sign)
        if roles:
            table[finger] = roles
    return table


# --- Function 2: robot qpos -> UDHM, by semantic insertion (UDHM / Fang et al. 3.1) ---

_SLOT_IDX = {name: i for i, name in enumerate(UDHM22_SLOTS)}
# role -> UDHM slot suffix for the 4 non-thumb fingers
_ROLE_SUFFIX = {"mcp_abd": "mcp_abd", "mcp_flex": "mcp_flex", "pip": "pip_flex",
                "dip": "dip_flex", "twist": "twist"}
# thumb has its own anatomical slots (cmc/mcp/ip), no pip/dip
_THUMB_ROLE_SLOT = {"mcp_flex": "thumb_mcp_flex", "mcp_abd": "thumb_mcp_abd",
                    "pip": "thumb_ip_flex", "dip": "thumb_ip_flex"}


def _slot_index(finger: str, role: str) -> int:
    name = _THUMB_ROLE_SLOT[role] if finger == "thumb" else f"{finger}_{_ROLE_SUFFIX[role]}"
    return _SLOT_IDX[name]


def robot_to_udhm(qpos: torch.Tensor, table: dict[str, dict[str, tuple[int, float]]]) -> torch.Tensor:
    """Function 2 -- robot qpos [B, J] -> UDHM vector [B, 22].

    Semantic insertion (Fang et al. UDHM, Sec. 3.1): each joint's angle is placed,
    with sign and normalized by pi, into its UDHM coordinate; missing coordinates
    stay zero. The robot HAS its joint angles, so this is direct -- no rotations,
    no angle inference (that is the human path, udhm_run_stage3). Signed (UDHM is a
    signed-DoF interface): flexion sign makes flexion positive; abduction keeps its
    natural +/- (adduction vs abduction).
    """
    out = qpos.new_zeros(qpos.shape[0], len(UDHM22_SLOTS))
    for finger, roles in table.items():
        for role, (col, sign) in roles.items():
            out[:, _slot_index(finger, role)] = float(sign) * qpos[:, col] / math.pi
    return out
