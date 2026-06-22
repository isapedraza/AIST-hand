"""Automatic sign calibration via joint excitation -- DexGrasp-Zero S1.B (Wu 2026).

Scope: this module computes ONLY the alignment sign s_j of each joint. The node
and primitive type come from the adapter YAML slot name -- that is the manual
functional partitioning the paper also relies on. We add the one thing a human
cannot read off by inspection: the direction that aligns the joint's physical
motion with the UDHM convention, so robot UDHM and human UDHM close the same way.

Method (kinematic -- a single revolute nudge is pure FK, no physics engine):
  1. build the Dong palm frame from wrist + index/middle/ring MCP reference links
  2. excite joint +eps from a base pose, run FK
  3. read the induced fingertip motion in the palm frame
  4. set the sign by the slot's primitive against the UDHM convention:
       FLEX+ = tip moves toward the palm  (+n_p; the Dong palm normal is volar)
       ABD+  = tip spreads away from the middle-finger axis (palm lateral Y)
       ROT+  = positive axial rotation of the bone

Only valid for anthropomorphic hands that have the 3 MCP reference links the Dong
palm frame needs. Non-anthropomorphic hands (e.g. Barrett, 3 radial fingers) have
no meaningful global palm frame -- annotate those manually.
"""
from __future__ import annotations

import numpy as np
import torch

from cross_emb.loaders.dong_math import _dong_block1_wrist_frame


def _primitive_of(slot: str) -> str:
    if "abd" in slot or "spread" in slot:
        return "ABD"
    if "twist" in slot:
        return "ROT"
    return "FLEX"


def compute_signs(loader, adapter: dict, base_pose: dict[str, float] | None = None,
                  eps: float = 0.05) -> dict[str, int]:
    """Return {joint_name: +1/-1} for every joint in the adapter.

    adapter: parsed adapter YAML with `frame`, `fingers`, `joints`.
             frame   = {wrist, index_mcp, middle_mcp, ring_mcp}  (link names)
             fingers = {finger: {base, tip}}                     (link names)
             joints  = {joint_name: {slot, ...}}
    base_pose: {joint: angle} held during excitation (default all zeros).
    """
    cj: list[str] = loader.chain_joint_names
    frame, fingers = adapter["frame"], adapter["fingers"]

    q0 = torch.zeros(len(cj), device=loader.device)
    for jn, val in (base_pose or {}).items():
        if jn in cj:
            q0[cj.index(jn)] = float(val)

    fk0 = loader.run_fk(q0[None])
    Rw = _dong_block1_wrist_frame(
        fk0[frame["wrist"]][:, :3, 3],      fk0[frame["index_mcp"]][:, :3, 3],
        fk0[frame["middle_mcp"]][:, :3, 3], fk0[frame["ring_mcp"]][:, :3, 3],
    )[0]
    RwT = Rw.t().cpu().numpy()                       # world -> palm
    n_p = np.array([0.0, 0.0, 1.0])                  # palm normal (palm-frame Z, volar)

    def tip_palm(finger, fk):
        return RwT @ fk[fingers[finger]["tip"]][0, :3, 3].cpu().numpy()

    def link_rot_palm(link, fk):
        return RwT @ fk[link][0, :3, :3].cpu().numpy()

    mid_tip0 = tip_palm("middle", fk0)

    signs: dict[str, int] = {}
    for jn, entry in adapter["joints"].items():
        if jn not in cj:
            continue
        finger = entry["slot"].split("_")[0]
        prim = _primitive_of(entry["slot"])

        qn = q0.clone(); qn[cj.index(jn)] += eps
        fkn = loader.run_fk(qn[None])
        t0, t1 = tip_palm(finger, fk0), tip_palm(finger, fkn)
        d = t1 - t0

        if prim == "FLEX":                           # toward palm = +n_p
            s = 1.0 if (d @ n_p) > 0 else -1.0
        elif prim == "ABD":                          # away from middle (lateral Y)
            side = t0[1] - mid_tip0[1]
            s = 1.0 if (abs(side) < 1e-6 or d[1] * side >= 0) else -1.0
        else:                                        # ROT: axial spin of the bone
            base = RwT @ fk0[fingers[finger]["base"]][0, :3, 3].cpu().numpy()
            bone = tip_palm(finger, fk0) - base
            bone /= np.linalg.norm(bone) + 1e-9
            tip_link = fingers[finger]["tip"]
            dR = link_rot_palm(tip_link, fkn) @ link_rot_palm(tip_link, fk0).T
            omega = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            s = 1.0 if (omega @ bone) >= 0 else -1.0

        signs[jn] = int(s)
    return signs


def verify_flex_against_limits(loader, adapter: dict, signs: dict[str, int]) -> dict[str, bool]:
    """Cross-check: a FLEX joint with a one-sided positive range flexes on +qpos,
    so its sign should be +1 (and -1 for a one-sided negative range). Returns
    {joint: ok} only for FLEX joints whose limits are clearly one-sided.
    """
    out: dict[str, bool] = {}
    for jn, entry in adapter["joints"].items():
        if _primitive_of(entry["slot"]) != "FLEX":
            continue
        spec = loader.joint_specs.get(jn)
        if spec is None or spec.lower is None or spec.upper is None:
            continue
        lo, hi = spec.lower, spec.upper
        if lo >= -1e-3 and hi > 1e-3:                # one-sided positive -> expect +1
            out[jn] = signs.get(jn) == 1
        elif hi <= 1e-3 and lo < -1e-3:              # one-sided negative -> expect -1
            out[jn] = signs.get(jn) == -1
    return out
