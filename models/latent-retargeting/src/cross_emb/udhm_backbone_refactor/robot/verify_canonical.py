"""Canonical-pose oracle for robot UDHM adapters (sign + slot routing check).

This is a DESIGNED EXCITATION, lifted from joint-space (paper S1.2 per-joint nudge)
into UDHM slot-space: instead of nudging one joint and projecting its tip onto a
palm frame (which is meaningless for the thumb -- the CMC axes are non-orthogonal,
non-intersecting and off the palm plane), we drive a whole semantically-known UDHM
pose THROUGH the adapter and read the rendered SHAPE.

    "fist" in UDHM  --[adapter]-->  robot qpos  -->  MuJoCo render  -->  eye
     intent (convention)            translator       physical          judge

The adapter is the device under test: the pose is robot-agnostic (lives in UDHM),
the adapter carries all the per-robot routing + sign. If the adapter is right the
render matches the convention; a wrong sign or a cross-family slot shows up as a
specific visible defect (a straight finger inside the fist, a finger crossing
inward instead of fanning, a thumb that points out instead of opposing).

What it catches: every SIGN error, and SLOT errors that cross a primitive family
(flex<->abd<->rot) or a finger -- i.e. anything that changes the rendered shape.
What it does NOT catch on its own: a swap between joints excited identically by a
pose (index pip<->dip in `fist`); add the `hook` pose to break that symmetry.

Run foreground (GUI -- your eyes are authoritative):
  python -m cross_emb.udhm_backbone_refactor.robot.verify_canonical \\
      --robot shadow --strip rh_ \\
      --adapter .../adapters/shadow_udhm_adapter.yaml --pose fist

Offscreen PNG (coarse first pass; image vision is fallible, confirm in GUI):
  ... --pose all --png /tmp/shadow_{pose}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

from cross_emb.udhm_backbone_refactor.core.udhm import UDHM22_SLOTS

# Reuse the proven menagerie scene loader (renders cleanly; raw URDF does not).
_REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(_REPO / "models/grasp-intent-classification/scripts"))
from mujoco_valid_poses_viewer import ROBOTS, build_scene  # noqa: E402

_FRAC = 0.9  # drive to 90% of the joint range so the primitive reads clearly


def canonical_poses() -> dict[str, dict[str, float]]:
    """UDHM canonical poses, derived from slot names so they never desync from
    UDHM22_SLOTS. Each value is a target in normalized UDHM units (+1 = full
    UDHM-positive). Each pose isolates a primitive family so a defect is local.
    """
    flex = {s: 1.0 for s in UDHM22_SLOTS if s.endswith("_flex")}
    # middle is the reference axis ("away from middle" is undefined for itself), so
    # keep it FIXED as the visual anchor -- otherwise the fan has no still reference.
    abd = {s: 1.0 for s in UDHM22_SLOTS if s.endswith("_abd") and s != "middle_mcp_abd"}
    pip = {s: 1.0 for s in UDHM22_SLOTS if s.endswith("_pip_flex")}
    return {
        "zeros": {},                                   # baseline: flat, adducted
        "fist": flex,                                  # all FLEX signs
        "spread": abd,                                 # ABD signs (middle anchored at 0)
        "thumb_opp": {"thumb_cmc_flex": 1.0,           # coupled thumb base (the
                      "thumb_cmc_spread": 1.0,         # motion palm-frame can't read)
                      "thumb_mcp_abd": 1.0},
        "pinky_twist": {"pinky_twist": 1.0},           # lone ROT slot
        "middle_abd": {"middle_mcp_abd": 1.0},         # middle alone (free conv: +1 = ulnar)
        "hook": pip,                                   # pip only -> breaks pip/dip swap
    }


def _actuated_joints(model) -> dict[str, tuple[int, float, float]]:
    """{joint_name: (qpos_adr, lo, hi)} for hinge/slide joints."""
    out: dict[str, tuple[int, float, float]] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-1.0, 1.0)
        out[name] = (adr, float(lo), float(hi))
    return out


def udhm_to_qpos(model, adapter: dict, pose: dict[str, float], strip: str = "") -> np.ndarray:
    """Push a UDHM pose through the adapter into a full MuJoCo qpos vector.

    For each mapped joint: drive = sign * pose[slot]  (robot_to_udhm is
    angle = sign * q, so the inverse sends UDHM-positive to q-direction sign).
    drive>0 -> positive limit, drive<0 -> negative limit, 0 -> rest (0 if in range).
    """
    jmap = _actuated_joints(model)

    def mj_name(key: str) -> str | None:
        if key in jmap:
            return key
        if strip and (strip + key) in jmap:
            return strip + key
        return None

    qpos = np.zeros(model.nq)
    missing: list[str] = []
    for joint, entry in adapter["joints"].items():
        name = mj_name(joint)
        if name is None:
            missing.append(joint)
            continue
        adr, lo, hi = jmap[name]
        drive = entry["sign"] * pose.get(entry["slot"], 0.0)
        if drive > 0:
            q = (hi if hi > 0 else 0.0) * _FRAC
        elif drive < 0:
            q = (lo if lo < 0 else 0.0) * _FRAC
        else:
            q = 0.0 if lo <= 0.0 <= hi else 0.5 * (lo + hi)
        qpos[adr] = q
    if missing:
        raise ValueError(f"adapter joints not found in model (strip={strip!r}): {missing}")
    return qpos


def _render_png(model, qpos: np.ndarray, out: Path, width: int = 900, height: int = 900) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data)
    img = renderer.render()
    import imageio.v3 as iio
    iio.imwrite(out, img)


def _view_gui(model, qpos: np.ndarray) -> None:
    import mujoco.viewer
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    print("  Cierra la ventana para continuar.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


def _view_gui_dynamic(model, qpos: np.ndarray) -> None:
    """Sawtooth-ramp the whole pose 0 -> target so the DIRECTION of travel is
    visible (the static endpoint hides it for abd/rot). Slow ramp = the motion,
    instant snap back to 0 = clearly not real, like the annotate checker.
    """
    import time
    import mujoco.viewer
    data = mujoco.MjData(model)
    RAMP, HOLD = 1.8, 0.4
    print("  Rampa lenta 0->pose = movimiento real; snap a 0 = reset. Cierra para continuar.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            phase = (time.time() - t0) % (RAMP + 2 * HOLD)
            if phase < HOLD:
                frac = 0.0
            elif phase < HOLD + RAMP:
                frac = (phase - HOLD) / RAMP
            else:
                frac = 1.0
            data.qpos[:] = qpos * frac
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.012)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--strip", default="", help="prefix to strip (shadow: rh_)")
    ap.add_argument("--pose", default="all",
                    help="pose name or 'all' (zeros|fist|spread|thumb_opp|pinky_twist|hook)")
    ap.add_argument("--png", default=None,
                    help="offscreen PNG path; use {pose} placeholder. Omit for GUI.")
    ap.add_argument("--dynamic", action="store_true",
                    help="GUI: ramp the pose 0->target so direction of travel is visible (abd/rot)")
    args = ap.parse_args()

    adapter = yaml.safe_load(args.adapter.read_text())
    model = mujoco.MjModel.from_xml_path(str(build_scene(ROBOTS[args.robot])))
    poses = canonical_poses()
    names = list(poses) if args.pose == "all" else [args.pose]

    for name in names:
        qpos = udhm_to_qpos(model, adapter, poses[name], strip=args.strip)
        print(f"[{name}] qpos nonzero: {int((qpos != 0).sum())}/{model.nq}")
        if args.png:
            out = Path(str(args.png).format(pose=name))
            _render_png(model, qpos, out)
            print(f"  PNG -> {out}")
        elif args.dynamic:
            _view_gui_dynamic(model, qpos)
        else:
            _view_gui(model, qpos)


if __name__ == "__main__":
    main()
