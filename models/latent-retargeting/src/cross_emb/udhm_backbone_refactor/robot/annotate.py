"""Manual UDHM adapter annotator (visual, MuJoCo).

The human IS the reference frame: instead of computing a palm frame and reading
the sign geometrically, we animate each actuated joint, the person WATCHES the
motion, and reports what they see. The code turns observation + convention into
the slot and the sign, and writes the adapter YAML.

Why this exists: the automatic excitation (primitives.compute_signs) needs a
global palm frame, which is meaningless for the thumb (out of the palm plane) and
for non-anthropomorphic hands (Barrett, 3 radial fingers). Watching sidesteps it.

Per actuated joint, as its value INCREASES, the person answers:
  - type:   (f)lex / (a)bd / (r)ot / (s)kip       (one DoF = one type)
  - finger: thumb/index/middle/ring/pinky
  - node:   mcp/pip/dip                            (FLEX only; abd->mcp, rot->twist)
  - dir:    flex -> toward palm? | abd -> away from middle? | rot -> sense
Convention (UDHM): FLEX+ toward palm, ABD+ away from middle. The code maps the
reported motion-on-increasing-value to +1/-1.

Uses the proven menagerie scene loader (mujoco_valid_poses_viewer) so the hand
renders correctly, instead of a raw URDF (which renders poorly). Menagerie joint
names carry a prefix (shadow: "rh_"); --strip maps them back to the adapter keys.

Run foreground (GUI in background segfaults):
  python -m cross_emb.udhm_backbone_refactor.robot.annotate \\
      --robot shadow --strip rh_ \\
      --out  .../adapters/shadow_udhm_adapter.yaml \\
      --only THJ5 THJ4 THJ3 THJ2 THJ1      # optional: annotate a subset
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Reuse the proven menagerie scene loader (renders cleanly; raw URDF does not).
_REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(_REPO / "models/grasp-intent-classification/scripts"))
from mujoco_valid_poses_viewer import ROBOTS, build_scene  # noqa: E402

# finger + node + type -> UDHM slot. Thumb has its own anatomy (cmc/mcp/ip).
_THUMB_SLOT = {("flex", "mcp"): "thumb_cmc_flex", ("flex", "pip"): "thumb_mcp_flex",
               ("flex", "dip"): "thumb_ip_flex",  ("abd", "mcp"): "thumb_mcp_abd",
               ("rot", "mcp"): "thumb_cmc_spread"}
_NODE_SUFFIX = {("flex", "mcp"): "mcp_flex", ("flex", "pip"): "pip_flex",
                ("flex", "dip"): "dip_flex", ("abd", "mcp"): "mcp_abd",
                ("rot", "mcp"): "twist"}


def _slot(finger: str, prim: str, node: str) -> str:
    if finger == "thumb":
        return _THUMB_SLOT[(prim, node)]
    return f"{finger}_{_NODE_SUFFIX[(prim, node)]}"


def _sign(prim: str, direction: str) -> int:
    """Direction reported as the joint value INCREASES -> sign vs UDHM convention.

    FLEX+ = toward palm; ABD+ = away from middle. If increasing the value already
    moves it the UDHM-positive way, sign = +1, else -1.
    """
    positive = {"flex": "palm", "abd": "away", "rot": "pos"}[prim]
    return 1 if direction == positive else -1


def _actuated_joints(model) -> list[tuple[str, int, float, float]]:
    """[(name, qpos_adr, lo, hi)] for hinge/slide joints, in model order."""
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-1.0, 1.0)
        out.append((name, int(adr), float(lo), float(hi)))
    return out


def _oscillator(model, data, viewer, state):
    """Background thread: sawtooth sweep of the current joint so 'increasing' is
    unambiguous. Slow ramp min->max (this is the INCREASING stroke), hold at max,
    then INSTANT snap back to min (clearly not a real motion). state: {adr,lo,hi,run}.

    RAMP/HOLD in seconds; the instant reset is what disambiguates direction.
    """
    base = data.qpos.copy()
    RAMP, HOLD = 1.8, 0.4
    t0 = time.time()
    last_adr = None
    while state["run"]:
        adr = state["adr"]
        if adr is None:
            time.sleep(0.02)
            continue
        if adr != last_adr:                         # new joint -> restart from min
            t0 = time.time()
            last_adr = adr
        phase = (time.time() - t0) % (RAMP + 2 * HOLD)
        if phase < HOLD:
            frac = 0.0                              # rest at min
        elif phase < HOLD + RAMP:
            frac = (phase - HOLD) / RAMP            # slow ramp up = increasing
        else:
            frac = 1.0                              # rest at max, then snaps to min
        data.qpos[:] = base
        data.qpos[adr] = state["lo"] + (state["hi"] - state["lo"]) * frac
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.012)


def _ask_sign_only(joint: str, slot: str) -> int | None:
    """Slot already known (e.g. from spec). Ask ONLY the observable direction.
    Returns sign +1/-1, or None to skip."""
    is_open = ("abd" in slot) or ("spread" in slot)
    if is_open:
        q = "  al AUMENTAR, (a)leja del medio/palma / a(c)erca / (s)kip / (q)uit: "
        pos = "a"
    else:  # flex
        q = "  al AUMENTAR, (h)acia palma / (l)ejos / (s)kip / (q)uit: "
        pos = "h"
    ans = input(f"\n[{joint}] -> {slot}\n{q}").strip().lower()
    if ans == "s" or ans == "":
        return None
    if ans == "q":
        raise KeyboardInterrupt
    return 1 if ans == pos else -1


def _ask(joint: str) -> dict | None:
    """Prompt the person. Returns adapter entry dict, or None to skip."""
    t = input(f"\n[{joint}] (la mano va LENTO de min a max = AUMENTAR, luego regresa de golpe)\n"
              f"  tipo (f)lex/(a)bd/(r)ot/(s)kip/(q)uit: ").strip().lower()
    if t in ("s", ""):
        return None
    if t == "q":
        raise KeyboardInterrupt
    prim = {"f": "flex", "a": "abd", "r": "rot"}[t]
    finger = input("  dedo thumb/index/middle/ring/pinky: ").strip().lower()
    if prim == "flex":
        node = input("  nodo mcp/pip/dip: ").strip().lower()
        direction = "palm" if input("  al AUMENTAR, va (h)acia palma / (l)ejos: ").strip().lower() == "h" else "far"
    elif prim == "abd":
        node = "mcp"
        direction = "away" if input("  al AUMENTAR, (s)e aleja del medio / a(c)erca: ").strip().lower() == "s" else "near"
    else:  # rot
        node = "mcp"
        direction = "pos" if input("  al AUMENTAR, sentido (p)ositivo / (n)egativo: ").strip().lower() == "p" else "neg"
    return {"slot": _slot(finger, prim, node), "sign": _sign(prim, direction)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), required=True)
    ap.add_argument("--out",   type=Path, required=True, help="adapter YAML to write/update")
    ap.add_argument("--strip", default="", help="prefix to strip from joint names for YAML keys (shadow: rh_)")
    ap.add_argument("--only",  nargs="*", default=None, help="annotate only these joints (post-strip names)")
    ap.add_argument("--signs-only", action="store_true",
                    help="slot already in adapter; ask ONLY the direction (sign). Use for thumbs.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(build_scene(ROBOTS[args.robot])))
    data = mujoco.MjData(model)

    def key(name: str) -> str:
        return name[len(args.strip):] if args.strip and name.startswith(args.strip) else name

    # signs-only needs the existing adapter (for the known slots).
    import yaml
    existing = yaml.safe_load(args.out.read_text()) if args.out.exists() else {"joints": {}}
    existing.setdefault("joints", {})

    joints = _actuated_joints(model)
    if args.only:
        want = set(args.only)
        joints = [j for j in joints if key(j[0]) in want]
    if args.signs_only:   # only joints already in the adapter (slot known)
        joints = [j for j in joints if key(j[0]) in existing["joints"]]

    entries: dict[str, dict] = {}
    print(f"Anotando {len(joints)} joints de {args.robot}. La mano oscila el joint actual; clasifica en la terminal.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        state = {"adr": None, "lo": 0.0, "hi": 0.0, "run": True}
        worker = threading.Thread(target=_oscillator, args=(model, data, viewer, state), daemon=True)
        worker.start()
        try:
            for name, adr, lo, hi in joints:
                state.update(adr=adr, lo=lo, hi=hi)   # oscillator picks up the new joint
                k = key(name)
                if args.signs_only:
                    slot = existing["joints"][k]["slot"]
                    sign = _ask_sign_only(k, slot)
                    if sign is not None:
                        entries[k] = {"slot": slot, "sign": sign}
                        print(f"    -> {slot}  sign={sign:+d}")
                else:
                    entry = _ask(k)
                    if entry is not None:
                        entries[k] = entry
                        print(f"    -> {entry['slot']}  sign={entry['sign']:+d}")
        except KeyboardInterrupt:
            print("\nInterrumpido; guardo lo anotado.")
        finally:
            state["run"] = False

    # Write/merge into the adapter YAML (joints block only).
    import yaml
    doc = yaml.safe_load(args.out.read_text()) if args.out.exists() else {"robot": args.robot}
    doc.setdefault("joints", {})
    doc["joints"].update(entries)
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    print(f"\nEscrito {len(entries)} joints en {args.out}")


if __name__ == "__main__":
    main()
