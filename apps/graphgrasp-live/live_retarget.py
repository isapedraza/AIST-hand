"""
Live retargeting: perception -> Dong kinematics -> Shadow Hand (MuJoCo).

Perception backend selectable: MediaPipe world landmarks (default), HaMeR, or
WiLoR remote inference (--source {hamer,wilor} --url ...). HaMeR and WiLoR feed
the same Dong path and give cleaner frontal MCP flexion (close the fist where
MediaPipe does not). WiLoR is much faster on GPU (lighter ViT) than HaMeR's ViT-H.

Usage:
    python apps/graphgrasp-live/live_retarget.py --ckpt /path/to/stage1_best_total.pt
    python apps/graphgrasp-live/live_retarget.py --ckpt ... --camera 1 --calib 5.0
    python apps/graphgrasp-live/live_retarget.py --ckpt ... --source hamer --url https://xxxx.trycloudflare.com
    python apps/graphgrasp-live/live_retarget.py --ckpt ... --source wilor --url https://xxxx.trycloudflare.com
    python apps/graphgrasp-live/live_retarget.py --ckpt ... --source hamer --url https://... --interpolate
"""

import _repo_path  # noqa: F401 -- adds latent-retargeting/src to sys.path

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from cross_emb.inference import Retargeter
from cross_emb.rotations import quat_wxyz_to_rot6d
from sinks import MuJocoSink, MergedMuJocoSink

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = Path.home() / "Downloads" / "fig41-captures"


def _camera_arg(v: str):
    """A bare integer is a local webcam index; anything else is a stream URL."""
    return int(v) if v.isdigit() else v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="Path to Stage 1 checkpoint (.pt).")
    parser.add_argument("--camera", type=_camera_arg, default=0,
                        help="Webcam index (int) OR a stream URL (phone-as-webcam: "
                             "an MJPEG app over USB tethering, http://PHONE_IP:8080/video)")
    parser.add_argument("--calib",  type=float, default=3.0, help="Calibration seconds")
    parser.add_argument("--source", choices=["mediapipe", "hamer", "wilor"], default="mediapipe",
                        help="Perception backend feeding Dong kinematics")
    parser.add_argument("--url", default=None,
                        help="Remote inference URL (required for --source hamer/wilor)")
    parser.add_argument("--interpolate", action="store_true",
                        help="Wrap source with SLERP interpolation (smooths freeze→jump for high-latency backends)")
    parser.add_argument("--robot", default=None,
                        help="Single robot to show in its own window (e.g. shadow). "
                             "Omit to show ALL robots in the ckpt merged side by side in one window.")
    parser.add_argument("--emit-udp", action="store_true",
                        help="Also stream the single robot's qpos over UDP to a sim "
                             "teleop receiver (dexjoco-shadow driver, port 5014). "
                             "Requires a single robot (use --robot shadow).")
    parser.add_argument("--emit-host", default="127.0.0.1", help="UDP target host for --emit-udp")
    parser.add_argument("--emit-port", type=int, default=5014, help="UDP target port for --emit-udp")
    parser.add_argument("--emit-wrist", action="store_true",
                        help="Also stream the WiLoR wrist pose (3x4) over UDP to the sim "
                             "teleop wrist receiver (port 5012). Requires --source wilor.")
    parser.add_argument("--wrist-port", type=int, default=5012, help="UDP target port for --emit-wrist")
    parser.add_argument("--session", type=Path,
                        help="Record raw WiLoR camera frames and timestamps into SESSION/operator.")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Skip the extra robot preview; requires --emit-udp.")
    args = parser.parse_args()
    if args.session and args.source != "wilor":
        parser.error("--session currently supports --source wilor")
    if args.session and not args.robot:
        parser.error("--session requires an explicit --robot")
    if args.no_viewer and not args.emit_udp:
        parser.error("--no-viewer requires --emit-udp")
    if args.session and (args.session / "operator").exists():
        parser.error("This session already has operator data; choose a new session.")

    if args.source in ("hamer", "wilor") and not args.url:
        parser.error(f"--source {args.source} requires --url")

    if args.emit_wrist and args.source != "wilor":
        parser.error("--emit-wrist requires --source wilor (wrist pose comes from WiLoRSource)")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    print(f"Checkpoint : {ckpt_path}")
    print(f"Camera     : {args.camera}")
    print(f"Calibration: {args.calib}s")
    print(f"Source     : {args.source}")
    print(f"Interpolate: {args.interpolate}")

    # Multi-robot ckpts share E_h/D_X; each robot has its own D_r. One retargeter
    # per robot; one perception stream drives them all. With --robot, show that
    # single hand in its own window (one passive viewer). Otherwise merge ALL
    # ckpt robots into ONE window side by side (multiple passive viewers in one
    # process segfault, so the many-hand case uses a single merged model).
    import torch
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_robots = list((ck.get("robots") or {}).keys())
    del ck
    names = [args.robot] if args.robot else (ckpt_robots or [None])

    rets = []
    for name in names:
        try:
            rets.append(Retargeter(ckpt_path, robot_name=name))
            print(f"  + {rets[-1].robot_name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"  - skip {name}: {e}")
    if not rets:
        parser.error("no supported robots in checkpoint")
    rot_repr = rets[0].human_rot_repr

    # Optional UDP emitter: stream the single robot's qpos to the sim teleop
    # receiver (the fork's teleop_driver). Decoupled — off unless --emit-udp.
    emit = None
    if args.emit_udp:
        if len(rets) != 1:
            parser.error("--emit-udp requires a single robot (use --robot shadow)")
        from sinks import UdpQposSink
        emit = UdpQposSink(host=args.emit_host, port=args.emit_port)
        print(f"Emitting qpos -> udp {args.emit_host}:{args.emit_port}")

    # Optional wrist-pose emitter (stage 2): independent channel to UDP 5012.
    wrist_emit = None
    if args.emit_wrist:
        from sinks import UdpPoseSink
        wrist_emit = UdpPoseSink(host=args.emit_host, port=args.wrist_port)
        print(f"Emitting wrist 3x4 -> udp {args.emit_host}:{args.wrist_port}")

    # UDHM tooling for capture-time RS (angular fidelity), Shadow only for now.
    # Reuses the same validated pipeline as
    # scripts/evaluate_udhm_cross_embodiment.py: robot_to_udhm on raw qpos,
    # human_to_udhm on Dong quats + DONG_LABELS (already positionally aligned
    # with DongKinematics' 1..20 joint_order -- verified 2026-08-26).
    udhm_tools = None
    if len(rets) == 1 and rets[0].robot_name == "shadow":
        try:
            from cross_emb.loaders.robot_loader import RobotLoader
            from cross_emb.loaders.robot_primitives import build_primitives, robot_to_udhm
            from cross_emb.loaders.human_to_udhm import human_to_udhm
            from cross_emb.loaders.human_loader import DONG_LABELS
            from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS
            _loader = RobotLoader(str(REPO_ROOT / "robot/hands/shadow_hand/shadow_hand_right.urdf"))
            _tabla = build_primitives(_loader, REPO_ROOT / "robot/hand-configs/shadow.yaml")
            udhm_tools = (robot_to_udhm, human_to_udhm, DONG_LABELS, UDHM22_SLOTS, _tabla)
        except Exception as e:
            print(f"[udhm] no disponible: {e}")

    if len(rets) == 1:
        sink = None if args.no_viewer else MuJocoSink(robot=rets[0].robot_name)
        def render(pose):
            q = rets[0](pose)
            if sink is not None:
                sink.update(q)
            if emit is not None:
                emit.update(q)
    else:
        sink = MergedMuJocoSink([r.robot_name for r in rets])
        def render(pose):           sink.update({r.robot_name: r(pose) for r in rets})

    source = None
    try:
        wrist_src = None
        if args.source == "hamer":
            from sources import HaMeRSource
            source = HaMeRSource(url=args.url, camera=args.camera, calib_seconds=args.calib)
        elif args.source == "wilor":
            from sources import WiLoRSource
            recording_metadata = None
            if args.session:
                with ckpt_path.open("rb") as checkpoint:
                    checkpoint_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
                recording_metadata = {
                    "robot": rets[0].robot_name, "source": args.source,
                    "checkpoint": str(ckpt_path.resolve()), "checkpoint_sha256": checkpoint_hash,
                    "calibration_seconds": args.calib, "interpolate": args.interpolate,
                    "source_root": str(REPO_ROOT),
                    "camera_device": args.camera if isinstance(args.camera, int) else "stream",
                    "emit_udp": args.emit_udp, "emit_port": args.emit_port,
                    "emit_wrist": args.emit_wrist, "wrist_port": args.wrist_port,
                    "timestamp_semantics": "Host time immediately after camera.read; not sensor exposure time",
                }
            source = WiLoRSource(url=args.url, camera=args.camera, calib_seconds=args.calib,
                                 record_directory=args.session / "operator" if args.session else None,
                                 record_metadata=recording_metadata)
            wrist_src = source   # keep handle for wrist_pose() before any wrapping
        else:
            from sources import MediaPipeSource
            source = MediaPipeSource(camera=args.camera, calib_seconds=args.calib)

        if args.interpolate:
            from sources import InterpolatedSource
            source = InterpolatedSource(source)

        can_capture = sink is not None and hasattr(source, "pop_capture_request") and len(rets) == 1
        if can_capture:
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Capturas     : espacio (con la ventana de la camara en foco) -> {CAPTURE_DIR}")

        print(f"Robots       : {[r.robot_name for r in rets]}")
        print(f"Rotation repr: {rot_repr}")
        print("Running. Q/ESC to quit.")
        while source.is_running() and (sink is None or sink.is_running()):
            quats = source.next_frame()
            if quats is None:
                continue
            pose = quat_wxyz_to_rot6d(quats) if rot_repr == "r6" else quats
            render(pose)
            if wrist_emit is not None:
                wp = wrist_src.wrist_pose()
                if wp is not None:
                    wrist_emit.update(wp)
            if can_capture and source.pop_capture_request():
                ts = time.strftime("%Y%m%d-%H%M%S")
                frame = source.current_frame_with_skeleton()
                if frame is not None:
                    import cv2
                    cv2.imwrite(str(CAPTURE_DIR / f"{ts}_camara.png"), frame)
                sink.capture(CAPTURE_DIR / f"{ts}_robot.png")
                print(f"[captura] {ts}")

                # Fidelidad input-vs-salida a la Santos et al. 2025 (Sec. IV-C):
                # yema estimada por WiLoR (input) vs yema del robot vía FK sobre
                # el qpos que en verdad se mando (output). Sin ground truth
                # externo -- mismo principio que su heat map de error euclidiano.
                #
                # Ambos lados se rotan a su PROPIO marco local de muñeca (Dong
                # Block 1, Eq. 5-7) antes de comparar -- misma funcion exacta que
                # ya usa el pipeline validado de Tabla 4.2 (dong_run_stage2 /
                # DongKinematics), no una comparacion de posiciones crudas en
                # marcos distintos (ese fue el bug de la primera version).
                points_w = getattr(source, "last_points_w", lambda: None)()
                tip_pos = sink.tip_positions() if hasattr(sink, "tip_positions") else None
                mcp_pts = sink.mcp_frame_points() if hasattr(sink, "mcp_frame_points") else None
                if points_w is not None and tip_pos is not None and mcp_pts is not None:
                    from cross_emb.loaders.dong_math import _dong_block1_wrist_frame, _dong_world_to_local
                    import torch as _torch

                    # Datos crudos, no solo el numero final: si el calculo vuelve
                    # a tener bug, se recalcula desde aqui sin recapturar en vivo.
                    q_robot_raw = rets[0](pose)
                    np.savez(
                        CAPTURE_DIR / f"{ts}_datos.npz",
                        points_w=points_w,  # [21,3] keypoints WiLoR (input crudo)
                        quats_human=quats.numpy() if hasattr(quats, "numpy") else np.asarray(quats),  # [1,20,4] Dong quats humanos
                        q_robot=np.asarray(q_robot_raw),  # qpos real enviado al robot
                        tip_thumb=tip_pos["thumb"], tip_index=tip_pos["index"],
                        tip_middle=tip_pos["middle"], tip_ring=tip_pos["ring"], tip_little=tip_pos["little"],
                        mcp_wrist=mcp_pts["wrist"], mcp_index=mcp_pts["index_mcp"],
                        mcp_middle=mcp_pts["middle_mcp"], mcp_ring=mcp_pts["ring_mcp"],
                        ref_r=sink.reference_length(),
                    )

                    def _t(v):
                        return _torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0)

                    # Lado humano: wrist=0, index_mcp=5, middle_mcp=9, ring_mcp=13 (WiLoR JOINTS).
                    R_h = _dong_block1_wrist_frame(_t(points_w[0]), _t(points_w[5]), _t(points_w[9]), _t(points_w[13]))
                    ref_h = float(np.linalg.norm(points_w[9] - points_w[0]))  # wrist->middle MCP

                    # Lado robot: mismos 3 puntos anatomicos (shadow.yaml frame_*).
                    R_r = _dong_block1_wrist_frame(
                        _t(mcp_pts["wrist"]), _t(mcp_pts["index_mcp"]), _t(mcp_pts["middle_mcp"]), _t(mcp_pts["ring_mcp"])
                    )
                    ref_r = sink.reference_length()

                    if ref_h > 1e-6 and ref_r:
                        # WiLoR JOINTS order: WRIST=0, THUMB_TIP=4, INDEX_TIP=8,
                        # MIDDLE_TIP=12, RING_TIP=16, PINKY_TIP=20.
                        human_tip_idx = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
                        lines = [f"[fidelidad] {ts}  (adimensional, normalizado por longitud de mano, ambos en marco local de muneca)"]
                        for finger, idx in human_tip_idx.items():
                            tip_h_local = _dong_world_to_local(_t(points_w[idx]), _t(points_w[0]), R_h)[0].numpy() / ref_h
                            tip_r_local = _dong_world_to_local(_t(tip_pos[finger]), _t(mcp_pts["wrist"]), R_r)[0].numpy() / ref_r
                            err = float(np.linalg.norm(tip_h_local - tip_r_local))
                            lines.append(f"  {finger:8s} NDS={err:.3f}")
                        report = "\n".join(lines)
                        print(report)

                        # RS-equivalente (fidelidad angular, UDHM): mismo principio
                        # -- WiLoR (input, via Dong quats) vs robot (output, via
                        # qpos que en verdad se mando). robot_to_udhm/human_to_udhm
                        # y DONG_LABELS: codigo ya existente y validado en
                        # scripts/evaluate_udhm_cross_embodiment.py, no inventado
                        # para esta captura.
                        if udhm_tools is not None:
                            robot_to_udhm, human_to_udhm, DONG_LABELS, UDHM22_SLOTS, tabla = udhm_tools
                            import torch as _torch
                            with _torch.no_grad():
                                udhm_r = robot_to_udhm(_torch.from_numpy(q_robot_raw).float(), tabla)
                                udhm_h = human_to_udhm(quats.float() if hasattr(quats, "float") else _torch.from_numpy(quats).float(), DONG_LABELS)
                            diff = (udhm_h - udhm_r).abs()
                            rs_lines = [f"[fidelidad] {ts}  RS (UDHM, angular, adimensional)"]
                            for i, slot in enumerate(UDHM22_SLOTS):
                                rs_lines.append(f"  {slot:16s} diff={diff[0, i].item():.3f}")
                            rs_lines.append(f"  MEAN |diff| = {diff.mean().item():.4f}")
                            rs_report = "\n".join(rs_lines)
                            print(rs_report)
                            lines.append("")
                            lines.append(rs_report)

                        (CAPTURE_DIR / f"{ts}_fidelidad.txt").write_text("\n".join(lines) + "\n")

    except KeyboardInterrupt:
        print("Stopping live capture.")
    finally:
        # Close the recording even on Ctrl+C; continue cleanup if one close fails.
        import sys
        active_error = sys.exc_info()[0] is not None
        cleanup_error = None
        for resource in (source, sink, emit, wrist_emit):
            if resource is not None:
                try:
                    resource.release()
                except Exception as exc:
                    print(f"[cleanup] {exc}")
                    cleanup_error = cleanup_error or exc
        if cleanup_error is not None and not active_error:
            raise cleanup_error


if __name__ == "__main__":
    main()
