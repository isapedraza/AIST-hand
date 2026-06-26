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
from pathlib import Path

from cross_emb.inference import Retargeter
from cross_emb.rotations import quat_wxyz_to_rot6d
from sinks import MuJocoSink, MergedMuJocoSink

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="Path to Stage 1 checkpoint (.pt).")
    parser.add_argument("--camera", type=int,   default=0)
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
    args = parser.parse_args()

    if args.source in ("hamer", "wilor") and not args.url:
        parser.error(f"--source {args.source} requires --url")

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

    if len(rets) == 1:
        sink = MuJocoSink(robot=rets[0].robot_name)
        def render(pose):
            q = rets[0](pose)
            sink.update(q)
            if emit is not None:
                emit.update(q)
    else:
        sink = MergedMuJocoSink([r.robot_name for r in rets])
        def render(pose):           sink.update({r.robot_name: r(pose) for r in rets})

    if args.source == "hamer":
        from sources import HaMeRSource
        source = HaMeRSource(url=args.url, camera=args.camera, calib_seconds=args.calib)
    elif args.source == "wilor":
        from sources import WiLoRSource
        source = WiLoRSource(url=args.url, camera=args.camera, calib_seconds=args.calib)
    else:
        from sources import MediaPipeSource
        source = MediaPipeSource(camera=args.camera, calib_seconds=args.calib)

    if args.interpolate:
        from sources import InterpolatedSource
        source = InterpolatedSource(source)

    print(f"Robots       : {[r.robot_name for r in rets]}")
    print(f"Rotation repr: {rot_repr}")
    print("Running. Q/ESC to quit.")
    while source.is_running() and sink.is_running():
        quats = source.next_frame()
        if quats is None:
            continue
        pose = quat_wxyz_to_rot6d(quats) if rot_repr == "r6" else quats
        render(pose)

    source.release()
    sink.release()
    if emit is not None:
        emit.release()


if __name__ == "__main__":
    main()
