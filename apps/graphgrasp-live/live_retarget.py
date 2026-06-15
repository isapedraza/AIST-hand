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
from sinks import MuJocoSink

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

    # One retargeter + one MuJoCo window per robot in the checkpoint (multi-robot
    # ckpts share E_h/D_X; each robot has its own D_r). One perception stream
    # drives them all so the hands mirror the same human pose side by side.
    import torch
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    robot_names = list((ck.get("robots") or {}).keys()) or [None]
    del ck

    pairs = []
    for name in robot_names:
        try:
            ret  = Retargeter(ckpt_path, robot_name=name)
            sink = MuJocoSink(robot=ret.robot_name)
            pairs.append((ret, sink))
            print(f"  + {ret.robot_name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"  - skip {name}: {e}")
    if not pairs:
        parser.error("no supported robots in checkpoint")
    rot_repr = pairs[0][0].human_rot_repr

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

    print(f"Robots       : {[r.robot_name for r, _ in pairs]}")
    print(f"Rotation repr: {rot_repr}")
    print("Running. Q/ESC to quit.")
    while source.is_running() and all(s.is_running() for _, s in pairs):
        quats = source.next_frame()
        if quats is None:
            continue
        pose = quat_wxyz_to_rot6d(quats) if rot_repr == "r6" else quats
        for ret, sink in pairs:
            sink.update(ret(pose))

    source.release()
    for _, sink in pairs:
        sink.release()


if __name__ == "__main__":
    main()
