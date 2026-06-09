"""
Live retargeting: perception -> Dong kinematics -> Shadow Hand (MuJoCo).

Perception backend selectable: MediaPipe world landmarks (default) or HaMeR
remote inference (--source hamer --url ...). HaMeR feeds the same Dong path and
gives cleaner frontal MCP flexion (closes the fist where MediaPipe does not).

Usage:
    python apps/graphgrasp-live/live_retarget.py --ckpt /path/to/stage1_best_total.pt
    python apps/graphgrasp-live/live_retarget.py --ckpt ... --camera 1 --calib 5.0
    python apps/graphgrasp-live/live_retarget.py --ckpt ... --source hamer --url https://xxxx.trycloudflare.com
"""

import _repo_path  # noqa: F401 -- adds latent-retargeting/src to sys.path

import argparse
from pathlib import Path

from cross_emb.inference import Retargeter
from sinks import MuJocoSink

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="Path to Stage 1 checkpoint (.pt).")
    parser.add_argument("--camera", type=int,   default=0)
    parser.add_argument("--calib",  type=float, default=3.0, help="Calibration seconds")
    parser.add_argument("--source", choices=["mediapipe", "hamer"], default="mediapipe",
                        help="Perception backend feeding Dong kinematics")
    parser.add_argument("--url", default=None,
                        help="HaMeR remote inference URL (required for --source hamer)")
    args = parser.parse_args()

    if args.source == "hamer" and not args.url:
        parser.error("--source hamer requires --url")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    print(f"Checkpoint : {ckpt_path}")
    print(f"Camera     : {args.camera}")
    print(f"Calibration: {args.calib}s")
    print(f"Source     : {args.source}")

    retargeter = Retargeter(ckpt_path)
    if args.source == "hamer":
        from sources import HaMeRSource
        source = HaMeRSource(url=args.url, camera=args.camera, calib_seconds=args.calib)
    else:
        from sources import MediaPipeSource
        source = MediaPipeSource(camera=args.camera, calib_seconds=args.calib)
    sink       = MuJocoSink()

    print("Running. Q/ESC to quit.")
    while source.is_running() and sink.is_running():
        quats = source.next_frame()
        if quats is None:
            continue
        qpos = retargeter(quats)
        sink.update(qpos)

    source.release()
    sink.release()


if __name__ == "__main__":
    main()
