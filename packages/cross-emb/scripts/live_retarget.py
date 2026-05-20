"""
Live retargeting: MediaPipe -> Dong kinematics -> Shadow Hand (MuJoCo).

Usage:
    python live_retarget.py --ckpt checkpoints/stage1_cam_5000.pt
    python live_retarget.py --ckpt checkpoints/stage1_cam_5000.pt --camera 1 --calib 5.0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_emb.inference import Retargeter
from cross_emb.inference.sources import MediaPipeSource
from cross_emb.inference.sinks import MuJocoSink


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="checkpoints/stage1_cam_5000.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calib",  type=float, default=3.0, help="Calibration seconds")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    print(f"Checkpoint : {ckpt_path}")
    print(f"Camera     : {args.camera}")
    print(f"Calibration: {args.calib}s")

    retargeter = Retargeter(ckpt_path)
    source     = MediaPipeSource(camera=args.camera, calib_seconds=args.calib)
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
