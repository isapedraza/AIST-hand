"""
Measure live source cadence for temporal-CAM planning.

This is a diagnostic tool for the real teleop setup. It measures how often a
source loop runs, how often it returns a valid Dong quaternion pose, and how
often that pose is actually new rather than a repeated async backend result.

Usage:
    python apps/graphgrasp-live/measure_source_cadence.py --source mediapipe --camera 0 --seconds 45
    python apps/graphgrasp-live/measure_source_cadence.py --source wilor --url https://... --camera 0 --seconds 45
    python apps/graphgrasp-live/measure_source_cadence.py --source hamer --url https://... --camera 0 --seconds 45
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import _repo_path  # noqa: F401 -- adds repo packages to sys.path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure valid/new sample cadence from a live retargeting source."
    )
    parser.add_argument("--source", choices=["mediapipe", "hamer", "wilor"], required=True)
    parser.add_argument("--url", default=None, help="Remote backend URL for --source hamer/wilor.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--calib", type=float, default=3.0, help="Calibration seconds.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. .json writes summary JSON; .csv writes event rows.",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be > 0")
    if args.calib < 0:
        parser.error("--calib must be >= 0")
    if args.source in {"hamer", "wilor"} and not args.url:
        parser.error(f"--source {args.source} requires --url")
    if args.source == "mediapipe" and args.url:
        parser.error("--url is only valid with --source hamer/wilor")
    return args


def _make_source(args: argparse.Namespace):
    if args.source == "hamer":
        from sources import HaMeRSource

        return HaMeRSource(url=args.url, camera=args.camera, calib_seconds=args.calib)
    if args.source == "wilor":
        from sources import WiLoRSource

        return WiLoRSource(url=args.url, camera=args.camera, calib_seconds=args.calib)

    from sources import MediaPipeSource

    return MediaPipeSource(camera=args.camera, calib_seconds=args.calib)


def _pose_hash(pose) -> int:
    arr = pose.detach().cpu().contiguous().numpy()
    return hash(arr.tobytes())


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    idx = (len(vals) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _stats_ms(dts: list[float]) -> dict[str, float | None]:
    if not dts:
        return {"mean": None, "median": None, "p95": None, "max": None}
    ms = [v * 1000.0 for v in dts]
    return {
        "mean": statistics.mean(ms),
        "median": statistics.median(ms),
        "p95": _percentile(ms, 0.95),
        "max": max(ms),
    }


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _summarize(
    args: argparse.Namespace,
    events: list[dict[str, Any]],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    duration = max(ended_at - started_at, 1e-9)
    loop_ticks = len(events)
    valid_events = [e for e in events if e["valid"]]
    new_events = [e for e in events if e["new"]]
    duplicate_events = [e for e in events if e["duplicate"]]
    none_count = loop_ticks - len(valid_events)

    valid_times = [e["t_rel"] for e in valid_events]
    new_times = [e["t_rel"] for e in new_events]
    dt_valid = [b - a for a, b in zip(valid_times, valid_times[1:])]
    dt_new = [b - a for a, b in zip(new_times, new_times[1:])]

    return {
        "source": args.source,
        "camera": args.camera,
        "url": args.url,
        "requested_seconds": args.seconds,
        "calib_seconds": args.calib,
        "measured_seconds": duration,
        "loop_ticks": loop_ticks,
        "valid_poses": len(valid_events),
        "new_samples": len(new_events),
        "none_count": none_count,
        "duplicates": len(duplicate_events),
        "loop_hz": loop_ticks / duration,
        "valid_pose_hz": len(valid_events) / duration,
        "new_sample_hz": len(new_events) / duration,
        "none_rate": none_count / loop_ticks if loop_ticks else 0.0,
        "duplicate_rate": (
            len(duplicate_events) / len(valid_events) if valid_events else 0.0
        ),
        "dt_valid_ms": _stats_ms(dt_valid),
        "dt_new_ms": _stats_ms(dt_new),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 56)
    print(f"Source          : {summary['source']}")
    print(f"Measured seconds: {summary['measured_seconds']:.2f}")
    print(f"Loop ticks      : {summary['loop_ticks']}")
    print(f"Valid poses     : {summary['valid_poses']}")
    print(f"New samples     : {summary['new_samples']}")
    print(f"Duplicates      : {summary['duplicates']}")
    print(f"None count      : {summary['none_count']}")
    print("")
    print(f"loop_hz         : {summary['loop_hz']:.2f}")
    print(f"valid_pose_hz   : {summary['valid_pose_hz']:.2f}")
    print(f"new_sample_hz   : {summary['new_sample_hz']:.2f}")
    print(f"none_rate       : {summary['none_rate'] * 100:.1f}%")
    print(f"duplicate_rate  : {summary['duplicate_rate'] * 100:.1f}%")

    for key, label in (("dt_valid_ms", "dt_valid_ms"), ("dt_new_ms", "dt_new_ms")):
        stats = summary[key]
        print("")
        print(label)
        print(f"  mean / median : {_fmt(stats['mean'])} / {_fmt(stats['median'])}")
        print(f"  p95 / max     : {_fmt(stats['p95'])} / {_fmt(stats['max'])}")
    print("=" * 56)


def _write_output(path: Path, summary: dict[str, Any], events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps({"summary": summary, "events": events}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tick", "t_rel", "valid", "new", "duplicate"],
            )
            writer.writeheader()
            writer.writerows(events)
        return
    raise ValueError("Output path must end in .json or .csv")


def main() -> None:
    args = _parse_args()
    source = _make_source(args)

    print(f"Source     : {args.source}")
    print(f"Camera     : {args.camera}")
    if args.url:
        print(f"URL        : {args.url}")
    print(f"Calibration: {args.calib:.1f}s")
    print(f"Measure    : {args.seconds:.1f}s after first valid pose")
    print("Keep the hand visible. Press Q/ESC in the source window to stop early.")

    events: list[dict[str, Any]] = []
    last_hash: int | None = None
    started_at: float | None = None
    tick = 0

    try:
        while source.is_running():
            now = time.perf_counter()
            pose = source.next_frame()

            if started_at is None:
                if pose is None:
                    continue
                started_at = now
                print("[measure] first valid pose received; measuring...")

            t_rel = now - started_at
            if t_rel >= args.seconds:
                break

            valid = pose is not None
            is_new = False
            duplicate = False
            if valid:
                h = _pose_hash(pose)
                is_new = h != last_hash
                duplicate = not is_new
                last_hash = h

            events.append(
                {
                    "tick": tick,
                    "t_rel": t_rel,
                    "valid": valid,
                    "new": is_new,
                    "duplicate": duplicate,
                }
            )
            tick += 1
    finally:
        ended_at = time.perf_counter()
        source.release()

    if started_at is None:
        raise RuntimeError("No valid pose was received before the source stopped.")

    summary = _summarize(args, events, started_at, ended_at)
    _print_summary(summary)

    if args.output:
        out = Path(args.output)
        _write_output(out, summary, events)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
