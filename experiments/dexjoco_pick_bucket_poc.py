#!/usr/bin/env python3
"""Small DexJoCo proof-of-concept for external Allegro qpos control.

This intentionally does not depend on ROS, OpenPI, or DexJoCo's own teleop
stack.  It just exercises the benchmark contract we care about:

  external q_allegro_16 -> env.step([tcp_pos3, tcp_quat4, q_allegro_16])
  -> info["succeed"]
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

import numpy as np


DEFAULT_CLOSE_POSE = np.asarray(
    [
        -0.05,
        0.90,
        1.00,
        0.75,
        -0.05,
        0.90,
        1.00,
        0.75,
        -0.05,
        0.90,
        1.00,
        0.75,
        0.90,
        0.55,
        0.85,
        0.85,
    ],
    dtype=np.float64,
)


class UdpHandSource:
    """Reads the latest q_allegro_16 packet, if one is available."""

    def __init__(self, port: int | None) -> None:
        self.sock: socket.socket | None = None
        self.latest: np.ndarray | None = None
        if port is not None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("127.0.0.1", port))
            self.sock.setblocking(False)

    def poll(self) -> np.ndarray | None:
        if self.sock is None:
            return self.latest

        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return self.latest

            q = self._parse_packet(data)
            if q is not None:
                self.latest = q

    @staticmethod
    def _parse_packet(data: bytes) -> np.ndarray | None:
        for dtype in (np.float64, np.float32):
            arr = np.frombuffer(data, dtype=dtype)
            if arr.size >= 16:
                return np.asarray(arr[:16], dtype=np.float64)
        return None

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--mode",
        choices=("home", "close", "cycle", "random-valid"),
        default="cycle",
        help="Fallback hand source when --udp-port is not receiving packets.",
    )
    parser.add_argument("--udp-port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer.")
    parser.add_argument("--save-video", type=Path, default=None)
    return parser.parse_args()


def hand_pose_for_step(
    mode: str,
    step: int,
    home: np.ndarray,
    close_pose: np.ndarray,
    rng: np.random.Generator,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
) -> np.ndarray:
    if mode == "home":
        q = home
    elif mode == "close":
        q = close_pose
    elif mode == "random-valid":
        q = rng.uniform(ctrl_low, ctrl_high)
    else:
        phase = (step % 120) / 119.0
        alpha = 1.0 - abs(2.0 * phase - 1.0)
        q = (1.0 - alpha) * home + alpha * close_pose
    return np.clip(q, ctrl_low, ctrl_high)


def get_raw_env(env):
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw


def maybe_open_video_writer(path: Path | None):
    if path is None:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=30)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.render else "egl")

    from dexjoco.tasks.pick_bucket.config import TaskConfig

    rng = np.random.default_rng(args.seed)
    udp = UdpHandSource(args.udp_port)
    video = maybe_open_video_writer(args.save_video)

    cfg = TaskConfig()
    env = cfg.get_environment(
        policy_mode=True,
        render_mode="human" if args.render else "rgb_array",
        image_obs=args.save_video is not None,
        randomize=False,
        seed=args.seed,
    )
    raw = get_raw_env(env)
    ctrl_ids = np.asarray(raw._allegro_ctrl_ids, dtype=int)
    ctrl_range = raw.model.actuator_ctrlrange[ctrl_ids]
    ctrl_low = ctrl_range[:, 0].astype(np.float64)
    ctrl_high = ctrl_range[:, 1].astype(np.float64)
    close_pose = np.clip(DEFAULT_CLOSE_POSE, ctrl_low, ctrl_high)

    successes = 0
    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            home = np.asarray(obs["state"][7:23], dtype=np.float64)
            episode_success = False
            t0 = time.time()

            for step in range(args.steps):
                tcp_pose = np.asarray(obs["state"][:7], dtype=np.float64)
                q_udp = udp.poll()
                q_hand = (
                    np.clip(q_udp, ctrl_low, ctrl_high)
                    if q_udp is not None
                    else hand_pose_for_step(
                        args.mode, step, home, close_pose, rng, ctrl_low, ctrl_high
                    )
                )
                action = np.concatenate([tcp_pose, q_hand]).astype(np.float64)
                obs, reward, done, trunc, info = env.step(action)

                if video is not None and "front" in obs:
                    video.append_data(obs["front"])

                if info.get("succeed", False):
                    episode_success = True
                if done or trunc:
                    break

            successes += int(episode_success)
            elapsed = time.time() - t0
            print(
                f"episode={episode} success={episode_success} "
                f"steps={step + 1} elapsed={elapsed:.2f}s info={info}"
            )

    finally:
        udp.close()
        if video is not None:
            video.close()
        env.close()

    rate = successes / max(args.episodes, 1)
    print(f"success_rate={successes}/{args.episodes} ({100.0 * rate:.1f}%)")


if __name__ == "__main__":
    main()
