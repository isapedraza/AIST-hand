"""
UDP qpos sink: stream retargeter qpos to a remote sim teleop receiver.

Sends qpos as raw float64 (Menagerie order), the wire format the dexjoco-shadow
teleop_driver FingerReceiver expects on UDP 5014 (right hand). Pairs the live
retargeter (this repo) with the functional sim (the fork) over localhost/LAN.

Same interface as MuJocoSink (update/is_running/release) so it drops into the
live_retarget loop alongside or instead of the local viewer.
"""
from __future__ import annotations

import socket

import numpy as np


class UdpQposSink:
    """Forward each qpos frame to host:port as float64 bytes (fire-and-forget)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5014):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def update(self, qpos: np.ndarray) -> None:
        self._sock.sendto(
            np.asarray(qpos).reshape(-1).astype(np.float64).tobytes(), self._addr
        )

    def is_running(self) -> bool:
        return True

    def release(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class UdpPoseSink:
    """Forward a wrist pose (3x4 [R|t]) to host:port as 12 float64 bytes, the
    sim_teleop VIVE wire format the fork's WristReceiver expects on UDP 5012."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5012):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def update(self, pose: np.ndarray) -> None:
        p = np.asarray(pose, dtype=np.float64).reshape(3, 4)
        self._sock.sendto(p.tobytes(), self._addr)

    def is_running(self) -> bool:
        return True

    def release(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass
