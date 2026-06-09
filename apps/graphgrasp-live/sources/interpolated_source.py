"""
InterpolatedSource — wraps any live source and SLERP-interpolates between frames.

Smooths the freeze→jump artifact that occurs when the inner source has high
latency (e.g. HaMeR/WiLoR over a Cloudflare tunnel at ~250 ms RTT).

Between inner frames the wrapper extrapolates α forward based on elapsed time,
clamped to [0, 1] so it never goes past the last known frame.

Usage:
    source = InterpolatedSource(HaMeRSource(url=..., camera=0))
    source = InterpolatedSource(WiLoRSource(url=..., camera=0))

If you later switch to 6D rotations [1, 20, 6], replace _interpolate() with
linear lerp + Gram-Schmidt re-orthogonalization; the wrapper logic is unchanged.
"""

from __future__ import annotations

import time

import torch


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """SLERP between two quaternion tensors of shape [..., 4]."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    # Use shortest path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    # Fall back to lerp when quaternions are nearly identical (avoid div/0)
    linear = q0 + t * (q1 - q0)
    theta = dot.acos()
    sin_theta = theta.sin()

    safe = sin_theta.abs() > 1e-6
    slerp = (((1.0 - t) * theta).sin() / sin_theta) * q0 + \
            ((t * theta).sin() / sin_theta) * q1

    out = torch.where(safe, slerp, linear)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _interpolate(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Interpolate between two quat frames [1, 20, 4]."""
    return _slerp(a, b, t)


class InterpolatedSource:
    """
    Drop-in wrapper for HaMeRSource / WiLoRSource.

    Calls inner.next_frame() each tick. When a new frame arrives, stores it
    as the new target. Between frames, returns a SLERP at fraction α based on
    elapsed time relative to the last observed inter-frame interval.
    """

    def __init__(self, inner):
        self._inner = inner
        self._prev: torch.Tensor | None = None
        self._curr: torch.Tensor | None = None
        self._t_prev: float = 0.0
        self._t_curr: float = 0.0

    # Delegate lifecycle to inner source
    def is_running(self) -> bool:
        return self._inner.is_running()

    def release(self) -> None:
        self._inner.release()

    def next_frame(self) -> torch.Tensor | None:
        frame = self._inner.next_frame()

        if frame is not None:
            now = time.perf_counter()
            self._prev = self._curr
            self._t_prev = self._t_curr
            self._curr = frame
            self._t_curr = now

        # No frame yet at all
        if self._curr is None:
            return None

        # Only one frame received — return it as-is
        if self._prev is None:
            return self._curr

        # Interpolate based on elapsed time vs last observed period
        period = self._t_curr - self._t_prev
        if period < 1e-6:
            return self._curr

        elapsed = time.perf_counter() - self._t_curr
        alpha = min(elapsed / period, 1.0)

        return _interpolate(self._prev, self._curr, alpha)
