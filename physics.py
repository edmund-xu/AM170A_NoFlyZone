"""Physics + energy model for stop-at-waypoint segments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from params import DroneParams


@dataclass(frozen=True)
class SegmentResult:
    """Convenient bundle of optimal segment data."""
    distance: float
    t_opt: float
    e_opt: float


class DronePhysics:
    """
    Segment model:
      v(t) = alpha * t * (T - t) (scalar speed along the segment direction)
      alpha chosen so that total distance traveled equals d.

    With linear drag (scalar along direction):
      F(t) = m a(t) + C v(t)
      Power(t) = hover_power + |F(t) * v(t)|
      Energy = ∫ Power(t) dt
    """

    def __init__(self, params: DroneParams) -> None:
        self.p = params

    # Core kinematics (1D along the segment)
    @staticmethod
    def _alpha_for_distance(d: float, T: float) -> float:
        """
        For v(t)=alpha t(T-t):
        distance = ∫0^T v(t) dt = alpha * T^3 / 6
        => alpha = 6d / T^3
        """
        return 6.0 * d / (T**3)

    @staticmethod
    def _v_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        return alpha * t * (T - t)

    @staticmethod
    def _a_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        # derivative of alpha*t(T-t) = alpha*(T - 2t)
        return alpha * (T - 2.0 * t)

    @staticmethod
    def _s_profile(alpha: float, T: float, t: np.ndarray) -> np.ndarray:
        # s(t) = ∫ v dt = alpha * (T t^2 / 2 - t^3 / 3)
        return alpha * (T * (t**2) / 2.0 - (t**3) / 3.0)

    # Energy
    def segment_energy(self, d: float, T: float) -> float:
        """
        Compute energy for a segment of length d executed in time T.
        Returns +inf if T is non-positive or d is negative.
        """
        if d < 0 or T <= 0:
            return float("inf")
        if d == 0:
            # If no movement, just hover for T seconds
            return self.p.hover_power * T

        alpha = self._alpha_for_distance(d, T)

        n = max(20, int(self.p.integration_steps))
        t = np.linspace(0.0, T, n)

        v = self._v_profile(alpha, T, t) # speed
        a = self._a_profile(alpha, T, t) # acceleration
        F = self.p.mass * a + self.p.drag_coeff * v  # required thrust (1D)

        power_thrust = np.abs(F * v)
        power_total = self.p.hover_power + power_thrust

        return float(np.trapz(power_total, t))

    # Bounds + optimization over T 
    def feasible_time_bounds(self, d: float) -> tuple[float, float]:
        """
        Compute a conservative [T_low, T_high] for searching T.

        Peak speed occurs at t=T/2:
          v_max_profile = alpha*(T/2)*(T/2) = alpha*T^2/4
                        = (6d/T^3)*T^2/4 = 1.5 d / T
          => enforce v_max_profile <= v_max  => T >= 1.5 d / v_max

        Peak acceleration magnitude occurs at endpoints (t=0 or t=T):
          a_max_profile = |alpha*T| = (6d/T^3)*T = 6d/T^2
          => enforce a_max_profile <= a_max => T >= sqrt(6d/a_max)
        """
        if d <= 0:
            return (1e-3, 1.0)

        t_v = 1.5 * d / max(self.p.v_max, 1e-9)
        t_a = math.sqrt(6.0 * d / max(self.p.a_max, 1e-9))
        t_low = max(1e-3, t_v, t_a)

        # generous upper bound so optimizer can find the U-shaped minimum
        t_high = max(t_low * 4.0, self.p.t_upper_per_meter * d)
        return (t_low, t_high)

    def find_optimal_time(self, d: float) -> SegmentResult:
        """Minimize segment_energy(d, T) over feasible T bounds."""
        if d < 0:
            return SegmentResult(distance=d, t_opt=float("nan"), e_opt=float("inf"))

        T_low, T_high = self.feasible_time_bounds(d)

        def obj(T: float) -> float:
            return self.segment_energy(d, T)

        res = minimize_scalar(obj, bounds=(T_low, T_high), method="bounded")
        return SegmentResult(distance=d, t_opt=float(res.x), e_opt=float(res.fun))

    # Build segment trajectory for plotting
    def segment_trajectory(
        self,
        A: np.ndarray,
        B: np.ndarray,
        T: float,
        steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Return a dict with arrays: t, pos (Nx2), vel (Nx2), acc (Nx2).
        This uses the same parabolic speed profile, so vel(0)=vel(T)=0.
        """
        A = np.asarray(A, dtype=float).reshape(2)
        B = np.asarray(B, dtype=float).reshape(2)
        dvec = B - A
        d = float(np.linalg.norm(dvec))
        if d == 0:
            n = steps or 50
            t = np.linspace(0.0, T, n)
            pos = np.repeat(A[None, :], n, axis=0)
            vel = np.zeros_like(pos)
            acc = np.zeros_like(pos)
            return {"t": t, "pos": pos, "vel": vel, "acc": acc}

        u = dvec / d  # direction unit vector

        alpha = self._alpha_for_distance(d, T)
        n = steps or max(50, int(self.p.integration_steps // 4))
        t = np.linspace(0.0, T, n)
        s = self._s_profile(alpha, T, t)
        v = self._v_profile(alpha, T, t)
        a = self._a_profile(alpha, T, t)

        pos = A[None, :] + s[:, None] * u[None, :]
        vel = v[:, None] * u[None, :]
        acc = a[:, None] * u[None, :]

        return {"t": t, "pos": pos, "vel": vel, "acc": acc}