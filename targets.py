"""Mission targets for generating waypoints and computing distances."""
from typing import Optional

import numpy as np
from numpy.random import default_rng

from obstacles import RectangleZone


class Targets:
    """
    Generates and manages waypoint coordinates for drone mission planning.
    """

    def __init__(
        self,
        num_targets: int,
        bounds: tuple[float, float],
        *,
        waypoint_set: Optional[list[tuple[float, float]]] = None,
        distribution: str = "uniform",
        seed: Optional[int] = None,
    ) -> None:
        self.num_targets = num_targets
        self.bounds = bounds
        self.distribution = distribution
        self._waypoint_set = waypoint_set
        self._rng = default_rng(seed)

    @staticmethod
    def _point_in_any_obstacle(
        point: np.ndarray,
        obstacles: list[RectangleZone] | None,
    ) -> bool:
        if not obstacles:
            return False
        x, y = float(point[0]), float(point[1])
        return any(obs.contains_point(x, y) for obs in obstacles)

    def generate_waypoints(
        self,
        obstacles: list[RectangleZone] | None = None,
        max_attempts: int = 10000,
    ) -> np.ndarray:
        """
        Return waypoints based on the selected distribution, avoiding no-fly zones.
        """
        if self._waypoint_set is not None:
            waypoints = np.array(self._waypoint_set, dtype=np.float64)
            for idx, pt in enumerate(waypoints):
                if self._point_in_any_obstacle(pt, obstacles):
                    raise ValueError(
                        f"Fixed waypoint {idx} at ({pt[0]:.1f}, {pt[1]:.1f}) lies inside a no-fly zone."
                    )
            return waypoints

        low, high = self.bounds

        # -------- default uniform sampling with rejection --------
        if self.distribution == "uniform":
            waypoints = []
            attempts = 0
            while len(waypoints) < self.num_targets:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Could not generate enough valid waypoints outside no-fly zones. "
                        "Try fewer/lower-size obstacles or larger bounds."
                    )
                pt = self._rng.uniform(low=low, high=high, size=2)
                if not self._point_in_any_obstacle(pt, obstacles):
                    waypoints.append(pt)
                attempts += 1
            return np.array(waypoints, dtype=np.float64)

        # -------- clustered sampling with rejection --------
        elif self.distribution == "clustered":
            num_clusters = 3
            cluster_centers = self._rng.uniform(low=low, high=high, size=(num_clusters, 2))
            waypoints = []
            attempts = 0
            i = 0
            while len(waypoints) < self.num_targets:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Could not generate enough clustered waypoints outside no-fly zones."
                    )
                center = cluster_centers[i % num_clusters]
                pt = center + self._rng.normal(scale=(high - low) / 15, size=2)
                pt = np.clip(pt, low, high)
                if not self._point_in_any_obstacle(pt, obstacles):
                    waypoints.append(pt)
                    i += 1
                attempts += 1
            return np.array(waypoints, dtype=np.float64)

        # -------- grid sampling with filtering --------
        elif self.distribution == "grid":
            grid_size = int(np.ceil(np.sqrt(self.num_targets)))
            x = np.linspace(low + (high - low) / 10, high - (high - low) / 10, grid_size)
            y = np.linspace(low + (high - low) / 10, high - (high - low) / 10, grid_size)
            xv, yv = np.meshgrid(x, y)
            grid_points = np.vstack([xv.ravel(), yv.ravel()]).T

            valid_points = [
                pt for pt in grid_points
                if not self._point_in_any_obstacle(pt, obstacles)
            ]

            if len(valid_points) < self.num_targets:
                raise RuntimeError(
                    "Not enough valid grid points outside no-fly zones."
                )

            valid_points = np.array(valid_points, dtype=np.float64)
            indices = self._rng.choice(len(valid_points), self.num_targets, replace=False)
            jitter = self._rng.normal(scale=(high - low) / 100, size=(self.num_targets, 2))
            candidates = valid_points[indices] + jitter
            candidates = np.clip(candidates, low, high)

            # fix any jittered points that slipped into obstacles
            fixed = []
            attempts = 0
            for pt in candidates:
                cur = pt.copy()
                while self._point_in_any_obstacle(cur, obstacles):
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            "Could not repair jittered grid waypoint outside no-fly zones."
                        )
                    cur = self._rng.uniform(low=low, high=high, size=2)
                    attempts += 1
                fixed.append(cur)

            return np.array(fixed, dtype=np.float64)

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def get_distance_matrix(self, waypoints: np.ndarray) -> np.ndarray:
        n = waypoints.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(
                        (waypoints[i, 0] - waypoints[j, 0]) ** 2
                        + (waypoints[i, 1] - waypoints[j, 1]) ** 2
                    )
        return dist_matrix