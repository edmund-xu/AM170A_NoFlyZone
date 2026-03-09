from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from obstacles import RectangleZone


GridNode = tuple[int, int]


class AStarPlanner:
    """
    Grid-based A* planner for obstacle-aware pairwise waypoint routing.
    """

    def __init__(
        self,
        bounds: tuple[float, float],
        obstacles: list[RectangleZone] | None = None,
        grid_step: float = 25.0,
        allow_diagonal: bool = True,
    ) -> None:
        self.low, self.high = bounds
        self.obstacles = obstacles or []
        self.grid_step = float(grid_step)
        self.allow_diagonal = allow_diagonal

        self.nx = int(round((self.high - self.low) / self.grid_step)) + 1
        self.ny = int(round((self.high - self.low) / self.grid_step)) + 1

    # ---------- coordinate transforms ----------
    def point_to_node(self, point: np.ndarray | tuple[float, float]) -> GridNode:
        x, y = float(point[0]), float(point[1])
        ix = int(round((x - self.low) / self.grid_step))
        iy = int(round((y - self.low) / self.grid_step))
        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        return (ix, iy)

    def node_to_point(self, node: GridNode) -> np.ndarray:
        ix, iy = node
        x = self.low + ix * self.grid_step
        y = self.low + iy * self.grid_step
        return np.array([x, y], dtype=float)

    # ---------- obstacle logic ----------
    def is_blocked_node(self, node: GridNode) -> bool:
        x, y = self.node_to_point(node)
        return any(obs.contains_point(x, y) for obs in self.obstacles)

    def validate_waypoints(self, waypoints):
        for idx, (x, y) in enumerate(waypoints):
            for obs in self.obstacles:
                if obs.contains_point(float(x), float(y)):
                    raise ValueError(
                        f"Waypoint {idx} at ({x:.1f}, {y:.1f}) lies inside a no-fly zone."
                    )

    # ---------- A* helpers ----------
    def neighbors(self, node: GridNode) -> list[tuple[GridNode, float]]:
        ix, iy = node
        steps_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        steps_diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        steps = steps_4 + steps_diag if self.allow_diagonal else steps_4

        out: list[tuple[GridNode, float]] = []
        for dx, dy in steps:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < self.nx and 0 <= jy < self.ny):
                continue

            nxt = (jx, jy)
            if self.is_blocked_node(nxt):
                continue

            step_cost = self.grid_step * math.sqrt(2.0) if dx != 0 and dy != 0 else self.grid_step
            out.append((nxt, step_cost))
        return out

    @staticmethod
    def heuristic(a: GridNode, b: GridNode) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    def reconstruct_path(
        self,
        came_from: dict[GridNode, GridNode],
        current: GridNode,
    ) -> list[GridNode]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def astar(self, start_pt: np.ndarray, goal_pt: np.ndarray) -> list[np.ndarray]:
        start = self.point_to_node(start_pt)
        goal = self.point_to_node(goal_pt)

        if self.is_blocked_node(start):
            raise ValueError(f"Start node {start} is inside a no-fly zone.")
        if self.is_blocked_node(goal):
            raise ValueError(f"Goal node {goal} is inside a no-fly zone.")

        open_heap: list[tuple[float, GridNode]] = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: dict[GridNode, GridNode] = {}
        g_score: dict[GridNode, float] = {start: 0.0}
        f_score: dict[GridNode, float] = {start: self.heuristic(start, goal)}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                node_path = self.reconstruct_path(came_from, current)
                point_path = [self.node_to_point(node) for node in node_path]
                point_path[0] = np.asarray(start_pt, dtype=float)
                point_path[-1] = np.asarray(goal_pt, dtype=float)
                return point_path

            for neighbor, step_cost in self.neighbors(current):
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_val = tentative_g + self.heuristic(neighbor, goal) * self.grid_step
                    f_score[neighbor] = f_val
                    heapq.heappush(open_heap, (f_val, neighbor))

        raise RuntimeError("A* failed to find a feasible path between two waypoints.")

    @staticmethod
    def polyline_length(path: list[np.ndarray]) -> float:
        total = 0.0
        for k in range(len(path) - 1):
            total += float(np.linalg.norm(path[k + 1] - path[k]))
        return total

    def build_obstacle_aware_distance_matrix(
        self,
        waypoints: np.ndarray,
    ) -> tuple[np.ndarray, dict[tuple[int, int], list[np.ndarray]]]:
        """
        Returns:
            distance_matrix[i,j] = obstacle-aware A* path length
            path_lookup[(i,j)] = list of points along the A* polyline
        """
        self.validate_waypoints(waypoints)

        n = waypoints.shape[0]
        dist_matrix = np.zeros((n, n), dtype=float)
        path_lookup: dict[tuple[int, int], list[np.ndarray]] = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i, j] = 0.0
                    path_lookup[(i, j)] = [waypoints[i].copy()]
                else:
                    path = self.astar(waypoints[i], waypoints[j])
                    dist = self.polyline_length(path)
                    dist_matrix[i, j] = dist
                    path_lookup[(i, j)] = path

        return dist_matrix, path_lookup