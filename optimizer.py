"""Routing optimization using segment stop-at-waypoint energy."""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

from physics import DronePhysics, SegmentResult


class RoutingOptimizer:
    """
    Build energy_matrix[i,j] = E_min(d_ij), then solve a TSP-like tour.

    Methods supported:
      - brute: exact by enumerating permutations (small n only)
      - nearest_neighbor: greedy heuristic
      - nn_2opt: nearest neighbor initialization + 2-opt local search improvement
    """
    def __init__(self, physics_model: DronePhysics) -> None:
        self.physics = physics_model

    def find_optimal_time(self, distance: float) -> SegmentResult:
        return self.physics.find_optimal_time(distance)

    def build_energy_matrix(self, distance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = distance_matrix.shape[0]
        energy_matrix = np.zeros((n, n), dtype=float)
        time_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    energy_matrix[i, j] = 0.0
                    time_matrix[i, j] = 0.0
                else:
                    seg = self.find_optimal_time(float(distance_matrix[i, j]))
                    energy_matrix[i, j] = seg.e_opt
                    time_matrix[i, j] = seg.t_opt

        return energy_matrix, time_matrix
    #Solver
    def solve_tsp(self, cost_matrix: np.ndarray, method: str = "brute") -> list[int]:
        """
        method:
          - "brute"
          - "nearest_neighbor"
          - "nn_2opt"
        """
        n = cost_matrix.shape[0]
        if n <= 1:
            return list(range(n))

        if method == "nearest_neighbor":
            return self._solve_nearest_neighbor(cost_matrix)
        if method == "nn_2opt":
            init = self._solve_nearest_neighbor(cost_matrix)
            improved = self._two_opt(cost_matrix, init)
            return improved

        return self._solve_brute(cost_matrix)
    #Core utilities
    @staticmethod
    def tour_cost(cost_matrix: np.ndarray, order: list[int]) -> float:
        """Cycle cost: order[0] -> ... -> order[-1] -> order[0]."""
        total = 0.0
        for k in range(len(order)):
            i = order[k]
            j = order[(k + 1) % len(order)]
            total += float(cost_matrix[i, j])
        return total

    def _solve_brute(self, cost_matrix: np.ndarray) -> list[int]:
        n = cost_matrix.shape[0]
        best_order: list[int] = []
        best_cost = float("inf")

        for perm in itertools.permutations(range(1, n)):
            order = [0] + list(perm)
            cost = self.tour_cost(cost_matrix, order)
            if cost < best_cost:
                best_cost = cost
                best_order = order

        return best_order

    def _solve_nearest_neighbor(self, cost_matrix: np.ndarray) -> list[int]:
        n = cost_matrix.shape[0]
        order = [0]
        unvisited = set(range(1, n))

        while unvisited:
            cur = order[-1]
            nxt = min(unvisited, key=lambda j: float(cost_matrix[cur, j]))
            order.append(nxt)
            unvisited.remove(nxt)

        return order

    #2-opt improvement
    def _two_opt(self, cost_matrix: np.ndarray, order: list[int]) -> list[int]:
        """
        Standard 2-opt local search for a Hamiltonian cycle.
        Keeps node 0 as the fixed start (order[0] == 0).
        """
        n = len(order)
        if n < 4:
            return order[:]

        best = order[:]
        best_cost = self.tour_cost(cost_matrix, best)

        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for k in range(i + 1, n - 1):
                    new_order = best[:]
                    new_order[i : k + 1] = reversed(new_order[i : k + 1])

                    new_cost = self.tour_cost(cost_matrix, new_order)
                    if new_cost < best_cost:
                        best = new_order
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return best