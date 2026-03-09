from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from obstacles import RectangleZone
from optimizer import RoutingOptimizer
from params import (
    SimulationConfig,
    get_default_params,
    get_default_sim_config,
    get_test_sim_config,
)
from pathfinding import AStarPlanner
from physics import DronePhysics
from plotting import Visualizer
from targets import Targets


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def compute_route_stats(
    simulation_config: SimulationConfig,
) -> dict[str, float | list[int] | np.ndarray | dict]:
    config = simulation_config or get_default_sim_config()
    params = get_default_params()

    targets = Targets(
        num_targets=config.num_targets,
        bounds=config.bounds,
        waypoint_set=config.waypoint_set,
        seed=config.seed,
    )
    waypoints = targets.generate_waypoints(
        obstacles=config.obstacles if config.use_obstacles else None
    )
    n = waypoints.shape[0]

    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    path_lookup = None
    if config.use_obstacles and config.obstacles:
        planner = AStarPlanner(
            bounds=config.bounds,
            obstacles=config.obstacles,
            grid_step=config.grid_step,
            allow_diagonal=True,
        )
        distance_matrix, path_lookup = planner.build_obstacle_aware_distance_matrix(
            waypoints
        )
        planning_mode = "Obstacle-aware A* path lengths"
    else:
        distance_matrix = targets.get_distance_matrix(waypoints)
        planning_mode = "Straight-line Euclidean distances"

    energy_matrix, time_matrix = optimizer.build_energy_matrix(distance_matrix)

    naive_order = list(range(n))
    nn_order = optimizer.solve_tsp(energy_matrix, method="nearest_neighbor")
    two_opt_order = optimizer.solve_tsp(energy_matrix, method="nn_2opt")

    brute_order = None
    brute_energy = None
    if n <= 8:
        brute_order = optimizer.solve_tsp(energy_matrix, method="brute")
        brute_energy = optimizer.tour_cost(energy_matrix, brute_order)

    def route_cost(order: list[int], matrix: np.ndarray) -> float:
        return optimizer.tour_cost(matrix, order)

    return {
        "planning_mode": planning_mode,
        "waypoints": waypoints,
        "path_lookup": path_lookup,
        "obstacles": config.obstacles,
        "naive_order": naive_order,
        "nn_order": nn_order,
        "two_opt_order": two_opt_order,
        "brute_order": brute_order,
        "naive_energy": route_cost(naive_order, energy_matrix),
        "nn_energy": route_cost(nn_order, energy_matrix),
        "two_opt_energy": route_cost(two_opt_order, energy_matrix),
        "brute_energy": brute_energy,
        "naive_distance": route_cost(naive_order, distance_matrix),
        "nn_distance": route_cost(nn_order, distance_matrix),
        "two_opt_distance": route_cost(two_opt_order, distance_matrix),
        "naive_time": route_cost(naive_order, time_matrix),
        "nn_time": route_cost(nn_order, time_matrix),
        "two_opt_time": route_cost(two_opt_order, time_matrix),
    }


def benchmark_results(
    N_values: list[int],
    trials_per_N: int = 10,
    bounds: tuple[float, float] = (0.0, 2000.0),
    use_obstacles: bool = False,
    obstacles: list[RectangleZone] | None = None,
    grid_step: float = 25.0,
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_ratio_twoopt_vs_nn = []
    std_ratio_twoopt_vs_nn = []

    mean_ratio_twoopt_vs_brute = []
    std_ratio_twoopt_vs_brute = []
    brute_Ns = []

    for N in N_values:
        ratios_twoopt_vs_nn = []
        ratios_twoopt_vs_brute = []

        for trial in range(trials_per_N):
            cfg = SimulationConfig(
                num_targets=N,
                seed=1000 * N + trial,
                bounds=bounds,
                use_obstacles=use_obstacles,
                obstacles=obstacles or [],
                grid_step=grid_step,
            )

            stats = compute_route_stats(cfg)

            E_nn = float(stats["nn_energy"])
            E_2 = float(stats["two_opt_energy"])
            E_min = stats["brute_energy"]

            ratios_twoopt_vs_nn.append(E_2 / E_nn)

            if E_min is not None:
                ratios_twoopt_vs_brute.append(E_2 / float(E_min))

        mean_ratio_twoopt_vs_nn.append(float(np.mean(ratios_twoopt_vs_nn)))
        std_ratio_twoopt_vs_nn.append(float(np.std(ratios_twoopt_vs_nn)))

        if len(ratios_twoopt_vs_brute) > 0:
            brute_Ns.append(N)
            mean_ratio_twoopt_vs_brute.append(float(np.mean(ratios_twoopt_vs_brute)))
            std_ratio_twoopt_vs_brute.append(float(np.std(ratios_twoopt_vs_brute)))

        print(f"N = {N}")
        print(f"  mean(E2/ENN)   = {np.mean(ratios_twoopt_vs_nn):.4f}")
        print(f"  std(E2/ENN)    = {np.std(ratios_twoopt_vs_nn):.4f}")
        if len(ratios_twoopt_vs_brute) > 0:
            print(f"  mean(E2/Emin)  = {np.mean(ratios_twoopt_vs_brute):.4f}")
            print(f"  std(E2/Emin)   = {np.std(ratios_twoopt_vs_brute):.4f}")
        print()

    visualizer = Visualizer()
    visualizer.plot_benchmark_results(
        N_values_nn=N_values,
        mean_e2_over_enn=mean_ratio_twoopt_vs_nn,
        std_e2_over_enn=std_ratio_twoopt_vs_nn,
        brute_Ns=brute_Ns,
        mean_e2_over_emin=mean_ratio_twoopt_vs_brute,
        std_e2_over_emin=std_ratio_twoopt_vs_brute,
    )

    print("Benchmark plots saved:")
    print(" - plots/benchmark_E2_over_ENN.png")
    if len(brute_Ns) > 0:
        print(" - plots/benchmark_E2_over_Emin.png")


def main(simulation_config: SimulationConfig | None = None) -> None:
    stats = compute_route_stats(simulation_config)
    physics = DronePhysics(get_default_params())

    print("=== Drone Routing Optimization (Stop-at-Waypoint) ===")
    print(f"Planning mode: {stats['planning_mode']}")
    print(f"Waypoints (x,y):\n{stats['waypoints']}\n")

    print("=== Route Indices (cycle closes back to 0) ===")
    print(f"Naive:     {stats['naive_order']}")
    print(f"Optimized: {stats['nn_order']} (nearest neighbor)")
    print(f"Super:     {stats['two_opt_order']} (nearest neighbor + 2-opt)")
    if stats["brute_order"] is not None:
        print(f"Brute:     {stats['brute_order']} (exact optimum)")
    print()

    print("=== Total Distance Comparison ===")
    print(f"Naive distance:     {stats['naive_distance']:.2f} m")
    print(f"Optimized distance: {stats['nn_distance']:.2f} m")
    print(f"Super distance:     {stats['two_opt_distance']:.2f} m")
    print()

    print("=== Total Energy Comparison ===")
    print(f"Naive energy:     {stats['naive_energy']:.2f} J")
    print(f"Optimized energy: {stats['nn_energy']:.2f} J")
    print(f"Super energy:     {stats['two_opt_energy']:.2f} J")
    if stats["brute_energy"] is not None:
        print(f"Brute energy:     {stats['brute_energy']:.2f} J")
        print(f"E2 / ENN  = {stats['two_opt_energy'] / stats['nn_energy']:.4f}")
        print(f"E2 / Emin = {stats['two_opt_energy'] / stats['brute_energy']:.4f}")
    print()

    print("=== Total Time (sum of per-segment Tmin) ===")
    print(f"Naive time:     {stats['naive_time']:.2f} s")
    print(f"Optimized time: {stats['nn_time']:.2f} s")
    print(f"Super time:     {stats['two_opt_time']:.2f} s")

    visualizer = Visualizer()
    visualizer.plot_energy_curve(physics, distance=1000.0)

    visualizer.plot_routes_three(
        waypoints=stats["waypoints"],
        naive_order=stats["naive_order"],
        optimized_order=stats["nn_order"],
        super_order=stats["two_opt_order"],
        filename="route_map.png",
        obstacles=stats["obstacles"],
        path_lookup=stats["path_lookup"],
    )

    visualizer.plot_total_energy_three(
        naive_energy=stats["naive_energy"],
        optimized_energy=stats["nn_energy"],
        super_energy=stats["two_opt_energy"],
        filename="total_energy.png",
    )

    print("\nPlots saved:")
    print(" - plots/energy_curve.png")
    print(" - plots/route_map.png")
    print(" - plots/total_energy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drone routing optimization with no-fly zones, A*, NN, 2-opt, and benchmarking"
    )
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use a fixed waypoint set with a no-fly zone",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run multi-configuration benchmark study",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random configurations per N",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=25.0,
        help="Grid spacing for A* pathfinding in meters",
    )
    args = parser.parse_args()

    if args.benchmark:
        benchmark_results(
            N_values=list(range(6, 13)),
            trials_per_N=args.trials,
            use_obstacles=False,
        )
    else:
        if args.test:
            cfg = get_test_sim_config()
        else:
            cfg = SimulationConfig(
                num_targets=args.num_targets,
                seed=args.seed,
                bounds=(0.0, 2000.0),
                use_obstacles=False,
                obstacles=[],
                grid_step=args.grid_step,
            )
        main(cfg)