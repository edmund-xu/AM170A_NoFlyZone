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
):

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
                bounds=(0.0, 2000.0),

                # BENCHMARKS DO NOT USE OBSTACLES
                use_obstacles=False,
                obstacles=[],

                grid_step=25.0,
            )

            stats = compute_route_stats(cfg)

            E_nn = float(stats["nn_energy"])
            E_2 = float(stats["two_opt_energy"])
            E_min = stats["brute_energy"]

            ratios_twoopt_vs_nn.append(E_2 / E_nn)

            if E_min is not None:
                ratios_twoopt_vs_brute.append(E_2 / float(E_min))

        mean_ratio_twoopt_vs_nn.append(np.mean(ratios_twoopt_vs_nn))
        std_ratio_twoopt_vs_nn.append(np.std(ratios_twoopt_vs_nn))

        if ratios_twoopt_vs_brute:
            brute_Ns.append(N)
            mean_ratio_twoopt_vs_brute.append(np.mean(ratios_twoopt_vs_brute))
            std_ratio_twoopt_vs_brute.append(np.std(ratios_twoopt_vs_brute))

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
    print(" - plots/benchmark_E2_over_Emin.png")


def main(simulation_config: SimulationConfig | None = None):

    stats = compute_route_stats(simulation_config)

    physics = DronePhysics(get_default_params())
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

    print("Plots saved:")
    print(" - plots/energy_curve.png")
    print(" - plots/route_map.png")
    print(" - plots/total_energy.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("-n", "--num-targets", type=int, default=7)
    parser.add_argument("-s", "--seed", type=int, default=None)

    args = parser.parse_args()

    if args.benchmark:

        benchmark_results(
            N_values=list(range(6, 13)),
            trials_per_N=args.trials,
        )

    else:

        # DEFAULT RUN USES A NO-FLY ZONE

        cfg = SimulationConfig(
            num_targets=args.num_targets,
            seed=args.seed,
            bounds=(0.0, 2000.0),

            use_obstacles=True,
            obstacles=[
                RectangleZone(xmin=350, xmax=650, ymin=350, ymax=650, pad=20),
                RectangleZone(xmin=900, xmax=1150, ymin=200, ymax=500, pad=20),
                RectangleZone(xmin=1200, xmax=1500, ymin=1100, ymax=1400, pad=20),
            ],
            grid_step=25.0,
        )

        main(cfg)