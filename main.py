from __future__ import annotations

import argparse

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


def main(simulation_config: SimulationConfig | None = None) -> None:
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
        distance_matrix, path_lookup = planner.build_obstacle_aware_distance_matrix(waypoints)
        planning_mode = "Obstacle-aware A* path lengths"
    else:
        distance_matrix = targets.get_distance_matrix(waypoints)
        planning_mode = "Straight-line Euclidean distances"

    energy_matrix, time_matrix = optimizer.build_energy_matrix(distance_matrix)

    # --- Three routes ---
    naive_order = list(range(n))
    optimized_order = optimizer.solve_tsp(energy_matrix, method="nearest_neighbor")
    super_order = optimizer.solve_tsp(energy_matrix, method="nn_2opt")

    def route_energy(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += float(energy_matrix[i, j])
        return total

    def route_time(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += float(time_matrix[i, j])
        return total

    def route_distance(order: list[int]) -> float:
        total = 0.0
        for k in range(len(order)):
            i, j = order[k], order[(k + 1) % len(order)]
            total += float(distance_matrix[i, j])
        return total

    naive_energy = route_energy(naive_order)
    optimized_energy = route_energy(optimized_order)
    super_energy = route_energy(super_order)

    naive_time = route_time(naive_order)
    optimized_time = route_time(optimized_order)
    super_time = route_time(super_order)

    naive_distance = route_distance(naive_order)
    optimized_distance = route_distance(optimized_order)
    super_distance = route_distance(super_order)

    print("=== Drone Routing Optimization (Stop-at-Waypoint) ===")
    print(f"Planning mode: {planning_mode}")
    print(f"Waypoints (x,y):\n{waypoints}\n")

    if config.use_obstacles and config.obstacles:
        print("No-fly zones:")
        for idx, obs in enumerate(config.obstacles):
            print(
                f"  Zone {idx}: "
                f"x=[{obs.xmin:.1f}, {obs.xmax:.1f}], "
                f"y=[{obs.ymin:.1f}, {obs.ymax:.1f}], "
                f"pad={obs.pad:.1f}"
            )
        print()

    print("=== Route Indices (cycle closes back to 0) ===")
    print(f"Naive:     {naive_order}")
    print(f"Optimized: {optimized_order} (nearest neighbor)")
    print(f"Super:     {super_order} (nearest neighbor + 2-opt)\n")

    print("=== Total Distance Comparison ===")
    print(f"Naive distance:     {naive_distance:.2f} m")
    print(f"Optimized distance: {optimized_distance:.2f} m")
    print(f"Super distance:     {super_distance:.2f} m\n")

    print("=== Total Energy Comparison ===")
    print(f"Naive energy:     {naive_energy:.2f} J")
    print(f"Optimized energy: {optimized_energy:.2f} J")
    print(f"Super energy:     {super_energy:.2f} J\n")

    print("=== Total Time (sum of per-segment Tmin) ===")
    print(f"Naive time:     {naive_time:.2f} s")
    print(f"Optimized time: {optimized_time:.2f} s")
    print(f"Super time:     {super_time:.2f} s")

    visualizer = Visualizer()
    visualizer.plot_energy_curve(physics, distance=1000.0)

    visualizer.plot_routes_three(
        waypoints=waypoints,
        naive_order=naive_order,
        optimized_order=optimized_order,
        super_order=super_order,
        filename="route_map.png",
        obstacles=config.obstacles,
        path_lookup=path_lookup,
    )

    visualizer.plot_total_energy_three(
        naive_energy=naive_energy,
        optimized_energy=optimized_energy,
        super_energy=super_energy,
        filename="total_energy.png",
    )

    print("\nPlots saved:")
    print(" - plots/energy_curve.png")
    print(" - plots/route_map.png")
    print(" - plots/total_energy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drone routing optimization with no-fly zones, A*, NN, and 2-opt"
    )
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Use a fixed waypoint set with a no-fly zone")
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="Enable a centered rectangular no-fly zone for random waypoint runs",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=25.0,
        help="Grid spacing for A* pathfinding in meters",
    )
    args = parser.parse_args()

    if args.test:
        cfg = get_test_sim_config()
    else:
        low, high = (0.0, 2000.0)
        obstacles = []
        if args.obstacles:
            center = 0.5 * (low + high)
            half_width = 120.0
            half_height = 120.0
            obstacles = [
                RectangleZone(
                    xmin=center - half_width,
                    xmax=center + half_width,
                    ymin=center - half_height,
                    ymax=center + half_height,
                    pad=20.0,
                )
            ]

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