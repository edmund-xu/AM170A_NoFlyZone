from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from obstacles import RectangleZone
from physics import DronePhysics

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


class Visualizer:
    def plot_energy_curve(self, physics_model: DronePhysics, distance: float) -> None:
        T_low, T_high = physics_model.feasible_time_bounds(distance)
        Ts = np.linspace(T_low, T_high, 250)

        base_params = physics_model.p
        base_drag = base_params.drag_coeff

        def energy_curve(drag_value: float) -> np.ndarray:
            test_params = replace(base_params, drag_coeff=drag_value)
            test_physics = DronePhysics(test_params)
            return np.array(
                [test_physics.segment_energy(distance, T) for T in Ts],
                dtype=float,
            )

        E_ideal = energy_curve(0.0)
        E_base = energy_curve(base_drag)
        E_low_drag = energy_curve(0.5 * base_drag)
        E_high_drag = energy_curve(2.0 * base_drag)

        idx = int(np.argmin(E_base))
        T_opt = float(Ts[idx])
        E_opt = float(E_base[idx])

        fig, ax = plt.subplots(figsize=(9.2, 5.8))

        ax.plot(
            Ts, E_ideal,
            linestyle="-.",
            linewidth=2.0,
            label="No drag reference",
        )
        ax.plot(
            Ts, E_base,
            linewidth=2.6,
            label=f"Baseline drag (C = {base_drag:.2f})",
        )
        ax.plot(
            Ts, E_low_drag,
            linestyle=":",
            linewidth=2.0,
            label="Reduced drag",
        )
        ax.plot(
            Ts, E_high_drag,
            linestyle="--",
            linewidth=2.0,
            label="Increased drag",
        )

        ax.scatter(
            [T_opt], [E_opt],
            s=110,
            edgecolor="black",
            zorder=5,
            label=f"Optimal time ≈ {T_opt:.1f} s",
        )
        ax.axvline(T_opt, linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Segment travel time T [s]", fontsize=13)
        ax.set_ylabel("Segment energy E(T) [J]", fontsize=13)
        ax.set_title(f"Segment Energy vs Travel Time (d = {distance:.0f} m)", fontsize=17)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "energy_curve.png", dpi=150)
        plt.close(fig)

    def plot_routes_three(
        self,
        waypoints: np.ndarray,
        naive_order: list[int],
        optimized_order: list[int],
        super_order: list[int],
        filename: str = "route_map.png",
        obstacles: list[RectangleZone] | None = None,
        path_lookup: dict[tuple[int, int], list[np.ndarray]] | None = None,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        obstacles = obstacles or []

        def draw_polyline(ax: Axes, pts: list[np.ndarray], *, label: str = "") -> None:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]

            ax.plot(xs, ys, linestyle="--", linewidth=2, alpha=0.9, label=label, zorder=2)

            if len(xs) >= 2:
                mid = len(xs) // 2
                x0, y0 = xs[max(0, mid - 1)], ys[max(0, mid - 1)]
                x1, y1 = xs[mid], ys[mid]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.8, alpha=0.8),
                )

        def draw_route(ax: Axes, order: list[int], title: str) -> None:
            for obs_idx, obs in enumerate(obstacles):
                obs.draw(
                    ax,
                    facecolor="red",
                    alpha=0.22 if obs_idx == 0 else 0.18,
                    label="No-fly zone" if obs_idx == 0 else "",
                )

            ax.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                s=90,
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
                label="Waypoint",
            )

            start_x, start_y = waypoints[0]
            ax.scatter(
                start_x,
                start_y,
                s=200,
                facecolor="none",
                edgecolor="green",
                linewidth=3,
                zorder=4,
                label="Start/End",
            )

            for idx, (x, y) in enumerate(waypoints):
                ax.text(x + 10, y + 10, str(idx), fontsize=11)

            for k in range(len(order)):
                i = order[k]
                j = order[(k + 1) % len(order)]

                if path_lookup is not None and (i, j) in path_lookup:
                    draw_polyline(ax, path_lookup[(i, j)], label="Route" if k == 0 else "")
                else:
                    xi, yi = waypoints[i]
                    xj, yj = waypoints[j]

                    ax.plot(
                        [xi, xj],
                        [yi, yj],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.9,
                        label="Route" if k == 0 else "",
                        zorder=2,
                    )

                    ax.annotate(
                        "",
                        xy=(xj, yj),
                        xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->", lw=1.8, alpha=0.8),
                    )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, alpha=0.35)
            ax.set_aspect("equal", adjustable="box")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=11)

        draw_route(axes[0], naive_order, "Naive Route")
        draw_route(axes[1], optimized_order, "Nearest Neighbor Route")
        draw_route(axes[2], super_order, "NN + 2-opt Route")

        fig.suptitle("Drone Route Comparison", fontsize=20, y=1.02)
        fig.tight_layout()

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_total_energy_three(
        self,
        naive_energy: float,
        optimized_energy: float,
        super_energy: float,
        filename: str = "total_energy.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 6))

        labels = ["Naive", "Nearest Neighbor", "NN + 2-opt"]
        vals = [naive_energy, optimized_energy, super_energy]

        bars = ax.bar(labels, vals, edgecolor="black", linewidth=1.2)

        ax.set_title("Total Energy Comparison", fontsize=20, pad=15)
        ax.set_ylabel("Energy [J]", fontsize=14)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,.0f} J",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        ax.set_ylim(0, max(vals) * 1.15)
        fig.tight_layout()

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close(fig)

    def plot_benchmark_ratio(
        self,
        N_values: list[int],
        mean_values: list[float],
        std_values: list[float] | None,
        *,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.8, 5.8))

        if std_values is not None and len(std_values) == len(N_values):
            ax.errorbar(
                N_values,
                mean_values,
                yerr=std_values,
                marker="o",
                markersize=7,
                linewidth=2.2,
                capsize=4,
            )
        else:
            ax.plot(
                N_values,
                mean_values,
                marker="o",
                markersize=7,
                linewidth=2.2,
            )

        ax.axhline(1.0, linestyle="--", linewidth=1.4, alpha=0.7)
        ax.set_xlabel("Number of waypoints N", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=17)
        ax.grid(True, alpha=0.35)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=180)
        plt.close(fig)

    def plot_benchmark_results(
        self,
        N_values_nn: list[int],
        mean_e2_over_enn: list[float],
        std_e2_over_enn: list[float] | None,
        brute_Ns: list[int] | None = None,
        mean_e2_over_emin: list[float] | None = None,
        std_e2_over_emin: list[float] | None = None,
    ) -> None:
        self.plot_benchmark_ratio(
            N_values=N_values_nn,
            mean_values=mean_e2_over_enn,
            std_values=std_e2_over_enn,
            ylabel="Mean ratio E2 / ENN",
            title="NN + 2-opt relative to Nearest Neighbor",
            filename="benchmark_E2_over_ENN.png",
        )

        if brute_Ns and mean_e2_over_emin is not None:
            self.plot_benchmark_ratio(
                N_values=brute_Ns,
                mean_values=mean_e2_over_emin,
                std_values=std_e2_over_emin,
                ylabel="Mean ratio E2 / Emin",
                title="NN + 2-opt relative to brute-force optimum",
                filename="benchmark_E2_over_Emin.png",
            )