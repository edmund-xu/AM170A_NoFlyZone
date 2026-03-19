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
        base_params = physics_model.p
        base_drag = base_params.drag_coeff

        T_low, T_high = physics_model.feasible_time_bounds(distance)

        # Main plotting grid for drag-based comparisons
        Ts = np.linspace(T_low, T_high, 400)

        # Wider grid for the no-drag validation curve so its minimum is visible
        T0_star_analytic = ((9.0 * base_params.mass * distance**2) / (2.0 * base_params.hover_power)) ** (1.0 / 3.0)
        T_no_drag_min = max(1e-3, min(0.55 * T_low, 0.75 * T0_star_analytic))
        Ts_no_drag = np.linspace(T_no_drag_min, T_high, 500)

        def energy_curve(drag_value: float, Tvals: np.ndarray) -> np.ndarray:
            test_params = replace(base_params, drag_coeff=drag_value)
            test_physics = DronePhysics(test_params)
            return np.array(
                [test_physics.segment_energy(distance, T) for T in Tvals],
                dtype=float,
            )

        def analytic_no_drag_energy(Tvals: np.ndarray) -> np.ndarray:
            m = base_params.mass
            P_h = base_params.hover_power
            return P_h * Tvals + (9.0 * m * distance**2) / (4.0 * Tvals**2)

        # Numerical curves
        E_base = energy_curve(base_drag, Ts)
        E_low_drag = energy_curve(0.5 * base_drag, Ts)
        E_high_drag = energy_curve(2.0 * base_drag, Ts)

        # No-drag numerical + analytical validation
        E_no_drag_num = energy_curve(0.0, Ts_no_drag)
        E_no_drag_analytic = analytic_no_drag_energy(Ts_no_drag)
        E0_star_analytic = float(analytic_no_drag_energy(np.array([T0_star_analytic]))[0])

        # Baseline optimum
        idx = int(np.argmin(E_base))
        T_opt = float(Ts[idx])
        E_opt = float(E_base[idx])

        fig, ax = plt.subplots(figsize=(11.5, 7.2))

        # No-drag numerical curve
        ax.plot(
            Ts_no_drag,
            E_no_drag_num,
            linestyle="-.",
            linewidth=3.0,
            label="No drag (numerical)",
        )

        # No-drag analytical validation overlay
        ax.plot(
            Ts_no_drag,
            E_no_drag_analytic,
            linestyle="--",
            linewidth=2.2,
            alpha=0.9,
            label="No drag (analytical)",
        )

        ax.plot(
            Ts,
            E_base,
            linewidth=3.2,
            label=f"Baseline drag (C = {base_drag:.2f})",
        )
        ax.plot(
            Ts,
            E_low_drag,
            linestyle=":",
            linewidth=3.0,
            label="Reduced drag",
        )
        ax.plot(
            Ts,
            E_high_drag,
            linestyle="--",
            linewidth=3.0,
            label="Increased drag",
        )

        # Baseline optimum marker
        ax.scatter(
            [T_opt],
            [E_opt],
            s=180,
            edgecolor="black",
            linewidth=1.5,
            zorder=6,
            label=f"Baseline optimum ≈ {T_opt:.1f} s",
        )
        ax.axvline(T_opt, linestyle="--", linewidth=2.0, alpha=0.7)

        # Analytical no-drag optimum marker
        ax.scatter(
            [T0_star_analytic],
            [E0_star_analytic],
            s=140,
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
            label=f"No-drag analytical optimum ≈ {T0_star_analytic:.1f} s",
        )

        # Optional: show feasible lower bound for the constrained search
        ax.axvline(T_low, linestyle=":", linewidth=2.0, alpha=0.8)
        ax.text(
            T_low,
            ax.get_ylim()[0] if len(ax.lines) == 0 else min(E_base.min(), E_no_drag_num.min()) * 1.02,
            " feasible $T_{low}$",
            fontsize=13,
            ha="left",
            va="bottom",
        )

        ax.set_xlabel("Segment travel time $T$ [s]", fontsize=18)
        ax.set_ylabel("Segment energy $E(T)$ [J]", fontsize=18)
        ax.set_title(
            f"Segment Energy vs Travel Time ($d={distance:.0f}$ m)",
            fontsize=24,
            pad=14,
        )
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=13, loc="best", frameon=True)

        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "energy_curve.png", dpi=250, bbox_inches="tight")
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
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 22))
        # Small top strip for title + legend, then 2 rows of plots
        gs = GridSpec(
            3, 2,
            figure=fig,
            height_ratios=[0.06, 1, 1],
            hspace=0.32,
            wspace=0.25,
        )

        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis("off")

        ax0 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[1, 1])
        ax2 = fig.add_subplot(gs[2, 0])
        ax_blank = fig.add_subplot(gs[2, 1])
        ax_blank.axis("off")

        axes = [ax0, ax1, ax2]
        obstacles = obstacles or []

        def draw_polyline(ax: Axes, pts: list[np.ndarray], *, label: str = "") -> None:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]

            ax.plot(
                xs, ys,
                linestyle="--",
                linewidth=2.4,
                alpha=0.95,
                label=label,
                zorder=2,
            )

            if len(xs) >= 2:
                mid = len(xs) // 2
                x0, y0 = xs[max(0, mid - 1)], ys[max(0, mid - 1)]
                x1, y1 = xs[mid], ys[mid]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=2.0, alpha=0.85),
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
                s=160,
                edgecolor="black",
                linewidth=1.4,
                zorder=3,
                label="Waypoint",
            )

            start_x, start_y = waypoints[0]
            ax.scatter(
                start_x,
                start_y,
                s=340,
                facecolor="none",
                edgecolor="green",
                linewidth=3.2,
                zorder=4,
                label="Start/End",
            )

            for idx, (x, y) in enumerate(waypoints):
                ax.text(x + 12, y + 12, str(idx), fontsize=16)

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
                        linewidth=2.4,
                        alpha=0.95,
                        label="Route" if k == 0 else "",
                        zorder=2,
                    )

                    ax.annotate(
                        "",
                        xy=(xj, yj),
                        xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->", lw=2.0, alpha=0.85),
                    )

            ax.set_title(title, fontsize=22, pad=12)
            ax.set_xlabel("x [m]", fontsize=18)
            ax.set_ylabel("y [m]", fontsize=18)
            ax.tick_params(axis="both", labelsize=15)
            ax.grid(True, alpha=0.35)
            ax.set_aspect("equal", adjustable="box")

        draw_route(axes[0], naive_order, "(a) Naive Route")
        draw_route(axes[1], optimized_order, "(b) Nearest Neighbor Route")
        draw_route(axes[2], super_order, "(c) NN + 2-opt Route")

        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        ax_header.legend(
            by_label.values(),
            by_label.keys(),
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=4,
            fontsize=16,
            frameon=True,
            columnspacing=2.0,
            handlelength=2.4,
            borderpad=0.8,
        )

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
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
            filename="E2_over_ENN.png",
        )

        if brute_Ns and mean_e2_over_emin is not None:
            self.plot_benchmark_ratio(
                N_values=brute_Ns,
                mean_values=mean_e2_over_emin,
                std_values=std_e2_over_emin,
                ylabel="Mean ratio E2 / Emin",
                title="NN + 2-opt relative to brute-force optimum",
                filename="E2_over_Emin.png",
            )