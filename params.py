"""
Physical constants and mission configuration for drone routing.

Stop at each waypoint:
- Segment velocity profile: v(t) = alpha * t * (T - t)
- Linear drag: F = m dv/dt + C v
- Energy: integral of (hover power + |F·v|) over time
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from obstacles import RectangleZone


@dataclass(frozen=True)
class DroneParams:
    """Physical + numerical params for the segment energy model."""
    mass: float
    drag_coeff: float
    hover_power: float
    v_max: float
    a_max: float
    integration_steps: int = 600
    t_upper_per_meter: float = 0.7


@dataclass
class SimulationConfig:
    num_targets: int = 5
    bounds: tuple[float, float] = (0.0, 2000.0)
    waypoint_set: Optional[list[tuple[float, float]]] = None
    seed: Optional[int] = None

    use_obstacles: bool = False
    obstacles: list[RectangleZone] = field(default_factory=list)
    grid_step: float = 25.0


def get_default_params() -> DroneParams:
    return DroneParams(
        mass=1.38,
        drag_coeff=1.00,
        hover_power=60.0,
        v_max=18.0,
        a_max=6.0,
        integration_steps=600,
        t_upper_per_meter=0.7,
    )


def get_default_sim_config() -> SimulationConfig:
    return SimulationConfig()


def get_test_sim_config() -> SimulationConfig:
    """
    Fixed waypoint test set with a centered rectangular no-fly zone.
    """
    return SimulationConfig(
        num_targets=6,
        waypoint_set=[
            (100.0, 100.0),
            (500.0, 120.0),
            (520.0, 520.0),
            (120.0, 540.0),
            (250.0, 750.0),
            (760.0, 760.0),
        ],
        bounds=(0.0, 900.0),
        use_obstacles=True,
        obstacles=[
            RectangleZone(
                xmin=300.0,
                xmax=600.0,
                ymin=300.0,
                ymax=600.0,
                pad=20.0,
            )
        ],
        grid_step=25.0,
    )