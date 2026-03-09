from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class RectangleZone:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    pad: float = 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return (
            self.xmin - self.pad <= x <= self.xmax + self.pad
            and self.ymin - self.pad <= y <= self.ymax + self.pad
        )

    def draw(
        self,
        ax: Axes,
        *,
        facecolor: str = "red",
        alpha: float = 0.22,
        label: str = "No-fly zone",
    ) -> None:
        rect = Rectangle(
            (self.xmin, self.ymin),
            self.xmax - self.xmin,
            self.ymax - self.ymin,
            facecolor=facecolor,
            edgecolor="darkred",
            linewidth=2.0,
            alpha=alpha,
            zorder=1,
            label=label,
        )
        ax.add_patch(rect)