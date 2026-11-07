# This is a set of functions written by claude solely for fancy plotting.

# fig, ax = new_chart("Title", "Subtitle", "x label", "y label")
# draw_line(ax, x, y, color=SERIES_COLORS[0], band=(lo, hi))
# draw_points(ax, x, y, color=SERIES_COLORS[1], label="empirical")
# label_end(ax, x, y, "22 ms")      # optional, for a single-series chart
# save_chart(fig, ax, path)         # adds a legend when two or more series are labelled

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # charts are written to files, so no display is needed
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Categorical slots 1-8. Use them in this order for series 1, 2, 3, ... and never cycle.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]


def new_chart(title: str, subtitle: str | None = None, xlabel: str = "", ylabel: str = "",
              figsize: tuple[float, float] = (8, 4.5)) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    fig.text(0.125, 0.955, title, color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(0.125, 0.905, subtitle, color=INK_2, fontsize=9.5, ha="left", va="top")
    fig.subplots_adjust(top=0.84 if subtitle else 0.88, bottom=0.14, left=0.125, right=0.97)
    return fig, ax


def draw_line(ax: Axes, x, y, *, color: str = SERIES_COLORS[0], band=None,
              label: str | None = None, markers: str = "all") -> None:
    # band: optional (low, high) arrays drawn as a faint wash behind the line.
    # markers: "all" dots every point, "end" dots only the last one (for dense series), "none".
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=color, alpha=0.10, linewidth=0)
    markevery = {"all": None, "end": [len(x) - 1], "none": None}[markers]
    ax.plot(x, y, color=color, linewidth=1.5, solid_capstyle="round", solid_joinstyle="round",
            marker=None if markers == "none" else "o", markersize=5.5, markevery=markevery,
            markerfacecolor=color, markeredgewidth=0, label=label, zorder=2)


def draw_points(ax: Axes, x, y, *, color: str = SERIES_COLORS[0], label: str | None = None) -> None:
    ax.scatter(x, y, s=36, color=color, linewidths=0, label=label, zorder=3)


def label_end(ax: Axes, x, y, text: str) -> None:
    # Labels the last point, in the free space to its right. Call after setting the left x-limit
    # and before save_chart().
    x_first, x_last = float(x[0]), float(x[-1])
    ax.annotate(text, xy=(x_last, y[-1]), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(right=x_last + 0.25 * (x_last - x_first))
    left = ax.get_xlim()[0]
    ax.set_xticks([t for t in ax.get_xticks() if left <= t <= x_last])


def save_chart(fig: Figure, ax: Axes, path: str | Path, zero_baseline: bool = True) -> None:
    if zero_baseline:
        ax.set_ylim(bottom=0)
    if len(ax.get_legend_handles_labels()[1]) >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
