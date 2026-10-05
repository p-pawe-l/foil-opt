"""Figures for the README, saved as PNGs in assets/."""

from pathlib import Path

import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASELINE = "#898781"
INK, INK_SECONDARY, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "lines.linewidth": 2,
    "font.size": 10,
})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_airfoil(ax, af: asb.Airfoil, color: str, label: str, **kwargs) -> None:
    xy = af.coordinates
    ax.plot(xy[:, 0], xy[:, 1], color=color, label=label, **kwargs)


def plot_shapes(designs: dict[str, asb.Airfoil], baseline: asb.Airfoil, baseline_label: str, path: Path) -> None:
    """One panel per design, each over the baseline, so near-identical shapes stay readable."""
    fig, axes = plt.subplots(len(designs), 1, figsize=(8, 1.9 * len(designs)), sharex=True, squeeze=False)
    for i, (ax, (name, af)) in enumerate(zip(axes[:, 0], designs.items())):
        plot_airfoil(ax, baseline, BASELINE, baseline_label, linewidth=1.2, linestyle="--")
        plot_airfoil(ax, af, SERIES[0], name)
        ax.set_aspect("equal")
        ax.set_ylim(-0.08, 0.13)
        ax.set_title(name, loc="left", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", ncols=2)
    axes[-1, 0].set_xlabel("x/c")
    save(fig, path)


def plot_convergence(histories: dict[str, list[np.ndarray]], gradient_cd: float | None, path: Path) -> None:
    """Median best feasible mean CD across seeds, with the min-max band."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (name, runs) in zip(SERIES, histories.items()):
        budget = max(h[-1, 0] for h in runs)
        grid = np.linspace(0, budget, 400)
        curves = np.array([best_at(h, grid) for h in runs])
        ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=color, alpha=0.15, linewidth=0)
        ax.plot(grid, np.nanmedian(curves, 0), color=color, label=name)
    if gradient_cd is not None:
        ax.axhline(gradient_cd, color=INK_SECONDARY, linestyle=":", linewidth=1.5, label="Gradient (IPOPT)")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best feasible mean CD")
    ax.set_title(f"Convergence, median and range over {len(next(iter(histories.values())))} seeds", loc="left")
    ax.legend()
    save(fig, path)


def best_at(history: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step-interpolate (evals, best cost) onto grid; NaN until the first feasible design."""
    idx = np.searchsorted(history[:, 0], grid, side="right") - 1
    values = np.where(idx >= 0, history[np.clip(idx, 0, None), 1], np.nan)
    return np.where(values < 1, values, np.nan)


def plot_pareto(mean_cd, max_abs_cm, designs: dict[str, asb.Airfoil], marked: dict[str, tuple[float, float]],
                baseline: asb.Airfoil, baseline_label: str, path: Path) -> None:
    fig, (ax, ax_shapes) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1.3, 1]})
    ax.plot(max_abs_cm, mean_cd, "o", color=SERIES[0], markersize=5, label="NSGA-II front")
    marker_colors = [BASELINE, SERIES[6]]
    for color, (name, (cm, cd)) in zip(marker_colors, marked.items()):
        ax.plot(cm, cd, "D", color=color, markersize=8, markeredgecolor=SURFACE, label=name)
    ax.set_xlabel("max |CM|")
    ax.set_ylabel("Mean CD")
    ax.set_title("Drag vs pitching moment", loc="left")
    ax.legend()

    plot_airfoil(ax_shapes, baseline, BASELINE, baseline_label, linewidth=1.2, linestyle="--")
    for color, (name, af) in zip(SERIES[1:], designs.items()):
        plot_airfoil(ax_shapes, af, color, name)
    ax_shapes.set_aspect("equal")
    ax_shapes.set_xlabel("x/c")
    ax_shapes.set_title("Designs along the front", loc="left")
    ax_shapes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncols=2)
    save(fig, path)
