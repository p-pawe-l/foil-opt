"""Printing and plotting of results."""

import aerosandbox as asb
import matplotlib.pyplot as plt

from .config import ALPHAS, MODEL_SIZE, RE

Named = list[tuple[str, asb.KulfanAirfoil]]


def print_polars(airfoils: Named) -> None:
    for name, af in airfoils:
        r = af.get_aero_from_neuralfoil(alpha=ALPHAS, Re=RE, model_size=MODEL_SIZE)
        for a, cl, cd, cm, c in zip(ALPHAS, r["CL"], r["CD"], r["CM"], r["analysis_confidence"]):
            print(f"  {name:10s} α={a:3.0f}°  CL={cl:.3f}  CD={cd:.5f}  L/D={cl / cd:6.1f}  CM={cm:+.3f}  conf={c:.2f}")


def plot_shapes(airfoils: Named, method: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    for name, af in airfoils:
        xy = af.coordinates
        ax.plot(xy[:, 0], xy[:, 1], label=name)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{method} + NeuralFoil, Re={RE:.0e}, α={ALPHAS.tolist()}°")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
