"""Minimal airfoil shape optimization: CMA-ES or Differential Evolution + NeuralFoil.

Goal: maximize average L/D over a few angles of attack at Re = 5e5,
keeping max thickness >= 12%, max camber <= 4% and only trusting confident NeuralFoil predictions.
Starts from a NACA 2412 described by Kulfan (CST) parameters.

Usage: python src/cma_example.py [--optimizer {cma,de}]

Modules (src/foil_opt/):
  config     problem settings
  geometry   genome <-> Kulfan airfoil
  aero       batched NeuralFoil call
  objective  cost = -mean L/D + penalties
  optimizer  CMA-ES and Differential Evolution
  report     printed polars and shape plot
"""

import argparse

from foil_opt.geometry import baseline, to_airfoil, to_genome
from foil_opt.objective import evaluate
from foil_opt.optimizer import OPTIMIZERS
from foil_opt.report import plot_shapes, print_polars

PLOT_PATHS = {"cma": "optimized_airfoil.png", "de": "optimized_airfoil_de.png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="cma")
    key = parser.parse_args().optimizer
    method, optimize = OPTIMIZERS[key]

    x0 = to_genome(baseline)
    print(f"Baseline NACA 2412: mean L/D = {-evaluate([x0])[0]:.1f}")
    print(f"Optimizer: {method}")

    best_x, best_cost = optimize(x0)

    best = to_airfoil(best_x)
    print(f"\nOptimized: mean L/D = {-best_cost:.1f}, max thickness = {best.max_thickness():.3f}, max camber = {best.max_camber():.3f}")

    airfoils = [("NACA 2412", baseline), ("optimized", best)]
    print_polars(airfoils)
    plot_shapes(airfoils, method, PLOT_PATHS[key])


if __name__ == "__main__":
    main()
