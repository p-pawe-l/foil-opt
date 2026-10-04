"""Minimal airfoil shape optimization: CMA-ES + NeuralFoil.

Goal: maximize average L/D over a few angles of attack at Re = 5e5,
keeping max thickness >= 12% and only trusting confident NeuralFoil predictions.
Starts from a NACA 2412 described by Kulfan (CST) parameters.

Modules (src/foil_opt/):
  config     problem settings
  geometry   genome <-> Kulfan airfoil
  aero       batched NeuralFoil call
  objective  cost = -mean L/D + penalties
  optimizer  CMA-ES loop
  report     printed polars and shape plot
"""

from foil_opt.geometry import baseline, to_airfoil, to_genome
from foil_opt.objective import evaluate
from foil_opt.optimizer import optimize
from foil_opt.report import plot_shapes, print_polars


def main():
    x0 = to_genome(baseline)
    print(f"Baseline NACA 2412: mean L/D = {-evaluate([x0])[0]:.1f}")

    es = optimize(x0)

    best = to_airfoil(es.best.x)
    print(f"\nOptimized: mean L/D = {-es.best.f:.1f}, max thickness = {best.max_thickness():.3f}")

    airfoils = [("NACA 2412", baseline), ("optimized", best)]
    print_polars(airfoils)
    plot_shapes(airfoils)


if __name__ == "__main__":
    main()
