"""Minimal airfoil shape optimization: CMA-ES + NeuralFoil.

Goal: maximize average L/D over a few angles of attack at Re = 5e5,
keeping max thickness >= 12% and only trusting confident NeuralFoil predictions.
Starts from a NACA 2412 described by Kulfan (CST) parameters.
"""

import aerosandbox as asb
import cma
import matplotlib.pyplot as plt
import neuralfoil as nf
import numpy as np

RE = 5e5
ALPHAS = np.array([2.0, 4.0, 6.0])
MIN_THICKNESS = 0.12
MIN_CONFIDENCE = 0.9
X_CHECK = np.linspace(0.01, 0.99, 50)

baseline = asb.KulfanAirfoil("naca2412")
TE_THICKNESS = baseline.TE_thickness 

def to_genome(af: asb.KulfanAirfoil) -> np.ndarray:
    return np.concatenate([af.upper_weights, af.lower_weights, [af.leading_edge_weight]])


def to_airfoil(x: np.ndarray) -> asb.KulfanAirfoil:
    return asb.KulfanAirfoil(
        upper_weights=x[:8],
        lower_weights=x[8:16],
        leading_edge_weight=x[16],
        TE_thickness=TE_THICKNESS,
    )


def evaluate(population: list[np.ndarray]) -> list[float]:
    """Return a cost per candidate (CMA-ES minimizes, so cost = -L/D + penalties)."""
    pop = np.array(population) 
    n, n_a = len(pop), len(ALPHAS)

    rep = np.repeat(pop, n_a, axis=0)
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(
            upper_weights=rep[:, :8].T,
            lower_weights=rep[:, 8:16].T,
            leading_edge_weight=rep[:, 16],
            TE_thickness=np.full(len(rep), TE_THICKNESS),
        ),
        alpha=np.tile(ALPHAS, n),
        Re=RE,
        model_size="large",
    )
    ld = (aero["CL"] / aero["CD"]).reshape(n, n_a)
    conf = aero["analysis_confidence"].reshape(n, n_a)

    costs = []
    for i, x in enumerate(pop):
        t = to_airfoil(x).local_thickness(X_CHECK)
        penalty = 0.0
        penalty += 1e3 * max(0.0, MIN_THICKNESS - t.max()) 
        penalty += 1e3 * np.sum(np.maximum(0.0, -t))
        penalty += 1e2 * np.sum(np.maximum(0.0, MIN_CONFIDENCE - conf[i]))
        costs.append(-ld[i].mean() + penalty)
    return costs

def main():
    x0 = to_genome(baseline)
    print(f"Baseline NACA 2412: mean L/D = {-evaluate([x0])[0]:.1f}")

    es = cma.CMAEvolutionStrategy(x0, 0.03, {"popsize": 24, "maxiter": 150, "seed": 1, "verbose": -9})
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, evaluate(candidates))
        if es.countiter % 10 == 0:
            print(f"iter {es.countiter:4d}  best mean L/D = {-es.best.f:.1f}")

    best = to_airfoil(es.best.x)
    print(f"\nOptimized: mean L/D = {-es.best.f:.1f}, max thickness = {best.max_thickness():.3f}")

    for name, af in [("NACA 2412", baseline), ("optimized", best)]:
        r = af.get_aero_from_neuralfoil(alpha=ALPHAS, Re=RE, model_size="large")
        for a, cl, cd, cm, c in zip(ALPHAS, r["CL"], r["CD"], r["CM"], r["analysis_confidence"]):
            print(f"  {name:10s} α={a:3.0f}°  CL={cl:.3f}  CD={cd:.5f}  L/D={cl / cd:6.1f}  CM={cm:+.3f}  conf={c:.2f}")

    fig, ax = plt.subplots(figsize=(9, 3))
    for name, af in [("NACA 2412", baseline), ("optimized", best)]:
        xy = af.coordinates
        ax.plot(xy[:, 0], xy[:, 1], label=name)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"CMA-ES + NeuralFoil, Re={RE:.0e}, α={ALPHAS.tolist()}°")
    fig.tight_layout()
    fig.savefig("optimized_airfoil.png", dpi=150)
    print("\nSaved optimized_airfoil.png")


if __name__ == "__main__":
    main()
