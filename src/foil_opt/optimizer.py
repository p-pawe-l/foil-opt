"""Optimizers over the airfoil genome. Each returns (best genome, best cost)."""

import cma
import numpy as np
from scipy.optimize import differential_evolution

from .config import CMA_OPTIONS, DE_HALF_WIDTH, DE_OPTIONS, LOG_EVERY, SIGMA0
from .objective import evaluate


def _log(iteration: int, best_cost: float) -> None:
    if iteration % LOG_EVERY == 0:
        print(f"iter {iteration:4d}  best mean L/D = {-best_cost:.1f}")


def optimize_cma(x0: np.ndarray) -> tuple[np.ndarray, float]:
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, CMA_OPTIONS)
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, evaluate(candidates))
        _log(es.countiter, es.best.f)
    return es.best.x, es.best.f


def optimize_de(x0: np.ndarray) -> tuple[np.ndarray, float]:
    bounds = list(zip(x0 - DE_HALF_WIDTH, x0 + DE_HALF_WIDTH))
    iteration = 0

    def callback(intermediate_result) -> None:
        nonlocal iteration
        iteration += 1
        _log(iteration, intermediate_result.fun)

    result = differential_evolution(
        # vectorized: scipy passes the population as columns, so one NeuralFoil call per generation
        lambda xs: np.array(evaluate(list(xs.T))),
        bounds,
        x0=x0,
        vectorized=True,
        updating="deferred",  # required by vectorized; whole generation is updated at once
        polish=False,  # gradient polishing would call evaluate one candidate at a time
        callback=callback,
        **DE_OPTIONS,
    )
    return result.x, result.fun


OPTIMIZERS = {
    "cma": ("CMA-ES", optimize_cma),
    "de": ("Differential Evolution", optimize_de),
}
