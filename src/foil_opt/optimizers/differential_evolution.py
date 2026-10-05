import numpy as np
from scipy.optimize import differential_evolution

from ..config import DE
from ..evaluator import Evaluator


def run_de(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    differential_evolution(
        lambda xs: evaluate(xs.T),  # vectorized: scipy passes candidates as columns
        bounds=list(zip(x0 - DE["half_width"], x0 + DE["half_width"])),
        x0=x0,
        popsize=DE["popsize"],
        mutation=DE["mutation"],
        recombination=DE["recombination"],
        maxiter=10**6,  # the evaluator's budget stops the run
        callback=lambda intermediate_result: evaluate.exhausted,
        vectorized=True,
        updating="deferred",
        polish=False,
        rng=seed,
    )
