import cma
import numpy as np

from ..config import CMA
from ..evaluator import Evaluator


def run_cma(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    es = cma.CMAEvolutionStrategy(x0, CMA["sigma0"], {"popsize": CMA["popsize"], "seed": seed, "verbose": -9})
    while not es.stop() and not evaluate.exhausted:
        candidates = es.ask()
        es.tell(candidates, list(evaluate(candidates)))
