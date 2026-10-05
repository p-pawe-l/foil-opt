import cma
import numpy as np

from .. import config
from ..evaluator import Evaluator


def run_cma(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    s = config.settings.optimizers.cma
    es = cma.CMAEvolutionStrategy(x0, s.sigma0, {"popsize": s.popsize, "seed": seed, "verbose": -9})
    while not es.stop() and not evaluate.exhausted:
        candidates = es.ask()
        es.tell(candidates, list(evaluate(candidates)))
