"""CMA-ES loop over the airfoil genome."""

import cma
import numpy as np

from .config import CMA_OPTIONS, LOG_EVERY, SIGMA0
from .objective import evaluate


def optimize(x0: np.ndarray) -> cma.CMAEvolutionStrategy:
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, CMA_OPTIONS)
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, evaluate(candidates))
        if es.countiter % LOG_EVERY == 0:
            print(f"iter {es.countiter:4d}  best mean L/D = {-es.best.f:.1f}")
    return es
