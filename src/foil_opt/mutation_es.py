"""Hand-written (mu + lambda) evolution strategy: Gaussian mutation only, no crossover.

Each generation:
  1. every parent produces lambda / mu children:  child = parent + sigma * N(0, I)
  2. all children are scored in one batched NeuralFoil call
  3. the best mu of parents + children survive (elitist: the best is never lost)
  4. sigma follows the 1/5 success rule: grow it if more than 1/5 of the children
     beat their own parent, shrink it otherwise
"""

from collections.abc import Callable

import numpy as np

from .config import ES_OPTIONS
from .objective import evaluate


def optimize_mutation(x0: np.ndarray, log: Callable[[int, float], None]) -> tuple[np.ndarray, float]:
    mu, lam = ES_OPTIONS["mu"], ES_OPTIONS["lambda"]
    sigma = ES_OPTIONS["sigma0"]
    rng = np.random.default_rng(ES_OPTIONS["seed"])
    parent_of_child = np.repeat(np.arange(mu), lam // mu)

    # Initial parents: x0 plus mutated copies of it
    parents = np.vstack([x0, x0 + sigma * rng.standard_normal((mu - 1, len(x0)))])
    parent_costs = np.array(evaluate(list(parents)))

    for iteration in range(1, ES_OPTIONS["maxiter"] + 1):
        # 1. Mutation
        children = parents[parent_of_child] + sigma * rng.standard_normal((len(parent_of_child), len(x0)))

        # 2. Evaluation
        child_costs = np.array(evaluate(list(children)))

        # 4. 1/5 success rule (measured against each child's own parent, before selection)
        success_rate = np.mean(child_costs < parent_costs[parent_of_child])
        sigma *= ES_OPTIONS["sigma_factor"] if success_rate > 0.2 else 1 / ES_OPTIONS["sigma_factor"]

        # 3. (mu + lambda) selection
        pool = np.vstack([parents, children])
        pool_costs = np.concatenate([parent_costs, child_costs])
        survivors = np.argsort(pool_costs)[:mu]
        parents, parent_costs = pool[survivors], pool_costs[survivors]

        log(iteration, parent_costs[0])
        if sigma < ES_OPTIONS["min_sigma"]:
            break

    return parents[0], parent_costs[0]
