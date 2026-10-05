"""(μ + λ) evolution strategy with Gaussian mutation only and the 1/5 success rule.

Each parent makes λ/μ children, the best μ of parents + children survive, and one global
step size σ grows when more than 1/5 of children beat their parent and shrinks otherwise.
"""

import numpy as np

from ..config import ES
from ..evaluator import Evaluator


def run_mutation_es(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    mu, lam, sigma = ES["mu"], ES["lambda"], ES["sigma0"]
    rng = np.random.default_rng(seed)
    parent_of_child = np.repeat(np.arange(mu), lam // mu)

    parents = np.vstack([x0, x0 + sigma * rng.standard_normal((mu - 1, len(x0)))])
    parent_costs = evaluate(parents)

    while not evaluate.exhausted and sigma > ES["min_sigma"]:
        children = parents[parent_of_child] + sigma * rng.standard_normal((lam, len(x0)))
        child_costs = evaluate(children)

        success_rate = np.mean(child_costs < parent_costs[parent_of_child])
        sigma *= ES["sigma_factor"] if success_rate > 0.2 else 1 / ES["sigma_factor"]

        pool = np.vstack([parents, children])
        pool_costs = np.concatenate([parent_costs, child_costs])
        survivors = np.argsort(pool_costs)[:mu]
        parents, parent_costs = pool[survivors], pool_costs[survivors]
