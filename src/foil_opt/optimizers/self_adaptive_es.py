"""(μ, λ) evolution strategy with one self-adapted step size per gene, plus restarts.

Every child inherits its parent's step sizes, perturbs them log-normally, then mutates with them:
good step sizes survive along with the good designs they produced. When the best design stops
improving for a while, the search restarts around it with fresh step sizes.
"""

import numpy as np

from ..config import SA_ES
from ..evaluator import Evaluator


def run_self_adaptive_es(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    mu, lam, n = SA_ES["mu"], SA_ES["lambda"], len(x0)
    tau_global, tau_local = 1 / np.sqrt(2 * n), 1 / np.sqrt(2 * np.sqrt(n))
    rng = np.random.default_rng(seed)

    center = x0
    while not evaluate.exhausted:
        parents = np.tile(center, (mu, 1))
        sigmas = np.full((mu, n), SA_ES["sigma0"])
        best, stalled = np.inf, 0

        while not evaluate.exhausted and stalled < SA_ES["stall_generations"]:
            idx = rng.integers(mu, size=lam)
            child_sigmas = sigmas[idx] * np.exp(
                tau_global * rng.standard_normal((lam, 1)) + tau_local * rng.standard_normal((lam, n))
            )
            children = parents[idx] + child_sigmas * rng.standard_normal((lam, n))
            costs = evaluate(children)

            survivors = np.argsort(costs)[:mu]
            parents, sigmas = children[survivors], child_sigmas[survivors]
            if costs[survivors[0]] < best:
                best, stalled = costs[survivors[0]], 0
            else:
                stalled += 1

        center = evaluate.best_x
