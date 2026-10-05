"""Budgeted cost function shared by all single-objective optimizers.

Counts evaluations, records the best feasible design and its convergence history.
"""

import numpy as np

from . import config
from .problem import evaluate


class Evaluator:
    def __init__(self, budget: int, verbose: bool = False):
        self.budget = budget
        self.verbose = verbose
        self.evals = 0
        self.best_x: np.ndarray | None = None
        self.best_cost = np.inf
        self.history: list[tuple[int, float]] = []  # (evaluations, best cost so far)
        self._next_log = config.settings.experiments.log_every

    @property
    def exhausted(self) -> bool:
        return self.evals >= self.budget

    def __call__(self, pop) -> np.ndarray:
        pop = np.atleast_2d(np.asarray(pop, dtype=float))
        cost = evaluate(pop).cost
        self.evals += len(pop)

        i = int(np.argmin(cost))
        if cost[i] < self.best_cost:
            self.best_cost, self.best_x = float(cost[i]), pop[i].copy()
        self.history.append((self.evals, self.best_cost))

        if self.verbose and self.evals >= self._next_log:
            print(f"evals {self.evals:6d}  best {describe(self.best_cost)}")
            self._next_log += config.settings.experiments.log_every
        return cost


def describe(cost: float) -> str:
    if cost < 1:
        return f"mean CD = {cost:.5f}"
    return f"infeasible, violation = {cost - 1:.3f}"
