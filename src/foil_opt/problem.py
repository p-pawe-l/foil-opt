"""The design problem: minimize mean CD at the target CLs, subject to the constraints."""

from dataclasses import dataclass, field

import numpy as np

from . import config
from .aero import analyze, analyze_at_cl
from .constraints import CONSTRAINTS, Constraint
from .geometry import to_airfoil

CL_GRID, RE_GRID = (g.ravel() for g in np.meshgrid(config.TARGET_CLS, config.REYNOLDS))
INFEASIBLE_OFFSET = 1.0  # above any mean CD, so every feasible design beats every infeasible one


@dataclass
class Evaluation:
    objective: np.ndarray  # (n,) mean CD
    violation: np.ndarray  # (n,) sum of normalized constraint violations
    aero: dict[str, np.ndarray]  # (n, n_points) arrays
    constraint_violations: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def feasible(self) -> np.ndarray:
        return self.violation == 0

    @property
    def cost(self) -> np.ndarray:
        """Feasibility rule as one number: feasible by objective, infeasible by violation."""
        return np.where(self.feasible, self.objective, INFEASIBLE_OFFSET + self.violation)

    @property
    def mean_ld(self) -> np.ndarray:
        return (self.aero["CL"] / self.aero["CD"]).mean(axis=1)


def evaluate(pop, constraints: list[Constraint] = CONSTRAINTS) -> Evaluation:
    pop = np.atleast_2d(np.asarray(pop, dtype=float))
    n = len(pop)
    aero = analyze_at_cl(pop, np.tile(CL_GRID, (n, 1)), np.tile(RE_GRID, (n, 1)))
    aero["CL_target"] = np.tile(CL_GRID, (n, 1))
    stall_alpha = np.tile(config.STALL_ALPHAS, (n, 1))
    aero["CL_stall"] = analyze(pop, stall_alpha, np.full_like(stall_alpha, config.REYNOLDS.min()))["CL"]

    per_constraint = {con.name: np.zeros(n) for con in constraints}
    for i, x in enumerate(pop):
        af = to_airfoil(x)
        point = {key: value[i] for key, value in aero.items()}
        for con in constraints:
            per_constraint[con.name][i] = con.violation(con.value(af, point))

    return Evaluation(
        objective=aero["CD"].mean(axis=1),
        violation=sum(per_constraint.values()),
        aero=aero,
        constraint_violations=per_constraint,
    )
