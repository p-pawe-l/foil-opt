"""Multi-objective search with NSGA-II (leap-ec): mean CD versus pitching-moment magnitude.

The CM limit becomes the second objective, every other constraint still applies.
"""

import random
from dataclasses import dataclass

import numpy as np
from leap_ec import ops
from leap_ec.multiobjective.nsga2 import generalized_nsga_2
from leap_ec.multiobjective.problems import MultiObjectiveProblem
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation

from .config import NSGA2
from .constraints import without
from .problem import INFEASIBLE_OFFSET, evaluate

CONSTRAINTS = without("CM")


class AirfoilProblem(MultiObjectiveProblem):
    def __init__(self):
        super().__init__(maximize=(False, False))

    def evaluate(self, phenome):
        return self.evaluate_multiple([phenome])[0]

    def evaluate_multiple(self, phenomes, individuals=None):
        # leap-ec clones keep their parent's cached phenome, so read the (mutated) genomes instead
        genomes = [ind.genome for ind in individuals] if individuals else phenomes
        return list(objectives(np.array(genomes)))


def objectives(pop: np.ndarray) -> np.ndarray:
    """(mean CD, max |CM|) per candidate; infeasible candidates get both set from their violation."""
    e = evaluate(pop, CONSTRAINTS)
    f = np.column_stack([e.objective, np.abs(e.aero["CM"]).max(axis=1)])
    f[~e.feasible] = INFEASIBLE_OFFSET + e.violation[~e.feasible, None]
    return f


@dataclass
class Front:
    genomes: np.ndarray
    mean_cd: np.ndarray
    max_abs_cm: np.ndarray


def run_nsga2(x0: np.ndarray, seed: int) -> Front:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    pop_size = NSGA2["pop_size"]

    final = generalized_nsga_2(
        max_generations=NSGA2["generations"],
        pop_size=pop_size,
        problem=AirfoilProblem(),
        representation=Representation(initialize=lambda: x0 + NSGA2["init_spread"] * rng.standard_normal(len(x0))),
        pipeline=[
            ops.tournament_selection,
            ops.clone,
            ops.UniformCrossover(p_swap=0.2),
            mutate_gaussian(std=NSGA2["sigma0"], expected_num_mutations="isotropic"),
            ops.pool(size=pop_size),
            ops.grouped_evaluate,
        ],
        init_evaluate=ops.grouped_evaluate,
    )

    genomes = np.array([ind.genome for ind in final])
    f = np.array([ind.fitness for ind in final])
    keep = non_dominated(f) & (f[:, 0] < INFEASIBLE_OFFSET)
    order = np.argsort(f[keep, 0])
    return Front(genomes[keep][order], f[keep, 0][order], f[keep, 1][order])


def non_dominated(f: np.ndarray) -> np.ndarray:
    dominated = np.array([np.any(np.all(f <= fi, axis=1) & np.any(f < fi, axis=1)) for fi in f])
    return ~dominated
