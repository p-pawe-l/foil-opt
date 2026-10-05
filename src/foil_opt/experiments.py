"""Running optimizers, storing results as JSON and loading them back."""

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .evaluator import Evaluator
from .geometry import baseline_genome
from .optimizers import OPTIMIZERS


@dataclass
class RunResult:
    optimizer: str
    seed: int
    budget: int
    evals: int
    seconds: float
    genome: list[float]
    cost: float
    history: list[tuple[int, float]]
    settings: dict | None = None  # the merged YAML settings of the run

    @property
    def x(self) -> np.ndarray:
        return np.array(self.genome)

    @property
    def feasible(self) -> bool:
        return self.cost < 1


def run(key: str, seed: int, budget: int, verbose: bool = False) -> RunResult:
    evaluator = Evaluator(budget, verbose)
    start = time.perf_counter()
    OPTIMIZERS[key][1](evaluator, baseline_genome(), seed)
    return RunResult(
        optimizer=key,
        seed=seed,
        budget=budget,
        evals=evaluator.evals,
        seconds=time.perf_counter() - start,
        genome=evaluator.best_x.tolist(),
        cost=evaluator.best_cost,
        history=evaluator.history,
        settings=config.settings.source,
    )


def save(result: RunResult) -> None:
    results = config.settings.paths.results
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{result.optimizer}_seed{result.seed}.json"
    path.write_text(json.dumps(asdict(result)))


def load(key: str) -> list[RunResult]:
    paths = sorted(config.settings.paths.results.glob(f"{key}_seed*.json"))
    return [RunResult(**json.loads(p.read_text())) for p in paths]


def best(results: list[RunResult]) -> RunResult:
    return min(results, key=lambda r: r.cost)
