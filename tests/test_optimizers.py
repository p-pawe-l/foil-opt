import pytest

from foil_opt.evaluator import Evaluator
from foil_opt.geometry import baseline_genome
from foil_opt.optimizers import OPTIMIZERS
from foil_opt.problem import evaluate


@pytest.mark.parametrize("key", [k for k in OPTIMIZERS if k != "grad"])
def test_short_run_stays_within_budget_and_beats_or_keeps_baseline(key):
    x0 = baseline_genome()
    evaluator = Evaluator(budget=500)
    OPTIMIZERS[key][1](evaluator, x0, 1)
    assert evaluator.evals <= 500 + 34  # may finish the generation it started
    assert evaluator.best_cost <= evaluate([x0]).cost[0]


def test_evaluator_tracks_best_and_history():
    x0 = baseline_genome()
    evaluator = Evaluator(budget=10)
    evaluator([x0, x0 * 0.5])
    assert evaluator.evals == 2
    assert evaluator.best_cost == pytest.approx(evaluate([x0]).cost[0])
    assert evaluator.history == [(2, evaluator.best_cost)]
