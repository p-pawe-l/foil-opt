import numpy as np
import pytest

from foil_opt.constraints import Constraint, le_radius
from foil_opt.geometry import baseline_genome, to_airfoil
from foil_opt.problem import CL_GRID, INFEASIBLE_OFFSET, evaluate


@pytest.fixture(scope="module")
def x0():
    return baseline_genome()


def test_baseline_is_feasible(x0):
    e = evaluate([x0])
    assert e.feasible[0]
    assert all(v[0] == 0 for v in e.constraint_violations.values())


def test_alpha_solve_hits_target_cl(x0):
    e = evaluate([x0])
    np.testing.assert_allclose(e.aero["CL"][0], CL_GRID, atol=0.01)


def test_batch_matches_single_evaluation(x0):
    pop = x0 + 0.02 * np.random.default_rng(0).standard_normal((5, len(x0)))
    batch = evaluate(pop)
    for i, x in enumerate(pop):
        single = evaluate([x])
        np.testing.assert_allclose(batch.objective[i], single.objective[0], rtol=1e-10)
        np.testing.assert_allclose(batch.violation[i], single.violation[0], rtol=1e-10)


def test_infeasible_costs_more_than_feasible(x0):
    thin = x0.copy()
    thin[:8] *= 0.5  # halve the upper surface: far too thin
    e = evaluate([x0, thin])
    assert e.feasible[0] and not e.feasible[1]
    assert e.cost[1] > INFEASIBLE_OFFSET > e.cost[0]


def test_constraint_violation_is_normalized():
    con = Constraint("t", lambda af, a: None, lower=0.1, upper=0.3, scale=0.1)
    assert con.violation(0.2) == 0
    assert con.violation(np.array([0.05, 0.2])) == pytest.approx(0.5)
    assert con.violation(0.35) == pytest.approx(0.5)


def test_pinched_nose_has_zero_le_radius(x0):
    pinched = x0.copy()
    pinched[0], pinched[8] = -0.1, 0.1
    assert le_radius(to_airfoil(pinched)) == 0
