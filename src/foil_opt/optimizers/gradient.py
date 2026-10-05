"""Gradient-based optimization with IPOPT through aerosandbox.Opti.

NeuralFoil and the constraint functions are differentiable, so the problem is solved directly:
the angles of attack become variables, with CL = target as equality constraints.
"""

import aerosandbox as asb
import aerosandbox.numpy as np
import casadi

from ..config import GRADIENT, MODEL_SIZE, REYNOLDS, STALL_ALPHAS
from ..constraints import without
from ..evaluator import Evaluator
from ..geometry import to_airfoil
from ..problem import CL_GRID, RE_GRID

try:
    casadi.GlobalOptions.setNumpyMode(-1)  # silence casadi's numpy-compatibility notice
except AttributeError:
    pass


def run_gradient(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    opti = asb.Opti()
    x = opti.variable(init_guess=x0)
    alpha = opti.variable(init_guess=CL_GRID / 0.1 - 2)
    af = to_airfoil(x)

    aero = af.get_aero_from_neuralfoil(alpha=alpha, Re=RE_GRID, model_size=MODEL_SIZE)
    aero["CL_stall"] = af.get_aero_from_neuralfoil(alpha=STALL_ALPHAS, Re=REYNOLDS.min(), model_size=MODEL_SIZE)["CL"]
    opti.subject_to(aero["CL"] == CL_GRID)

    margin = GRADIENT["margin"]  # IPOPT meets limits only to ~1e-3, so aim slightly inside them
    for con in without("CL error"):
        value = con.value(af, aero)
        if con.lower is not None:
            opti.subject_to(value >= con.lower + margin * con.scale)
        if con.upper is not None:
            opti.subject_to(value <= con.upper - margin * con.scale)

    opti.minimize(np.mean(aero["CD"]))
    sol = opti.solve(max_iter=GRADIENT["max_iter"], verbose=False, behavior_on_failure="return_last")
    evaluate([sol(x)])
