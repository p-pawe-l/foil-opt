"""Gradient-based optimization with IPOPT through aerosandbox.Opti.

NeuralFoil and the constraint functions are differentiable, so the problem is solved directly:
the angles of attack become variables, with CL = target as equality constraints.
"""

import aerosandbox as asb
import aerosandbox.numpy as np
import casadi

from .. import config
from ..constraints import build_constraints
from ..evaluator import Evaluator
from ..geometry import to_airfoil
from ..problem import operating_points

try:
    casadi.GlobalOptions.setNumpyMode(-1)  # silence casadi's numpy-compatibility notice
except AttributeError:
    pass


def run_gradient(evaluate: Evaluator, x0: np.ndarray, seed: int) -> None:
    p, s = config.settings.problem, config.settings.optimizers.gradient
    cl, re = operating_points()
    opti = asb.Opti()
    x = opti.variable(init_guess=x0)
    alpha = opti.variable(init_guess=cl / 0.1 - 2)
    af = to_airfoil(x)

    aero = af.get_aero_from_neuralfoil(alpha=alpha, Re=re, model_size=p.model_size)
    stall = af.get_aero_from_neuralfoil(alpha=p.stall_alphas, Re=p.reynolds.min(), model_size=p.model_size)
    aero["CL_stall"] = stall["CL"]
    opti.subject_to(aero["CL"] == cl)

    margin = s.margin  # IPOPT meets limits only to ~1e-3, so aim slightly inside them
    for con in build_constraints(exclude=("cl_error",)):
        value = con.value(af, aero)
        if con.lower is not None:
            opti.subject_to(value >= con.lower + margin * con.scale)
        if con.upper is not None:
            opti.subject_to(value <= con.upper - margin * con.scale)

    opti.minimize(np.mean(aero["CD"]))
    sol = opti.solve(max_iter=s.max_iter, verbose=False, behavior_on_failure="return_last")
    evaluate([sol(x)])
