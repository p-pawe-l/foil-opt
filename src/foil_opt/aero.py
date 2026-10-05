"""Batched NeuralFoil calls for a whole population."""

import neuralfoil as nf
import numpy as np

from .config import ALPHA_SECANT_STEPS, MODEL_SIZE
from .geometry import LE, LOWER, TE_THICKNESS, UPPER

LIFT_SLOPE_GUESS = 0.1  # per degree
ZERO_LIFT_ALPHA_GUESS = -2.0


def analyze(pop: np.ndarray, alpha: np.ndarray, re: np.ndarray) -> dict[str, np.ndarray]:
    """Aero coefficients for every candidate at every (alpha, Re) column; arrays are (n, k)."""
    n, k = alpha.shape
    rows = np.repeat(pop, k, axis=0)
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(
            upper_weights=rows[:, UPPER].T,
            lower_weights=rows[:, LOWER].T,
            leading_edge_weight=rows[:, LE],
            TE_thickness=np.full(len(rows), TE_THICKNESS),
        ),
        alpha=alpha.ravel(),
        Re=re.ravel(),
        model_size=MODEL_SIZE,
    )
    return {key: np.asarray(value).reshape(n, k) for key, value in aero.items()}


def analyze_at_cl(pop: np.ndarray, target_cl: np.ndarray, re: np.ndarray) -> dict[str, np.ndarray]:
    """Find alpha giving each target CL with a few secant steps; result includes "alpha"."""
    a0 = np.broadcast_to(target_cl / LIFT_SLOPE_GUESS + ZERO_LIFT_ALPHA_GUESS, re.shape).copy()
    a1 = a0 + 1.0
    cl0 = analyze(pop, a0, re)["CL"]
    aero = analyze(pop, a1, re)
    for _ in range(ALPHA_SECANT_STEPS):
        slope = np.clip((aero["CL"] - cl0) / (a1 - a0), 0.03, 0.3)
        a0, cl0 = a1, aero["CL"]
        a1 = a1 + (target_cl - aero["CL"]) / slope
        aero = analyze(pop, a1, re)
    return aero | {"alpha": a1}
