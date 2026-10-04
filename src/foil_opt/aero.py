"""Batched NeuralFoil evaluation of a whole population in a single call."""

import neuralfoil as nf
import numpy as np

from .config import ALPHAS, MODEL_SIZE, RE
from .geometry import LE, LOWER, TE_THICKNESS, UPPER


def batch_aero(pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (L/D, confidence) arrays of shape (n_candidates, n_alphas)."""
    n, n_a = len(pop), len(ALPHAS)

    # One row per (candidate, alpha) pair
    rep = np.repeat(pop, n_a, axis=0)
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(
            upper_weights=rep[:, UPPER].T,
            lower_weights=rep[:, LOWER].T,
            leading_edge_weight=rep[:, LE],
            TE_thickness=np.full(len(rep), TE_THICKNESS),
        ),
        alpha=np.tile(ALPHAS, n),
        Re=RE,
        model_size=MODEL_SIZE,
    )
    ld = (aero["CL"] / aero["CD"]).reshape(n, n_a)
    conf = aero["analysis_confidence"].reshape(n, n_a)
    return ld, conf
