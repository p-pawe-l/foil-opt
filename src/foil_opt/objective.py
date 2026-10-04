"""Cost function: negative mean L/D plus constraint penalties (CMA-ES minimizes)."""

import numpy as np

from .aero import batch_aero
from .config import (
    CONFIDENCE_PENALTY,
    CROSSING_PENALTY,
    MIN_CONFIDENCE,
    MIN_THICKNESS,
    THICKNESS_PENALTY,
    X_CHECK,
)
from .geometry import to_airfoil


def penalty(x: np.ndarray, conf: np.ndarray) -> float:
    t = to_airfoil(x).local_thickness(X_CHECK)
    p = 0.0
    p += THICKNESS_PENALTY * max(0.0, MIN_THICKNESS - t.max())  # too thin
    p += CROSSING_PENALTY * np.sum(np.maximum(0.0, -t))  # surfaces cross
    p += CONFIDENCE_PENALTY * np.sum(np.maximum(0.0, MIN_CONFIDENCE - conf))  # untrusted prediction
    return p


def evaluate(population: list[np.ndarray]) -> list[float]:
    """Return a cost per candidate."""
    pop = np.array(population)
    ld, conf = batch_aero(pop)
    return [-ld[i].mean() + penalty(x, conf[i]) for i, x in enumerate(pop)]
