"""Airfoil parametrization: mapping between Kulfan (CST) airfoils and flat genome vectors.

Genome layout (17 values): [8 upper weights | 8 lower weights | leading-edge weight]
"""

import aerosandbox as asb
import numpy as np

from .config import BASELINE_NAME

N_WEIGHTS = 8
UPPER = slice(0, N_WEIGHTS)
LOWER = slice(N_WEIGHTS, 2 * N_WEIGHTS)
LE = 2 * N_WEIGHTS

baseline = asb.KulfanAirfoil(BASELINE_NAME)
TE_THICKNESS = baseline.TE_thickness  # kept fixed during optimization


def to_genome(af: asb.KulfanAirfoil) -> np.ndarray:
    return np.concatenate([af.upper_weights, af.lower_weights, [af.leading_edge_weight]])


def to_airfoil(x: np.ndarray) -> asb.KulfanAirfoil:
    return asb.KulfanAirfoil(
        upper_weights=x[UPPER],
        lower_weights=x[LOWER],
        leading_edge_weight=x[LE],
        TE_thickness=TE_THICKNESS,
    )
