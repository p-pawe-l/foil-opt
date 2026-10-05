"""Genome layout: [8 upper weights | 8 lower weights | leading-edge weight]."""

import aerosandbox as asb
import numpy as np

from .config import BASELINE_NAME

N_WEIGHTS = 8
N_GENES = 2 * N_WEIGHTS + 1
UPPER = slice(0, N_WEIGHTS)
LOWER = slice(N_WEIGHTS, 2 * N_WEIGHTS)
LE = 2 * N_WEIGHTS

baseline = asb.KulfanAirfoil(BASELINE_NAME)
TE_THICKNESS = baseline.TE_thickness


def to_genome(af: asb.KulfanAirfoil) -> np.ndarray:
    return np.concatenate([af.upper_weights, af.lower_weights, [af.leading_edge_weight]])


def to_airfoil(x) -> asb.KulfanAirfoil:
    return asb.KulfanAirfoil(
        upper_weights=x[UPPER],
        lower_weights=x[LOWER],
        leading_edge_weight=x[LE],
        TE_thickness=TE_THICKNESS,
    )


def baseline_genome() -> np.ndarray:
    return to_genome(baseline)
