"""Design constraints, written with aerosandbox.numpy so they work numerically and symbolically.

The value functions are defined here; which constraints apply and their limits come from the settings.
A value may be a scalar or a vector: numerically the worst element counts, in the gradient
optimizer every element is constrained.
"""

from collections.abc import Callable
from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np

from . import config


@dataclass(frozen=True)
class Constraint:
    name: str
    value: Callable[[asb.KulfanAirfoil, dict], object]
    lower: float | None = None
    upper: float | None = None
    scale: float = 1.0

    def violation(self, value) -> float:
        """Normalized amount by which the constraint is broken, 0 when satisfied."""
        value = np.asarray(value)
        v = 0.0
        if self.lower is not None:
            v = max(v, (self.lower - value.min()) / self.scale)
        if self.upper is not None:
            v = max(v, (value.max() - self.upper) / self.scale)
        return max(v, 0.0)


def le_radius(af: asb.KulfanAirfoil):
    """aerosandbox's LE_radius, but 0 for a pinched nose (where the library raises)."""
    if np.is_casadi_type(af.upper_weights):
        return af.LE_radius()
    if af.upper_weights[0] <= 0 or af.lower_weights[0] >= 0:
        return 0.0
    return af.LE_radius()


def _x_check():
    return config.settings.problem.x_check


VALUES: dict[str, Callable[[asb.KulfanAirfoil, dict], object]] = {
    "max_thickness": lambda af, a: np.max(af.local_thickness(_x_check())),
    "no_crossing": lambda af, a: af.local_thickness(_x_check()),
    "aft_thickness": lambda af, a: af.local_thickness(config.settings.problem.x_aft),
    "max_camber": lambda af, a: af.local_camber(_x_check()),
    "le_radius": lambda af, a: le_radius(af),
    "te_angle": lambda af, a: af.TE_angle(),
    "cm": lambda af, a: a["CM"],
    "confidence": lambda af, a: a["analysis_confidence"],
    "cl_max": lambda af, a: np.max(a["CL_stall"]),
    "cl_error": lambda af, a: np.abs(a["CL"] - a["CL_target"]),
}


def build_constraints(exclude: tuple[str, ...] = ()) -> list[Constraint]:
    """The constraints enabled in the settings, minus `exclude`."""
    unknown = set(config.settings.constraints) - set(VALUES)
    if unknown:
        raise ValueError(f"unknown constraint(s) {sorted(unknown)}, available: {sorted(VALUES)}")
    return [
        Constraint(name, VALUES[name], limits.lower, limits.upper, limits.scale)
        for name, limits in config.settings.constraints.items()
        if name not in exclude
    ]
