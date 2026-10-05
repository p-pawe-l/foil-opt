"""Design constraints, written with aerosandbox.numpy so they work numerically and symbolically.

Each value function takes one airfoil and its aero dict and may return a scalar or a vector;
numerically the worst element counts, in the gradient optimizer every element is constrained.
"""

from collections.abc import Callable
from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np

from . import config as c


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


CONSTRAINTS = [
    Constraint("max thickness", lambda af, a: np.max(af.local_thickness(c.X_CHECK)),
               lower=c.MIN_THICKNESS, scale=c.MIN_THICKNESS),
    Constraint("no crossing", lambda af, a: af.local_thickness(c.X_CHECK), lower=0.0, scale=c.MIN_THICKNESS),
    Constraint("aft thickness", lambda af, a: af.local_thickness(c.X_AFT),
               lower=c.MIN_AFT_THICKNESS, scale=c.MIN_AFT_THICKNESS),
    Constraint("max camber", lambda af, a: af.local_camber(c.X_CHECK), upper=c.MAX_CAMBER, scale=c.MAX_CAMBER),
    Constraint("LE radius", lambda af, a: le_radius(af), lower=c.MIN_LE_RADIUS, scale=c.MIN_LE_RADIUS),
    Constraint("TE angle", lambda af, a: af.TE_angle(), lower=c.MIN_TE_ANGLE, scale=c.MIN_TE_ANGLE),
    Constraint("CM", lambda af, a: a["CM"], lower=c.MIN_CM, scale=abs(c.MIN_CM)),
    Constraint("confidence", lambda af, a: a["analysis_confidence"], lower=c.MIN_CONFIDENCE),
    Constraint("CL max", lambda af, a: np.max(a["CL_stall"]), lower=c.MIN_CL_MAX, scale=c.MIN_CL_MAX),
    Constraint("CL error", lambda af, a: np.abs(a["CL"] - a["CL_target"]), upper=c.CL_TOLERANCE, scale=c.CL_TOLERANCE),
]


def without(*names: str) -> list[Constraint]:
    return [con for con in CONSTRAINTS if con.name not in names]
