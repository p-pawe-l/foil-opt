"""Text output: per-design tables and the optimizer comparison table."""

import numpy as np

from .constraints import build_constraints
from .experiments import RunResult
from .geometry import to_airfoil
from .problem import evaluate, operating_points


def print_design(name: str, x: np.ndarray) -> None:
    e = evaluate([x])
    af = to_airfoil(x)
    a = {key: value[0] for key, value in e.aero.items()}
    status = "feasible" if e.feasible[0] else f"INFEASIBLE (violation {e.violation[0]:.3f})"
    print(f"\n{name}: mean CD = {e.objective[0]:.5f}, mean L/D = {e.mean_ld[0]:.1f}, {status}")
    print(f"  thickness {af.max_thickness():.3f}  camber {af.max_camber():.3f}  "
          f"LE radius {af.LE_radius():.4f}  TE angle {af.TE_angle():.1f}°  CL max {a['CL_stall'].max():.2f}")
    print("    CL      Re   alpha       CD    L/D      CM  conf")
    for i, re in enumerate(operating_points()[1]):
        print(f"  {a['CL_target'][i]:4.2f}  {re:6.0e}  {a['alpha'][i]:5.2f}°  {a['CD'][i]:.5f}  "
              f"{a['CL'][i] / a['CD'][i]:5.1f}  {a['CM'][i]:+.3f}  {a['analysis_confidence'][i]:.2f}")
    violated = [name for name, v in e.constraint_violations.items() if v[0] > 0]
    if violated:
        print("  violated:", ", ".join(violated))
    print("  active constraints:", ", ".join(active_constraints(x)) or "none")


def comparison_table(results: dict[str, list[RunResult]], names: dict[str, str]) -> str:
    rows = [
        "| Optimizer | Runs | Evaluations | Time | Best mean CD | Median mean CD | Worst mean CD | Best mean L/D |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, runs in results.items():
        costs = np.array([r.cost for r in runs])
        best = runs[int(np.argmin(costs))]
        ld = evaluate([best.x]).mean_ld[0]
        evals = "n/a" if key == "grad" else f"{np.median([r.evals for r in runs]):,.0f}"
        rows.append(
            f"| {names[key]} | {len(runs)} | {evals} | "
            f"{np.median([r.seconds for r in runs]):.1f} s | {costs.min():.5f} | {np.median(costs):.5f} | "
            f"{costs.max():.5f} | {ld:.1f} |"
        )
    return "\n".join(rows)


def active_constraints(x: np.ndarray, margin: float = 0.02) -> list[str]:
    """Constraints within `margin` (normalized) of their limit."""
    e = evaluate([x])
    af = to_airfoil(x)
    point = {key: value[0] for key, value in e.aero.items()}
    active = []
    for con in build_constraints():
        value = np.asarray(con.value(af, point))
        near_lower = con.lower is not None and (value.min() - con.lower) / con.scale < margin
        near_upper = con.upper is not None and (con.upper - value.max()) / con.scale < margin
        if near_lower or near_upper:
            active.append(con.name)
    return active
