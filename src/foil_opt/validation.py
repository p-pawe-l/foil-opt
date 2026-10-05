"""Check NeuralFoil's predictions for a design against XFoil at the same target CLs."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import aerosandbox as asb
import numpy as np

from .problem import evaluate, operating_points


@dataclass
class Comparison:
    cl: np.ndarray
    re: np.ndarray
    cd_neuralfoil: np.ndarray
    cd_xfoil: np.ndarray
    cm_neuralfoil: np.ndarray
    cm_xfoil: np.ndarray


def find_xfoil() -> str | None:
    """The XFOIL environment variable, or `xfoil` on the PATH."""
    return os.environ.get("XFOIL") or shutil.which("xfoil")


def run_xfoil(af: asb.Airfoil, re: float, cls: np.ndarray, xfoil: str) -> dict[str, np.ndarray]:
    """Viscous XFoil polar at the given CLs; NaN where XFoil did not converge."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        af.write_dat(tmp / "airfoil.dat")
        commands = [
            "PLOP", "G F", "",
            "LOAD airfoil.dat",
            "PANE",
            "OPER", f"VISC {re:.0f}", "ITER 200",
            "PACC", "polar.txt", "",
            *(f"CL {cl}" for cl in cls),
            "", "QUIT",
        ]
        env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}  # keep XFoil headless
        subprocess.run([xfoil], input="\n".join(commands), cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
        rows = parse_polar(tmp / "polar.txt")

    result = {key: np.full(len(cls), np.nan) for key in ("alpha", "CL", "CD", "CM")}
    for alpha, cl, cd, _cdp, cm in rows:
        i = np.argmin(np.abs(cls - cl))
        if abs(cls[i] - cl) < 0.01:
            result["alpha"][i], result["CL"][i], result["CD"][i], result["CM"][i] = alpha, cl, cd, cm
    return result


def parse_polar(path: Path) -> list[tuple[float, ...]]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("---")), len(lines))
    return [tuple(float(v) for v in line.split()[:5]) for line in lines[start + 1 :] if line.strip()]


def validate(x: np.ndarray, af: asb.KulfanAirfoil, xfoil: str) -> Comparison:
    cl, re = operating_points()
    nf = evaluate([x]).aero
    xf = {key: np.full(len(cl), np.nan) for key in ("CD", "CM")}
    for reynolds in np.unique(re):
        at_re = re == reynolds
        result = run_xfoil(af.to_airfoil(), reynolds, cl[at_re], xfoil)
        xf["CD"][at_re], xf["CM"][at_re] = result["CD"], result["CM"]
    return Comparison(cl, re, nf["CD"][0], xf["CD"], nf["CM"][0], xf["CM"])
