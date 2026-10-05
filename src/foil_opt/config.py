"""Problem settings, constraint limits and optimizer options."""

from pathlib import Path

import numpy as np

# Design conditions: minimize mean CD over every (CL, Re) pair
TARGET_CLS = np.array([0.5, 0.8])
REYNOLDS = np.array([3e5, 1e6])
MODEL_SIZE = "large"

# Angle-of-attack solve for each target CL (secant iterations)
ALPHA_SECANT_STEPS = 3
CL_TOLERANCE = 0.01

# Stall check: CL max over this alpha sweep at the lowest Reynolds number
STALL_ALPHAS = np.array([10.0, 12.0, 14.0, 16.0])

# Constraint limits
MIN_THICKNESS = 0.12
MIN_AFT_THICKNESS = 0.015  # at X_AFT
MAX_CAMBER = 0.04
MIN_LE_RADIUS = 0.007
MIN_TE_ANGLE = 8.0  # degrees
MIN_CM = -0.1
MIN_CONFIDENCE = 0.9
MIN_CL_MAX = 1.2

X_CHECK = np.linspace(0.01, 0.99, 50)
X_AFT = 0.9

# Optimizers
BASELINE_NAME = "naca2412"
CMA = {"sigma0": 0.03, "popsize": 24}
DE = {"half_width": 1.0, "popsize": 2, "mutation": (0.5, 1.0), "recombination": 0.7}
ES = {"mu": 6, "lambda": 24, "sigma0": 0.03, "sigma_factor": 1.2, "min_sigma": 1e-6}
SA_ES = {"mu": 6, "lambda": 24, "sigma0": 0.03, "stall_generations": 60}
GRADIENT = {"max_iter": 500, "margin": 0.01}  # margin: fraction of each limit kept as safety
NSGA2 = {"pop_size": 60, "generations": 200, "sigma0": 0.02, "init_spread": 0.03}

# Experiments
DEFAULT_BUDGET = 10_000
DEFAULT_SEEDS = 5
LOG_EVERY = 1000  # evaluations

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
RESULTS_DIR = ROOT / "results"
