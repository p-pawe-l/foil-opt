"""Problem settings: flow conditions, constraints and optimizer options."""

import numpy as np

# Flow conditions
RE = 5e5
ALPHAS = np.array([2.0, 4.0, 6.0])
MODEL_SIZE = "large"

# Constraints
MIN_THICKNESS = 0.12
MIN_CONFIDENCE = 0.9
X_CHECK = np.linspace(0.01, 0.99, 50)  # chord stations where thickness is checked

# Penalty weights
THICKNESS_PENALTY = 1e3
CROSSING_PENALTY = 1e3
CONFIDENCE_PENALTY = 1e2

# CMA-ES
SIGMA0 = 0.03
CMA_OPTIONS = {"popsize": 24, "maxiter": 2000, "seed": 1, "verbose": -9}
LOG_EVERY = 10

# Output
BASELINE_NAME = "naca2412"
PLOT_PATH = "optimized_airfoil.png"
