"""Problem settings: flow conditions, constraints and optimizer options."""

import numpy as np

# Flow conditions
RE = 5e5
ALPHAS = np.array([2.0, 4.0, 6.0])
MODEL_SIZE = "large"

# Constraints
MIN_THICKNESS = 0.12
MAX_CAMBER = 0.04  # NACA 2412 has 0.02
MIN_CONFIDENCE = 0.9
X_CHECK = np.linspace(0.01, 0.99, 50)  # chord stations where thickness and camber are checked

# Penalty weights
THICKNESS_PENALTY = 1e3
CROSSING_PENALTY = 1e3
CAMBER_PENALTY = 1e4  # strong: extra camber buys a lot of L/D
CONFIDENCE_PENALTY = 1e2

# CMA-ES
SIGMA0 = 0.03
CMA_OPTIONS = {"popsize": 24, "maxiter": 2000, "seed": 1, "verbose": -9}

# Differential Evolution
DE_HALF_WIDTH = 1.0  # search box is baseline genome +/- this
DE_OPTIONS = {"popsize": 2, "maxiter": 2000, "mutation": (0.5, 1.0), "recombination": 0.7, "rng": 1}

LOG_EVERY = 10

# Output
BASELINE_NAME = "naca2412"
