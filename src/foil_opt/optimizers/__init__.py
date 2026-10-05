"""Single-objective optimizers. Each runs until the evaluator's budget is spent or it converges."""

from collections.abc import Callable

import numpy as np

from ..evaluator import Evaluator
from .cma_es import run_cma
from .differential_evolution import run_de
from .gradient import run_gradient
from .mutation_es import run_mutation_es
from .self_adaptive_es import run_self_adaptive_es

Optimizer = Callable[[Evaluator, np.ndarray, int], None]

OPTIMIZERS: dict[str, tuple[str, Optimizer]] = {
    "cma": ("CMA-ES", run_cma),
    "de": ("Differential Evolution", run_de),
    "es": ("(μ+λ)-ES, 1/5 rule", run_mutation_es),
    "sa-es": ("Self-adaptive ES", run_self_adaptive_es),
    "grad": ("Gradient (IPOPT)", run_gradient),
}
