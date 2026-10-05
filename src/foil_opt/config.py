"""Settings from YAML: the package's default.yaml, optionally overridden by a user file.

Modules read `config.settings` when they run, so `config.use(path)` takes effect everywhere.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

DEFAULT_FILE = Path(__file__).with_name("default.yaml")


@dataclass
class Problem:
    baseline: str
    target_cls: np.ndarray
    reynolds: np.ndarray
    model_size: str
    alpha_secant_steps: int
    stall_alphas: np.ndarray
    x_check: np.ndarray
    x_aft: float

    def __post_init__(self):
        self.target_cls = np.asarray(self.target_cls, dtype=float)
        self.reynolds = np.asarray(self.reynolds, dtype=float)
        self.stall_alphas = np.asarray(self.stall_alphas, dtype=float)
        self.x_check = np.linspace(float(self.x_check["start"]), float(self.x_check["stop"]), int(self.x_check["num"]))


@dataclass
class Limits:
    lower: float | None = None
    upper: float | None = None
    scale: float | None = None

    def __post_init__(self):
        if self.lower is None and self.upper is None:
            raise ValueError("a constraint needs `lower` and/or `upper`")
        if self.scale is None:
            self.scale = abs(self.lower if self.lower is not None else self.upper) or 1.0


@dataclass
class CMA:
    sigma0: float
    popsize: int


@dataclass
class DE:
    half_width: float
    popsize: int
    mutation: tuple[float, float]
    recombination: float


@dataclass
class ES:
    parents: int
    children: int
    sigma0: float
    sigma_factor: float
    min_sigma: float


@dataclass
class SelfAdaptiveES:
    parents: int
    children: int
    sigma0: float
    stall_generations: int


@dataclass
class Gradient:
    max_iter: int
    margin: float


@dataclass
class NSGA2:
    pop_size: int
    generations: int
    sigma0: float
    init_spread: float


@dataclass
class Optimizers:
    cma: CMA
    de: DE
    es: ES
    sa_es: SelfAdaptiveES
    gradient: Gradient
    nsga2: NSGA2


@dataclass
class Experiments:
    budget: int
    seeds: int
    log_every: int


@dataclass
class Paths:
    assets: Path
    results: Path


@dataclass
class Settings:
    problem: Problem
    constraints: dict[str, Limits]
    optimizers: Optimizers
    experiments: Experiments
    paths: Paths
    source: dict  # the merged YAML, saved with every run


def load(path: str | Path | None = None) -> Settings:
    data = read(DEFAULT_FILE)
    if path is not None:
        data = merge(data, read(Path(path)))
    try:
        return Settings(
            problem=build(Problem, data["problem"], "problem"),
            constraints={
                name: build(Limits, limits, f"constraints.{name}")
                for name, limits in data["constraints"].items()
                if limits is not None
            },
            optimizers=Optimizers(**{
                f.name: build(f.type, data["optimizers"][f.name], f"optimizers.{f.name}") for f in fields(Optimizers)
            }),
            experiments=build(Experiments, data["experiments"], "experiments"),
            paths=Paths(**{key: Path(value) for key, value in data["paths"].items()}),
            source=data,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid settings{f' in {path}' if path else ''}: {error}") from None


def build(cls, values: dict, where: str):
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in {where}, expected {sorted(names)}")
    return cls(**{f.name: cast(f.type, values[f.name]) for f in fields(cls) if f.name in values})


def cast(type_, value):
    """Convert to the field's type; YAML reads numbers such as 3e5 as strings."""
    if value is None:
        return None
    if type_ in (float, float | None):
        return float(value)
    if type_ is int:
        return int(float(value))
    if getattr(type_, "__origin__", None) is tuple:
        return tuple(float(v) for v in value)
    return value


def read(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = merge(base[key], value)
        else:
            merged[key] = value
    return merged


settings = load()


def use(path: str | Path | None) -> None:
    global settings
    settings = load(path)
