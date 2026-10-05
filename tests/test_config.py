import numpy as np
import pytest

from foil_opt import config
from foil_opt.constraints import build_constraints
from foil_opt.problem import operating_points


@pytest.fixture(autouse=True)
def default_settings():
    yield
    config.use(None)


def write(tmp_path, text):
    path = tmp_path / "settings.yaml"
    path.write_text(text)
    return path


def test_defaults_load():
    s = config.load()
    assert s.problem.baseline == "naca2412"
    assert s.constraints["cm"].lower == -0.1
    assert len(s.problem.x_check) == 50


def test_override_merges_over_defaults(tmp_path):
    config.use(write(tmp_path, "problem: {reynolds: [3e5, 2e6]}\noptimizers: {cma: {popsize: 12}}\n"))
    s = config.settings
    np.testing.assert_array_equal(s.problem.reynolds, [3e5, 2e6])
    assert s.problem.target_cls.tolist() == [0.5, 0.8]
    assert s.optimizers.cma.popsize == 12 and s.optimizers.cma.sigma0 == 0.03
    assert operating_points()[1].tolist() == [3e5, 3e5, 2e6, 2e6]


def test_null_disables_constraint(tmp_path):
    config.use(write(tmp_path, "constraints: {cm: null}\n"))
    assert "cm" not in [c.name for c in build_constraints()]


def test_scale_defaults_to_limit_magnitude():
    assert config.settings.constraints["max_thickness"].scale == 0.12


@pytest.mark.parametrize("text, message", [
    ("optimizers: {cma: {sigma: 0.1}}\n", "unknown key"),
    ("constraints: {cm: {lower: null}}\n", "lower"),
])
def test_invalid_settings_are_rejected(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        config.load(write(tmp_path, text))


def test_unknown_constraint_is_rejected(tmp_path):
    config.use(write(tmp_path, "constraints: {wingspan: {lower: 1}}\n"))
    with pytest.raises(ValueError, match="unknown constraint"):
        build_constraints()
