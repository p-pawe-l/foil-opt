import numpy as np

from foil_opt.geometry import N_GENES, baseline, baseline_genome, to_airfoil, to_genome


def test_genome_round_trip():
    x = baseline_genome()
    assert x.shape == (N_GENES,)
    np.testing.assert_allclose(to_genome(to_airfoil(x)), x)


def test_airfoil_from_genome_matches_baseline_shape():
    xs = np.linspace(0.05, 0.95, 10)
    np.testing.assert_allclose(to_airfoil(baseline_genome()).local_thickness(xs), baseline().local_thickness(xs))
