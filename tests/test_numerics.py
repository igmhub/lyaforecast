from types import SimpleNamespace

import numpy as np
import pytest

from lyaforecast.analytic_p1d_PD2013 import P1D_z_kms_PD2013
from lyaforecast.fisher import Fisher
from lyaforecast.power_spectrum import PowerSpectrum
from lyaforecast.tracer import Tracer
from lyaforecast.weights import Weights


@pytest.mark.parametrize("z,expected", [
    (2., [13.958281526612275, 13.958281526612275, 13.958281526612275,
          8.045533032895229, 1.4548300995570178]),
    (3., [47.5907489173414, 47.5907489173414, 47.5907489173414,
          22.34021442552742, 3.327422904064485]),
    (4., [126.01210455963869, 126.01210455963869, 126.01210455963869,
          49.33076875448149, 6.321198328980145]),
])
def test_p1d_matches_pre_cleanup_reference(z, expected):
    k = np.array([0., 1e-8, 1e-4, .009, .1])
    np.testing.assert_allclose(P1D_z_kms_PD2013(z, k), expected, rtol=1e-13)
    np.testing.assert_array_equal(PowerSpectrum.compute_p1d_palanque2013(None, z, k),
                                  P1D_z_kms_PD2013(z, k))
    for ki, value in zip(k, expected):
        assert P1D_z_kms_PD2013(z, ki) == pytest.approx(value, rel=1e-13)


@pytest.fixture
def fisher_case():
    k = np.linspace(.01, .5, 64)
    power = SimpleNamespace(k=k, logk=np.log(k), dlogk=(k[1]-k[0])/k,
                            mu=np.array([.125, .375, .625, .875]))
    cosmo = SimpleNamespace(growth_factor_ratios=np.array([.9, 1.1]),
                            growth_rate_zbins=np.array([.95, .97]), growth_rate=.96)
    fisher = Fisher(power, cosmo, 1000*k**2, zbin_index=0, reconstruction_factor=2)
    p = 100*np.exp(-k)*(1+.05*np.sin(100*k))
    models = {"lya_lya": np.tile(p, (4, 1)), "lya_qso": np.tile(-.6*p, (4, 1)),
              "qso_qso": np.tile(2*p, (4, 1))}
    measurements = {name: values.copy() for name, values in models.items()}
    measurements["lya_lya"] += 20
    measurements["qso_qso"] += 30
    return fisher, models, measurements


@pytest.mark.parametrize("names,expected", [
    (["lya_qso"], [[31.758372064006114, 44.87443813299788],
                    [44.87443813299788, 368.7608929915612]]),
    (["lya_lya", "lya_qso", "qso_qso"],
     [[504.1771545823611, 575.6191152043224], [575.6191152043224, 3779.4027935892436]]),
])
def test_fisher_matches_pre_cleanup_reference(fisher_case, names, expected):
    fisher, models, measurements = fisher_case
    matrix = fisher.compute_fisher({name: models[name] for name in names}, measurements, names)
    np.testing.assert_allclose(matrix, expected, rtol=1e-9)


def test_gaussian_covariance_and_reversed_pairs(fisher_case):
    fisher, _, measurements = fisher_case
    powers = {name: values[0] for name, values in measurements.items()}
    labels = ["lya_lya", "lya_qso", "qso_qso"]
    covariance, index = fisher.gaussian_covariance_array_func(powers, labels)
    np.testing.assert_array_equal(covariance, covariance.swapaxes(0, 1))
    np.testing.assert_allclose(covariance[1, 1],
                              (powers["lya_lya"]*powers["qso_qso"] + powers["lya_qso"]**2)
                              / fisher._num_modes)
    np.testing.assert_array_equal(fisher.p_entry_from_label(powers, "qso", "lya"),
                                  powers["lya_qso"])
    assert index == {name: i for i, name in enumerate(labels)}
    assert np.all(np.linalg.eigvalsh(np.moveaxis(covariance, -1, 0)) > 0)


def test_reconstruction_only_changes_discrete_derivatives(fisher_case):
    fisher, models, _ = fisher_case
    rows = np.stack([values[0] for values in models.values()])
    before = fisher.compute_derivatives(rows, .6, list(models))
    fisher._reconstruction_factor = 1
    after = fisher.compute_derivatives(rows, .6, list(models))
    np.testing.assert_array_equal(before[:2], after[:2])
    assert not np.allclose(before[2], after[2])


def test_continuous_and_discrete_normalization_remain_distinct(make_config, tmp_path):
    z = np.array([1.5, 2., 2.5, 3.])
    m = np.array([19., 20., 21., 22.])
    table = tmp_path / "density.txt"
    np.savetxt(table, np.column_stack([np.repeat(z, len(m)), np.tile(m, len(z)), np.ones(16)]))
    config = make_config()
    for section in ("tracer 1", "tracer 2"):
        config[section]["dn dz"] = str(table)
        config[section]["target density"] = "80"
        config[section]["min_band_mag"] = "20"
        config[section]["max_band_mag"] = "21"
    continuous = Tracer(config["tracer 1"])
    qso = Tracer(config["tracer 2"])
    config["tracer 2"]["tracer"] = "lbg"
    lbg = Tracer(config["tracer 2"])
    zz, mm = np.meshgrid(z, m, indexing="ij")
    cell = (z[1]-z[0])*(m[1]-m[0])
    # Continuous selection applies magnitude cuts before normalization above
    # z=2.15; discrete QSO normalizes above 2.15 without those cuts; LBG uses all z.
    counts = [t.get_dn_dzdm(zz.ravel(), mm.ravel()).reshape(4, 4)*cell
              for t in (continuous, qso, lbg)]
    np.testing.assert_allclose(counts[0][:, 1:3], 20.)
    np.testing.assert_allclose(counts[0][:, [0, 3]], 0., atol=1e-12)
    np.testing.assert_allclose(counts[1], 10.)
    np.testing.assert_allclose(counts[2], 5.)


def test_noise_assignment_accepts_size_one_interpolator_arrays():
    weights = Weights.__new__(Weights)
    weights._lya_tracer = SimpleNamespace(num_exp=4)
    weights._pix_kms = 1.
    weights._cosmo = SimpleNamespace(velocity_from_wavelength=lambda z: 1.)
    weights._z_bin, weights._zq, weights._lambda_mean = 2.4, 2.7, 4000.
    weights.maglist = np.array([20., 21.])
    weights._spectrograph = SimpleNamespace(get_pixel_rms_noise=lambda *args: np.array([2.]))
    np.testing.assert_array_equal(weights._get_pix_var_m(), [4., 4.])
