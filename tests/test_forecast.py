import logging
from types import SimpleNamespace

import numpy as np
import pytest

import lyaforecast.forecast_new as module
from lyaforecast.fisher import Fisher
from lyaforecast.survey import Survey
from lyaforecast.utils import setup_logger


@pytest.fixture
def lightweight_forecast(monkeypatch):
    """Exercise real configuration, tracer loading, and orchestration, sans CAMB."""
    calls = []

    def cosmo(filename, z_ref, z_centres):
        return SimpleNamespace(z_ref=z_ref, z_bins=np.asarray(z_centres),
                               growth_rate_zbins=np.ones(len(z_centres)), growth_rate=1.)

    class Covariance:
        def __init__(self, *args):
            self.corr = args[5]
            self.num_modes = np.ones(500)
            self.pix_width_kms = self.pix_res_kms = 1.

        def __call__(self, *args):
            pass

        def compute_eff_density_and_noise(self, corr):
            calls.append(("noise", corr))

        def compute_total_power(self, k, mu, corr):
            return np.ones_like(k)

    class TestFisher:
        def __init__(self, *args, **kwargs):
            pass

        def compute_fisher(self, models, measurements, names):
            calls.append(("fisher", list(names), set(measurements)))
            # Every covariance entry needs both tracer orderings to resolve.
            for first in names:
                for second in names:
                    for a in first.split("_"):
                        for b in second.split("_"):
                            assert f"{a}_{b}" in measurements or f"{b}_{a}" in measurements
            assert list(models) == names
            return np.eye(2)*len(names)

        print_bao = staticmethod(Fisher.print_bao)

    monkeypatch.setattr(module, "CosmoCamb", cosmo)
    monkeypatch.setattr(module, "Spectrograph", lambda *args: None)
    monkeypatch.setattr(module, "Covariance", Covariance)
    monkeypatch.setattr(module, "Fisher", TestFisher)
    monkeypatch.setattr(module.PowerSpectrum, "compute_p3d_hmpc_smooth",
                        lambda self, z, k, *args: np.ones_like(k))
    return calls


@pytest.mark.parametrize("name,count", [
    ("lbg_lae_3x2pt.ini", 3), ("lya_lbg_lae_3x2pt.ini", 3),
    ("lya_lbg_lae_6x2pt.ini", 6), ("lya_qso_2x2pt.ini", 2),
    ("lya_qso_lbg_lae_15x2pt.ini", 15), ("lya_qso_lbg_lae_4x2pt.ini", 4),
    ("lya_qso_lbg_lae_8x2pt.ini", 8),
])
def test_all_canonical_selections_and_result_contract(name, count, make_config,
                                                     save_config, lightweight_forecast):
    forecast = module.NewForecast(save_config(make_config(name)))
    data = forecast.new_run_forecast()
    assert len(forecast.correlations_to_compute) == count
    assert set(data) == set(forecast.correlations) | {
        "redshifts", "zedges", "fiducial redshift", "sigma_at", "sigma_ap", "corr_coef"}
    assert data["zedges"].shape == (forecast.survey.num_z_bins + 1,)
    for corr in forecast.correlations:
        assert set(data[corr]) == {"sigma_at", "sigma_ap", "corr_coef"}
        for values in data[corr].values():
            assert values.shape == data["redshifts"].shape
        if corr not in forecast.correlations_to_compute:
            for values in data[corr].values():
                np.testing.assert_array_equal(values, 0)
    calculated = {call[1] for call in lightweight_forecast if call[0] == "noise"}
    assert calculated == set(forecast.correlations)
    np.testing.assert_allclose(data["sigma_ap"], 1/np.sqrt(count))


def test_reversed_cross_selection(make_config, save_config, lightweight_forecast):
    config = make_config()
    config["control"]["correlations"] = "qso_lya(qso)"
    forecast = module.NewForecast(save_config(config))
    assert forecast.correlations_to_compute == ["lya(qso)_qso"]
    forecast.new_run_forecast()


def test_custom_lya_and_discrete_bias(make_config, save_config, lightweight_forecast):
    config = make_config()
    for section, bias in (("tracer 1", "-0.2 -0.3"), ("tracer 2", "2 4")):
        config[section]["bias z"] = "2 3"
        config[section]["bias val"] = bias
    forecast = module.NewForecast(save_config(config))
    assert forecast.power_spectrum.bias._get_density_bias(2.5, "lya") == pytest.approx(-.25)
    assert forecast.power_spectrum.bias._get_density_bias(2.5, "qso") == pytest.approx(3.)


def test_explicit_redshift_centres_and_edges(make_config):
    config = make_config()
    config["survey"]["z bin centres"] = "2.2,2.4 2.6"
    survey = Survey(config)
    np.testing.assert_allclose(survey.z_bin_centres, [2.2, 2.4, 2.6])
    np.testing.assert_allclose(survey.z_bin_edges, [[2.1, 2.3, 2.5], [2.3, 2.5, 2.7]])
    del config["survey"]["z bin centres"]
    config["survey"]["z bin edges"] = "2.1,2.3 2.5,2.7"
    np.testing.assert_allclose(Survey(config).z_bin_edges, survey.z_bin_edges)


def test_logger_reuse_directory_creation_and_root_isolation(tmp_path, capsys):
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    directory = tmp_path / "new" / "directory"
    first = setup_logger(directory)
    second = setup_logger(directory)
    assert first is second
    assert len(second.handlers) == 2
    assert root.handlers == handlers and root.level == level
    first.info("unique marker")
    assert (directory / "forecast.log").read_text().count("unique marker") == 1
    assert capsys.readouterr().err.count("unique marker") == 1
    other = setup_logger(tmp_path / "other")
    other.info("other marker")
    assert "other marker" not in (directory / "forecast.log").read_text()


def test_repeated_forecast_initialization(make_config, save_config, lightweight_forecast):
    path = save_config(make_config())
    first, second = module.NewForecast(path), module.NewForecast(path)
    assert first.logger is second.logger
    assert len(first.logger.handlers) == 2
