import os
import subprocess
import sys

import numpy as np
import pytest

from lyaforecast.forecast_new import NewForecast


@pytest.mark.integration
@pytest.mark.parametrize("name", [
    "lbg_lae_3x2pt.ini", "lya_lbg_lae_3x2pt.ini", "lya_lbg_lae_6x2pt.ini",
    "lya_qso_2x2pt.ini", "lya_qso_lbg_lae_15x2pt.ini", "lya_qso_lbg_lae_4x2pt.ini",
    "lya_qso_lbg_lae_8x2pt.ini",
])
def test_reduced_real_forecasts(name, make_config, save_config):
    forecast = NewForecast(save_config(make_config(name, reduced=True)))
    data = forecast.new_run_forecast()
    assert data["redshifts"].shape == (2,)
    for key in ("sigma_ap", "sigma_at"):
        assert np.all(np.isfinite(data[key])) and np.all(data[key] > 0)
    assert np.all(np.abs(data["corr_coef"]) <= 1)
    for name in forecast.correlations_to_compute:
        assert np.all(data[name]["sigma_ap"] > 0)
        assert np.all(data[name]["sigma_at"] > 0)


@pytest.mark.integration
def test_cli_forecast_outside_checkout(make_config, save_config, tmp_path):
    path = save_config(make_config(reduced=True))
    outside = tmp_path / "outside"
    outside.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-m", "lyaforecast.scripts.run_bao_forecast", "-i", str(path)],
                   cwd=outside, env=env, check=True, capture_output=True, text=True)
