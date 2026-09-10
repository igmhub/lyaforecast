import configparser
from pathlib import Path

import pytest

from lyaforecast.utils import get_file


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "examples/desi2").glob("*.ini"))


@pytest.fixture
def make_config(tmp_path):
    """Copy a canonical INI; only smoke tests request smaller numerical grids."""
    def make(name="lya_qso_2x2pt.ini", reduced=False):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(ROOT / "examples/desi2" / name)
        config["output"]["filename"] = str(tmp_path / name.removesuffix(".ini") / "output")
        if reduced:
            config["survey"]["num z bins"] = "2"
            config["survey"]["z bin min"] = "2.3"
            config["survey"]["z bin max"] = "2.7"
            config["survey"]["num mag bins"] = "24"
            config["power spectrum"]["num_k_bins"] = "64"
            config["power spectrum"]["num_mu_bins"] = "4"
            # Only the smoke fixture reduces CAMB's integration range. The
            # packaged Planck18.ini and full numerical baselines stay unchanged.
            camb_config = tmp_path / "smoke-camb.ini"
            camb_config.write_text(get_file("Planck18.ini").read_text().replace(
                "transfer_kmax = 1152.5", "transfer_kmax = 2.0"))
            config["cosmo"]["filename"] = str(camb_config)
        return config
    return make


@pytest.fixture
def save_config(tmp_path):
    def save(config):
        path = tmp_path / "forecast.ini"
        with path.open("w") as stream:
            config.write(stream)
        return path
    return save
