"""Installed historical templates remain available for external adaptation."""
import configparser
from pathlib import Path

import pytest

from lyaforecast.utils import get_dir, get_file


TEMPLATES = (
    "DESI_2_lbg_x_lae.ini",
    "DESI_2_lya_lbg_x_lae.ini",
    "DESI_2_lya_lbg_x_lbg.ini",
    "DESI_2_lya_qso_x_lae.ini",
    "DESI_2_lya_qso_x_lbg.ini",
    "DESI_2_lya_qso_x_qso.ini",
    "DESI_2_lya_x_lbg_x_lae.ini",
    "DESI_lya_qso_x_qso.ini",
)


def test_installed_template_inventory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = get_dir("default_configs")
    assert {path.name for path in directory.glob("*.ini")} == set(TEMPLATES)


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_lookup_inputs_and_portable_output(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = get_file(name)
    assert path.parent == get_dir("default_configs")
    config = configparser.ConfigParser()
    config.optionxform = str
    assert config.read(path)
    output = Path(config["output"]["filename"])
    assert not output.is_absolute()
    assert output == Path("outputs") / Path(name).stem / "lyacast_test.dat"
    for section in config.sections():
        for key, value in config[section].items():
            if key == "snr-file-dir":
                assert get_dir(value).is_dir()
            elif key == "dn dz" or (section == "cosmo" and key == "filename"):
                assert get_file(value).is_file()
    # Resolving a template must never create output directories or run a forecast.
    assert not output.parent.exists()
