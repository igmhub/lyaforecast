import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lyaforecast.scripts.run_bao_forecast import get_args
from lyaforecast.utils import get_dir, get_file


@pytest.mark.parametrize("name,digest", [
    ("dn_dzdr_qso_desi_2.dat", "78914d3a9ad0307ad94072d1b39a54b41d4043ebca5a0e4d1d4d202a3d489c8d"),
    ("lbg_matched_dndzdr.txt", "995e1b06e09735e4babdcdfb6f0d253371c9d108b06c5826eb92a9774defcdef"),
    ("lae_matched_dndzdr.txt", "f56f7d3ba3144ae0fdfaaeba93c557d337e74f8de993110133aaaba3d7f1302e"),
])
def test_exact_bundled_inputs(name, digest):
    assert hashlib.sha256(get_file(name).read_bytes()).hexdigest() == digest


def test_resource_lookup_and_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_file("Planck18.ini").is_file()
    assert len(list(get_dir("DESI-2-QSO").glob("*.dat"))) == 12
    local = tmp_path / "Planck18.ini"
    local.write_text("local file wins")
    assert get_file("Planck18.ini") == Path("Planck18.ini")
    monkeypatch.setenv("FORECAST_TEST_DATA", str(tmp_path))
    assert get_file("$FORECAST_TEST_DATA/Planck18.ini") == local
    assert get_dir("$FORECAST_TEST_DATA") == tmp_path
    with pytest.raises(RuntimeError):
        get_file("does-not-exist.ini")


def test_cli_requires_one_config():
    assert get_args(["-i", "one.ini"]).configs == "one.ini"
    assert get_args(["--configs", "one.ini"]).configs == "one.ini"
    for args in ([], ["-i", "one.ini", "two.ini"]):
        with pytest.raises(SystemExit) as error:
            get_args(args)
        assert error.value.code == 2


def test_import_and_help_from_outside_checkout(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", "from lyaforecast.forecast_new import NewForecast; "
                    "from lyaforecast.utils import get_file; assert get_file('Planck18.ini').is_file()"],
                   cwd=tmp_path, env=env, check=True)
    subprocess.run([sys.executable, "-m", "lyaforecast.scripts.run_bao_forecast", "--help"],
                   cwd=tmp_path, env=env, check=True)
