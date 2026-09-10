import os
from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NAMES = ["plot_P1D", "plot_P3D", "plot_cosmo", "plot_fisher", "plot_noise", "plot_qsoLF", "plot_bias"]


@pytest.mark.examples
@pytest.mark.parametrize("name", NAMES)
def test_example_script(name, make_config, save_config, tmp_path):
    pytest.importorskip("matplotlib")
    config = save_config(make_config(reduced=True))
    output = tmp_path / "figures"
    env = dict(os.environ, MPLBACKEND="Agg")
    process = subprocess.run([sys.executable, str(EXAMPLES / "scripts" / f"{name}.py"),
                              "-i", str(config), "-o", str(output)],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert list(output.glob("*.png"))
    assert all(p.stat().st_size > 0 for p in output.glob("*.png"))


@pytest.mark.examples
@pytest.mark.parametrize("name", NAMES)
def test_notebook_execution(name, make_config, save_config, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from jupyter_client import KernelManager

    config = save_config(make_config(reduced=True))
    monkeypatch.setenv("LYAFORECAST_EXAMPLE_CONFIG", str(config))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for variable in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR"):
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
    path = EXAMPLES / "notebooks" / f"{name}.ipynb"
    original = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
    # Use the interpreter running pytest, not an unrelated user kernel.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = nbclient.NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    assert path.read_bytes() == original
    nbformat.write(notebook, tmp_path / path.name)
