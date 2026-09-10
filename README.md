# lyaforecast

Forecast BAO constraints from Lyman-alpha forests and discrete tracers, including
QSO, LBG, and LAE auto- and cross-correlations. The supported entry point is
`NewForecast`; the old `Forecast` implementations have been removed. Their
configuration templates remain available for external users to copy and adapt.

## Installation

Python 3.11 or newer is required. From a checkout:

```bash
python -m pip install .
# Editable development installation, including tests and examples:
python -m pip install -e ".[dev,plot,notebook]"
```

NumPy, SciPy, and CAMB are runtime dependencies. Matplotlib and notebook tools are
optional; neither Astropy nor mcfit is required. Package data are included in
wheels, so installed forecasts do not depend on a source-tree working directory.

## Run a forecast

```python
from lyaforecast.forecast_new import NewForecast

forecast = NewForecast("examples/desi2/lya_qso_2x2pt.ini")
data = forecast.new_run_forecast()
print(data["redshifts"])
print(data["sigma_ap"], data["sigma_at"])
```

The equivalent command is:

```bash
run-bao-forecast -i examples/desi2/lya_qso_2x2pt.ini
```

Pass exactly one configuration. The seven INIs in [examples/desi2](examples/desi2)
are the authoritative examples; their scientific settings should not be changed
as part of maintenance:

| Configuration | Selected correlations |
| --- | --- |
| `lbg_lae_3x2pt.ini` | LBG and LAE auto-correlations and their cross-correlation |
| `lya_lbg_lae_3x2pt.ini` | LBG-background Lyα auto and its crosses with LBG and LAE |
| `lya_lbg_lae_6x2pt.ini` | All pairs of those three tracers |
| `lya_qso_2x2pt.ini` | QSO-background Lyα auto and its cross with QSO |
| `lya_qso_lbg_lae_4x2pt.ini` | QSO-background Lyα auto and crosses with QSO, LBG, LAE |
| `lya_qso_lbg_lae_8x2pt.ini` | Both Lyα autos and their crosses with QSO, LBG, LAE |
| `lya_qso_lbg_lae_15x2pt.ini` | All pairs of both Lyα fields, QSO, LBG, and LAE |

`[tracer N]` sections define tracers; `[control] correlations` selects pairs or
`all`. Cross-pair names may use either ordering. Spectra not selected for the
Fisher result may still be required for its covariance and are computed.

The exact density inputs are bundled. Their provenance and checksums are in
[the resource notes](lyaforecast/resources/data/README.md). Input paths support
environment-variable expansion and existing local paths, followed by package
resource lookup. Relative local paths are relative to the current working
directory, not the INI directory. The example output paths are likewise relative
to the working directory; choose a writable destination when copying an INI.

`[output] filename` determines the directory containing `forecast.log`, which is
created if necessary. The Python method returns results in memory; it does not
write a forecast results file or automatically plot. The existing `overwrite`
and `plot bao` entries do not control serialization or plotting in this workflow.
Repeated forecasts sharing a directory append to its log without adding duplicate
handlers or changing application root logging.

The returned dictionary contains:

- `redshifts`, `zedges`, and `fiducial redshift`;
- one dictionary per correlation containing `sigma_ap`, `sigma_at`, and
  `corr_coef` arrays; unselected correlations retain zero-filled arrays;
- top-level `sigma_ap`, `sigma_at`, and `corr_coef` arrays for the combined result.

There is no `total` result key. The current power model uses Kaiser bias terms;
`linear power` remains accepted for compatibility and does not enable a separate
nonlinear model. Scientific models, numerical grids, density normalization,
BAO peak extraction, and reconstruction damping are unchanged by modernization.

## Installed configuration templates

The eight INIs in `lyaforecast/resources/default_configs/` are retained as
installed templates for external users. They are included in both wheels and
source distributions and can be located by filename:

```python
from pathlib import Path
from shutil import copyfile
from lyaforecast.utils import get_file

copyfile(get_file("DESI_2_lya_qso_x_qso.ini"), Path("my_forecast.ini"))
```

Edit the copied file, leaving the installed resource unchanged. These templates
preserve their original scientific settings, input choices, section names, and
comments, except for portable output paths and their adjacent comments. Outputs
use `outputs/<template-stem>/lyacast_test.dat`, relative to the current working
directory. Their existing input paths already resolve to bundled resources.

Some templates use the legacy `[lya forest]` / `[tracer]` schema and require
adaptation to `[tracer N]` sections and the current controls before use with
`NewForecast`. Updating paths does not migrate that schema or restore the old
`Forecast` API. Use `examples/desi2` as the validated runnable reference for the
current workflow; templates are starting points, not equivalent replacements for
those scientific configurations. No template is loaded automatically.

## Diagnostic examples

Install the `plot` and `notebook` extras. Seven scripts and seven notebooks cover
P1D, P3D/BAO decomposition, CAMB cosmology, forecasts, pixel noise, tracer density,
and bias models. Scripts and notebooks share presentation helpers while using
package methods for scientific calculations.

```bash
python examples/scripts/plot_fisher.py -i examples/desi2/lya_qso_2x2pt.ini -o outputs/diagnostics
python examples/scripts/plot_P1D.py -o outputs/p1d
```

Run notebooks from `examples/notebooks` with the installed package available in
the kernel. `LYAFORECAST_EXAMPLE_CONFIG` can select another INI (use an absolute
path). Notebooks use temporary output directories and are committed without
outputs. The previous standalone CAMB scratch notebook has been merged into
`plot_cosmo`; obsolete g-band and unsupported nonlinear comparisons were removed.

## Development and validation

These small development workloads, including all seven full DESI-2 examples,
are suitable for NERSC login nodes. No compute-node allocation or Slurm job is
needed. Limit numerical-library threads for repeatable local comparisons:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -m "not examples"
python -m pytest -m examples
ruff check .
python -m build
python -m twine check dist/*
```

`python -m pytest` runs the complete suite; example checks need the optional
extras. Unit tests use deterministic inputs. Integration/example tests use
smaller grids and a temporary reduced CAMB integration range; packaged settings
are unchanged. CI tests Python 3.11–3.14, builds distributions, tests an installed
wheel outside the checkout, and executes the examples on Python 3.13.

Before future numerical cleanup, run all seven full INIs with exact inputs and
save outputs before editing source. Keep local baseline dictionaries, effective
configurations, logs, input hashes, Git state, dependency versions, and timings
under `.validation/baseline/<timestamp>/`. Compare every result in the same
environment with `rtol=1e-10`, `atol=1e-12`; investigate differences instead of
silently relaxing tolerances. Baseline pickles are trusted local artifacts and
are not distribution files. Timing comparisons are diagnostic because login-node
load varies. Do not commit baselines, generated forecasts, logs, or figures.
