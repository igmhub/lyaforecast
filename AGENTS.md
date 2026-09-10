# Repository Guidelines

## Structure and Supported Workflow

`lyaforecast/` contains the scientific package. The supported workflow is
`NewForecast(path).new_run_forecast()` from `lyaforecast.forecast_new`.
`examples/desi2/` contains the seven authoritative INIs. Bundled density, CAMB,
and spectrograph inputs live in `lyaforecast/resources/`; resolve them through
`lyaforecast.utils.get_file` or `get_dir`. Diagnostic notebooks and scripts share
presentation code in `examples/scripts/_diagnostics.py`. Keep numerical formulas
in package modules, not examples. Metadata and tool settings live in `pyproject.toml`.

The eight files in `resources/default_configs/` within the package are retained
installed templates for external users. Preserve their names, scientific
settings, input choices, and legacy schemas. Only their output paths and adjacent
comments have been modernized. Do not remove or migrate them without explicit
user instructions; `examples/desi2` remains the validated runnable reference.

## Running and Checking Changes

Development and testing workloads, including all seven full DESI-2 forecasts,
are small enough to run on NERSC login nodes. Do not request compute nodes or
Slurm allocations for these tasks. Use `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` for local validation.

- Install: `python -m pip install -e ".[dev,plot,notebook]"` (Python 3.11+).
- Forecast: `run-bao-forecast -i examples/desi2/lya_qso_2x2pt.ini`.
- Tests: `python -m pytest`; use `-m "not examples"` for core checks.
- Lint: `ruff check .`; avoid unrelated reformatting or numerical rewrites.
- Distributions: `python -m build` and `python -m twine check dist/*`.

## Numerical and Test Requirements

Before changing scientific code, run all seven authoritative configurations and
save their original dictionaries, effective INIs, logs, input hashes, versions,
Git state, and timings under `.validation/baseline/<timestamp>/`. Resolve
environment problems first. If baseline execution requires a source fix, discuss
it with the user before changing code. Compare all outputs after changes in the
same environment with `rtol=1e-10`, `atol=1e-12`; investigate deviations.

Preserve vectorized NumPy operations, tracer normalization differences, result
keys, and unselected spectra needed for covariance. Use focused pytest tests with
deterministic arrays, bundled inputs, and temporary output directories. Reduced
smoke settings must never modify authoritative files. Clear notebook outputs.

## Contributions

Use four-space indentation, snake_case functions, PascalCase classes, and
NumPy-style public docstrings. Document data provenance without inventing missing
settings. Never commit generated outputs or personal paths. Keep commits focused;
PRs should explain the scientific effect and list exact validation commands.
