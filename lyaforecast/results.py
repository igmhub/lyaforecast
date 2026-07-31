"""Class to handle results for each config. Includes BAO errors as a function of redshift, magnitude etc.."""


class Results:
    """Container for BAO forecast results per redshift and magnitude bin."""

    def __init__(self, survey):
        """
        Parameters
        ----------
        survey : Survey
            Survey instance used to determine array sizes and redshift bins.
        """
        per_mmax_results = dict(lya_auto = [], cross = [], tracer_auto = [])
        per_z_results = dict(lya_auto = [], cross = [], tracer_auto = [])
        pass

    def _log_results(self):
        """Log forecast results to the configured logger."""
        pass

    def _write_results(self):
        """Write forecast results to the output file."""
        pass
