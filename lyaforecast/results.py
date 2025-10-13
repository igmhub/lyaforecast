"""Class to handle results for each config. Includes BAO errors as a function of redshift, magnitude etc.."""

class Results:
    def __init__(self, survey):
        #load option flags.

        #Use config to define array sizes
        # survey.maglist
        # survey.num_z_bins
        # survey.num_mag_bins
        # survey.z_bin_centres

        per_mmax_results = dict(lya_auto = [], cross = [], tracer_auto = [])
        per_z_results = dict(lya_auto = [], cross = [], tracer_auto = [])

        #write or log or both
        pass


    #print results
    def _log_results(self):
        pass

    #write results to file

    def _write_results(self):
        pass


