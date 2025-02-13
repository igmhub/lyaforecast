"""Plot power spectra and parameter error bars as functions of survey properties."""
class Plots:
    def __init__(self,config) -> None:
        self.plot_distances = config['control'].getboolean('plot distances')

        #We will require in some cases that n_redshift_bins > 1.
        n_redshift_bins = config['power spectrum'].getint('num z bins')

    def plot_da_h_z(self):
        pass