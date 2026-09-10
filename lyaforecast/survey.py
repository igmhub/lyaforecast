""" In this module we store all of the survey specifications used in the forecast,
       including the quasar luminosity function."""
import numpy as np


class Survey:
    """Survey specifications including area, redshift bins, and magnitude grid."""

    BAND_OPTIONS = ['r']
    TRACER_OPTIONS = ['qso', 'lbg', 'lae']

    def __init__(self, config):
        """
        Parameters
        ----------
        config : configparser.ConfigParser
            Parsed configuration object containing a ``[survey]`` section.
        """
        # survey area
        self.area_deg2 = np.array(config['survey'].get('survey_area').split()).astype('float')[0]

        # z bins to eval model
        self._get_z_bins(config)

        # magnitude range and nbins
        self.mag_min = config['survey'].getfloat('min_band_mag', 16)
        self.mag_max = config['survey'].getfloat('max_band_mag', 23)
        self.num_mag_bins = config['survey'].getint('num mag bins', 10)
        self.maglist = np.linspace(self.mag_min, self.mag_max, self.num_mag_bins)

        # resolution in km/s or dimensionless
        self.res_kms = config['survey'].getfloat('pix_res_kms', None)
        self.resolution = config['survey'].getfloat('resolution', None)

        # get magnitude band
        self.band = config['survey'].get('band')
        if self.band not in self.BAND_OPTIONS:
            raise ValueError(f'Please choose from accepted bandpasses: {self.BAND_OPTIONS}')


    def split_str(self,vals) :
        tmp2=list()
        for tmp in vals.split(" ") :
            tmp2 += tmp.split(",")
        return tmp2

    def _get_z_bins(self, config):
        """Parse redshift bin edges and centres from config.

        Parameters
        ----------
        config : configparser.ConfigParser
            Parsed configuration object with ``[survey]`` z-bin settings.
        """
        survey_cfg = config['survey']
        self.zmin = survey_cfg.getfloat('z bin min', 2)
        self.zmax = survey_cfg.getfloat('z bin max', 4)
        self.num_z_bins = survey_cfg.getint('num z bins', 1)
        z_bin_edges = survey_cfg.get('z bin edges', None)
        z_bin_centres = survey_cfg.get('z bin centres', None)
        if z_bin_edges is not None:
            assert(z_bin_centres is None) # cannot define both
            z_bin_edges = np.array(self.split_str(z_bin_edges)).astype(float)
            z_bin_min=z_bin_edges[:-1]
            z_bin_max=z_bin_edges[1:]
            self.z_bin_centres = (z_bin_min+z_bin_max)/2.
            self.z_bin_edges   = np.array([z_bin_min,z_bin_max])
            self.num_z_bins    = self.z_bin_centres.size
        elif z_bin_centres is not None:
            self.z_bin_centres = np.array(self.split_str(z_bin_centres)).astype(float)
            dz = np.zeros(self.z_bin_centres.size)
            dz[1:-1] = (self.z_bin_centres[2:] - self.z_bin_centres[:-2]) / 2.
            dz[0] = self.z_bin_centres[1] - self.z_bin_centres[0]
            dz[-1] = self.z_bin_centres[-1] - self.z_bin_centres[-2]
            self.z_bin_edges = np.array([self.z_bin_centres - dz / 2, self.z_bin_centres + dz / 2])
            self.num_z_bins    = self.z_bin_centres.size
        else:
            z_list = np.linspace(self.zmin, self.zmax, self.num_z_bins + 1)
            self.z_bin_edges = np.array(
                [[z_list[i], z_list[i + 1]] for i in range(self.num_z_bins)]
            ).T
            self.z_bin_centres = self.z_bin_edges.mean(axis=0)
