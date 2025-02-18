""" In this module we store all of the survey specifications used in the forecast,
       including the quasar luminosity function."""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from lyaforecast.utils import get_file

class Survey:

    BAND_OPTIONS = ['r']

    def __init__(self,config):
        #survey conditions (not sure where to put these yet)
        self.area_deg2 = np.array(config['survey'].get('survey_area').split()).astype('float')[0]
        self.qso_density = np.array(config['survey'].get('qso density').split()).astype('float')[0]
        # quasar redshift range
        self.zq_min = config['survey'].getfloat('z_qso_min')
        self.zq_max = config['survey'].getfloat('z_qso_max')
        # z bins to eval model
        self.zmin = config['survey'].getfloat('z bin min', 2)
        self.zmax = config['survey'].getfloat('z bin max', 4)
        self.num_z_bins = config['survey'].getint('num z bins', 1)
        # magnitude range and nbins
        self.mag_min = config['survey'].getfloat('min_band_mag',16)
        self.mag_max = config['survey'].getfloat('max_band_mag',23)
        self.num_mag_bins = config['survey'].getint('num mag bins',10)
        self.maglist = np.linspace(self.mag_min,self.mag_max,self.num_mag_bins)
        # pixel width and resolution in km/s (for now)
        self.pix_kms = config['survey'].getfloat('pix_width_kms')
        self.res_kms = np.array(config['survey'].get('pix_res_kms').split()).astype('float')[0]
        # definition of forest (lrmin,lrmax)
        self.lrmin = config['survey'].getfloat('min_rest_frame_lya')
        self.lrmax = config['survey'].getfloat('max_rest_frame_lya')
        #theoretica obs wavelength limits, must also apply limits of instrument.
        #At one point I will set a bound of lmin > l_obs_lower_limit etc.
        self.l_obs_lower_limit = self.lrmin * (1 + self.zq_min)
        self.l_obs_upper_limit = self.lrmax * (1 + self.zq_max)
        #number of exposures
        self.num_exp = config['survey'].getint('num exposures')

        #get magnitude band 
        self.band = config['survey'].get('band')
        if self.band not in self.BAND_OPTIONS:
            raise ValueError(f'Please choose from accepted bandpasses: {self.BAND_OPTIONS}')

        #get file with luminosity function
        self._qso_lum_file = get_file(config['survey'].get('qso-lum-file'))

        self._setup_YecheFile()

    def _setup_YecheFile(self):
        """Setup objects from file"""
        # use file from Christophe / Nathalie
        # read table       
        #print("reading",self._snr_dir,"in QuasarLF:_setup_YecheFile")
        # read table       

        z,m,tdNdmdzddeg2 = np.loadtxt(self._qso_lum_file,unpack=True)
        z = np.unique(z)
        m = np.unique(m)
        #scale density of quasars to desired number. By default given staright from QLF
        if self.qso_density is not None:
            # the DESI requirement is 50 quasars per square degree above 2.15
            z_min_lya = 2.15
            current_total_density = np.sum(tdNdmdzddeg2.reshape(z.size,m.size)[z>z_min_lya])
            print("Scaling dndzdmag from a total density (z>={}) of {} to {}/deg2".format(z_min_lya,current_total_density,self.qso_density))
            tdNdmdzddeg2 *= (self.qso_density/current_total_density)
        
        # This assumes entries are evenly spaced. 
        dz = z[1] - z[0]
        dm = m[1] - m[0]
        tdNdmdzddeg2 /= (dz*dm)
        tdNdmdzddeg2 = np.reshape(tdNdmdzddeg2,[len(z),len(m)])

        # figure out allowed redshift range (will check out of bounds)
        zmin = z[0] - 0.5*dz
        zmax = z[-1] + 0.5*dz
        # figure out allowed magnitude range (will check out of bounds)
        mmin = m[0] - 0.5*dm
        mmax = m[-1] + 0.5*dm

        # create interpolation object
        self.get_qso_lum_func = RectBivariateSpline(z,m,
                  tdNdmdzddeg2,bbox=[zmin,zmax,mmin,mmax],
                  kx=2,ky=2)

    def range_z(self):
        """Return range of quasar redshifts covered by QL"""
        return self.zmin,self.zmax

    def range_mag(self):
        """Return range of magnitudes covered by QL"""
        return self.mmin,self.mmax

    # @property
    # def get_qso_lum_func(self,zq,mag):
    #     """Quasar luminosity function, observed units.
    #        Per unit redshift, unit (observed) magnitude and square degree."""
        
    #     return self._get_qso_lum_func(zq,mag)