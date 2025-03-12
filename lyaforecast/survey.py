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
        # z bins to eval model
        self.zmin = config['survey'].getfloat('z bin min', 2)
        self.zmax = config['survey'].getfloat('z bin max', 4)
        self.num_z_bins = config['survey'].getint('num z bins', 1)
        self._bin_space = np.linspace(self.zmin, self.zmax, self.num_z_bins+1)
        self.z_bin_centres = self._bin_space[:-1] + np.diff(self._bin_space)[0]/2
        # quasar redshift range and num bins for 3d density
        self.zq_min = config['survey'].getfloat('z_qso_min')
        self.zq_max = config['survey'].getfloat('z_qso_max')
        self.nzq = config['survey'].getint('num qso bins')
        # magnitude range and nbins
        self.mag_min = config['survey'].getfloat('min_band_mag',16)
        self.mag_max = config['survey'].getfloat('max_band_mag',23)
        self.num_mag_bins = config['survey'].getint('num mag bins',10)
        self.maglist = np.linspace(self.mag_min,self.mag_max,self.num_mag_bins)
        # pixel width and resolution in km/s (for now)
        self.pix_kms = config['survey'].getfloat('pix_width_kms')
        self.res_kms = config['survey'].getfloat('pix_res_kms',None)
        #in angstrom
        self.pix_ang = config['survey'].getfloat('pix_width_ang',None)
        self.resolution = config['survey'].getfloat('resolution',None)
        # definition of forest (lrmin,lrmax)
        self.lrmin = config['survey'].getfloat('min_rest_frame_lya')
        self.lrmax = config['survey'].getfloat('max_rest_frame_lya')
        #theoretica obs wavelength limits, must also apply limits of instrument.
        #At one point I will set a bound of lmin > l_obs_lower_limit etc.
        self.l_obs_lower_limit = self.lrmin * (1 + self.zq_min)
        self.l_obs_upper_limit = self.lrmax * (1 + self.zq_max)
        #number of exposures
        self.num_exp = config['survey'].getfloat('num exposures')

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

        #Calum: This previous method makes no sense to me? It returns a negative number when m>mmax
        #why consider the region mmax to mmax + 0.5 * dm?
        # figure out allowed redshift range (will check out of bounds)
        self._zmin = z[0] #- 0.5*dz
        self._zmax = z[-1] #+ 0.5*dz
        # figure out allowed magnitude range (will check out of bounds)
        self._mmin = m[0] #- 0.5*dm
        self._mmax = m[-1] #+ 0.5*dm
        # create interpolation object
        self._get_qso_lum_func = RectBivariateSpline(z,m,
                  tdNdmdzddeg2,bbox=[self._zmin,self._zmax,self._mmin,self._mmax],
                  kx=2,ky=2)

    def get_qso_lum_func(self,x_query, y_query):
            
            # x_clamped = np.clip(x_query, self._zmin, self._zmax)
            # y_clamped = np.clip(y_query, 17, 23)

            #clamp points to a minimum/maximum magnitude. It's kind of arbitrary right now.
            #Otherwise interpolator is a bit shit.
            #removing this temporarily to use LBGs (much fainter)
            # out_of_bounds_mu = np.where(y_query > 23)
            # out_of_bounds_ml = np.where(y_query < 17)
            points = self._get_qso_lum_func(x_query, y_query, grid=False)
            # points[out_of_bounds_mu] = 1e-20
            # points[out_of_bounds_ml] = 1e-20

            return points