""" In this module we store all of the survey specifications used in the forecast,
       including the quasar luminosity function."""
import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from lyaforecast.utils import get_file
from scipy.ndimage import gaussian_filter1d


class Survey:

    BAND_OPTIONS = ['r']
    TRACER_OPTIONS = ['qso','lbg','lae']

    def __init__(self,config):
        #survey area
        self.area_deg2 = np.array(config['survey'].get('survey_area').split()).astype('float')[0]
        
        # z bins to eval model
        self._get_z_bins(config)

        # magnitude range and nbins
        self.mag_min = config['survey'].getfloat('min_band_mag',16)
        self.mag_max = config['survey'].getfloat('max_band_mag',23)
        self.num_mag_bins = config['survey'].getint('num mag bins',10)
        self.maglist = np.linspace(self.mag_min,self.mag_max,self.num_mag_bins)
        
        # resolution in km/s or dimensionless
        self.res_kms = config['survey'].getfloat('pix_res_kms',None)
        self.resolution = config['survey'].getfloat('resolution',None)

        #get magnitude band 
        self.band = config['survey'].get('band')
        if self.band not in self.BAND_OPTIONS:
            raise ValueError(f'Please choose from accepted bandpasses: {self.BAND_OPTIONS}')
        

        # definition of forest (lrmin,lrmax)
        self.lrmin = config['lya forest'].getfloat('min_rest_frame_lya')
        self.lrmax = config['lya forest'].getfloat('max_rest_frame_lya')

        #number of exposures (for snr of forest)
        self.num_exp = config['lya forest'].getfloat('num exposures')

        #pixel width (in angstroms or kms)
        self.pix_kms = config['lya forest'].getfloat('pix_width_kms',None)
        self.pix_ang = config['lya forest'].getfloat('pix_width_ang',None)


        #luminosity functions
        self._lya_tracer_dzdz_file = get_file(config['lya forest'].get('dn dz'))
        self._tracer_dzdz_file = get_file(config['tracer'].get('dn dz'))

        # tracers
        self.lya_tracer = config['lya forest'].get('tracer', 'qso')
        self.tracer = config['tracer'].get('tracer', None)
        if self.lya_tracer not in self.TRACER_OPTIONS or self.tracer not in self.TRACER_OPTIONS:
            raise ValueError(f'Please choose from accepted source tracers: {self.TRACER_OPTIONS}')
        
        # densities
        self.lya_density = config['lya forest'].getfloat('target density')
        self.tracer_density = config['tracer'].getfloat('target density')

        # dn/dzdm
        self._source_dndz = self._setup_dndzdm_lya(self._lya_tracer_dzdz_file)
        if self.tracer is not None:
           self._tracer_dndz = self._setup_dndzdm_tracer(self._tracer_dzdz_file)

    def _get_z_bins(self,config):
        survey_cfg = config['survey']
        z_centres_flag = survey_cfg.get('z bin centres')

        self.zmin = survey_cfg.getfloat('z bin min', 2)
        self.zmax = survey_cfg.getfloat('z bin max', 4)
        self.num_z_bins = survey_cfg.getint('num z bins', 1)

        if z_centres_flag:
            self.z_bin_centres = np.fromstring(z_centres_flag, sep=',')
            dz = np.diff(self.z_bin_centres, prepend=self.z_bin_centres[0], append=self.z_bin_centres[-1]) / 2
            self.z_bin_edges = np.array([self.z_bin_centres - dz[:-1], self.z_bin_centres + dz[1:]])
        else:
            self._z_list = np.linspace(self.zmin, self.zmax, self.num_z_bins+1)
            self.z_bin_edges = np.array([[self._z_list[i], self._z_list[i + 1]] for i in range(self.num_z_bins)])
            self.z_bin_centres = self.z_bin_edges.mean(axis=0)

    def _setup_dndzdm_lya(self,file):
        """Setup dndz/dm from file"""

        z,m,tdNdmdzddeg2 = np.loadtxt(file,unpack=True)
        z = np.unique(z)
        m = np.unique(m)

        #scale density of quasars to desired number. By default given staright from QLF
        if self.lya_density is not None:
            # the DESI requirement is 50 quasars per square degree above 2.15
            z_min_lya = 2.15
            current_total_density = np.sum(tdNdmdzddeg2.reshape(z.size,m.size)[z>z_min_lya])
            print("Scaling lya dndzdm from a total density (z>={}) of {} to {}/deg2".format(z_min_lya,current_total_density,self.lya_density))
            tdNdmdzddeg2 *= (self.lya_density/current_total_density)
        
         # This assumes entries are evenly spaced. 
        dz = z[1] - z[0]
        dm = m[1] - m[0]

        tdNdmdzddeg2 /= (dz*dm)
        tdNdmdzddeg2 = np.reshape(tdNdmdzddeg2,[len(z),len(m)])

        # figure out allowed redshift range (will check out of bounds)
        self._zmin = z[0] #- 0.5*dz
        self._zmax = z[-1] #+ 0.5*dz
        # figure out allowed magnitude range (will check out of bounds)
        self._lya_mmin = m[0] #- 0.5*dm
        self._lya_mmax = m[-1] #+ 0.5*dm

        if self.lya_tracer == 'lbg':
            #smooth (currently) noisy dndz
            sigma_smooth = 1.5
            tdNdmdzddeg2 = gaussian_filter1d(tdNdmdzddeg2,sigma_smooth,axis=0)

        interpolator = RectBivariateSpline(z,m,
                  tdNdmdzddeg2,bbox=[self._zmin,self._zmax,self._lya_mmin,self._lya_mmax],
                  kx=2,ky=2)
        
        return interpolator
    
    def _setup_dndzdm_tracer(self,file):
        """Setup dndz/dm from file"""

        z,m,tdNdmdzddeg2 = np.loadtxt(file,unpack=True)
        z = np.unique(z)
        m = np.unique(m)

        #scale density of quasars to desired number. By default taken straight from QLF
        if self.tracer_density is not None:
            #re-scale based on lya qso requirements
            if self.tracer == 'qso':
                current_total_density = np.sum(tdNdmdzddeg2.reshape(z.size,m.size)[z>2.15])
            else:
                current_total_density = np.sum(tdNdmdzddeg2.reshape(z.size,m.size))
            print("Scaling dndzdm tracer from a total density of {} to {}/deg2".format(current_total_density,
                                                                                       self.tracer_density))
            tdNdmdzddeg2 *= (self.tracer_density/current_total_density)
        
         # This assumes entries are evenly spaced. 
        dz = z[1] - z[0]
        dm = m[1] - m[0]

        tdNdmdzddeg2 /= (dz*dm)
        tdNdmdzddeg2 = np.reshape(tdNdmdzddeg2,[len(z),len(m)])

        # figure out allowed redshift range (will check out of bounds)
        self._zmin = z[0] #- 0.5*dz
        self._zmax = z[-1] #+ 0.5*dz
        # figure out allowed magnitude range (will check out of bounds)
        self._tracer_mmin = m[0] #- 0.5*dm
        self._tracer_mmax = m[-1] #+ 0.5*dm

        if (self.tracer == 'lbg')|(self.tracer == 'lae'):
            #smooth (currently) noisy dndz
            sigma_smooth = 1.5
            tdNdmdzddeg2 = gaussian_filter1d(tdNdmdzddeg2,sigma_smooth,axis=0)

        interpolator = RectBivariateSpline(z,m,
                  tdNdmdzddeg2,bbox=[self._zmin,self._zmax,self._tracer_mmin,self._tracer_mmax],
                  kx=2,ky=2)
        
        return interpolator
    
    #maybe this function isn't necessary
    def get_dn_dzdm(self,z, m, which='lya'):
            
        if which == 'lya':
            points = self._source_dndz(z, m, grid=False)
            points[m>self._lya_mmax] = 1e-20
            points[m<self._lya_mmin] = 1e-20
        elif which == 'tracer':
            points = self._tracer_dndz(z, m, grid=False)
            points[m>self._tracer_mmax] = 1e-20
            points[m<self._tracer_mmin] = 1e-20
            
        return points
    def _setup_dn_dz(self):
        """This interpolator is only a function of z, for a given maximum m"""
        z,m,dn_dzddeg2 = np.loadtxt(self._qso_lum_file,unpack=True)
        z = np.unique(z)

        #scale density of quasars to desired number. By default given staright from QLF
        if self.qso_density is not None:
            # the DESI requirement is 50 quasars per square degree above 2.15
            z_min_lya = 2.15
            current_total_density = np.sum(dn_dzddeg2[z>z_min_lya])
            print("Scaling dndzdmag from a total density (z>={}) of {} to {}/deg2".format(z_min_lya,current_total_density,self.qso_density))
            dn_dzddeg2 *= (self.qso_density/current_total_density)
        
        # This assumes entries are evenly spaced. 
        dz = z[1] - z[0]
        dn_dzddeg2 /= dz

        self._get_qso_lum_func = UnivariateSpline(z, dn_dzddeg2)

    def get_qso_lum_func_old(self,z, m=None):
            
            # x_clamped = np.clip(x_query, self._zmin, self._zmax)
            # y_clamped = np.clip(y_query, 17, 23)

            #clamp points to a minimum/maximum magnitude. It's kind of arbitrary right now.
            #Otherwise interpolator is a bit shit.
            #removing this temporarily to use LBGs (much fainter)
            # out_of_bounds_mu = np.where(y_query > 23)
            # out_of_bounds_ml = np.where(y_query < 17)
            if self.desi_sv:
                points = self._get_qso_lum_func(z)
            else:
                points = self._get_qso_lum_func(z, m, grid=False)
                points[m>self._mmax] = 1e-20
                points[m<self._mmin] = 1e-20
                #points[21<(m<self._mmin)] = self._get_qso_lum_func(z, self._mmin, grid=False)

            # points[out_of_bounds_mu] = 1e-20
            # points[out_of_bounds_ml] = 1e-20
            

            return points