import numpy as np
from lyaforecast.weights import Weights

class Covariance: 
    """Compute error-bars for Lyman alpha P(z,k,mu) for a given survey.
        Different redshift bins are treated as independent, and right
        now this object only deals with one redshift bin at a time."""
    
    TRACER_OPTIONS = ['qso','lbg']
    def __init__(self, config, cosmo, survey, spectrograph, power_spectrum):

        # Config
        self._config = config
        # Cosmological model
        self._cosmo = cosmo
        # survey instance
        self._survey = survey
        # spectrograph instance
        self._spectrograph = spectrograph
        # power spectrum instance
        self._power_spec = power_spectrum
        
        #whether to iterate over magnitude instead of redshift
        self.per_mag = self._config['control'].getboolean('per mag')

        #tracer type
        self._tracer = self._config['tracer'].get('tracer')
        if self._tracer not in self.TRACER_OPTIONS:
            raise ValueError(f'Please choose from accepted tracers: {self.TRACER_OPTIONS}')

        # These will be dependent on redshift bins, 
        # which are passed during foreacast run (for now).
        self.lmin = None
        self.lmax = None
        self._aliasing_weights = None
        self._effective_noise_power = None
                
        # verbosity level - I'll remove this later.
        self.verbose = 1

    def __call__(self,lmin,lmax):
        """Load central wavelength, redshift, p3d_w and p1d_w, given wavelength range"""
        self.lmin = lmin
        self.lmax = lmax
        #load central redshift/wavelength of bin
        self._get_zq_bin()
        #get pix width in kms
        self._get_pix_kms()
        #get resolution in kms (for mean wavelength)
        self._get_res_kms()
        #get redshift limits of bins
        self._get_redshift_limits()

    def _get_pix_kms(self):
        #get pix width in kms, whether angstrom or km/s is provided.
        z = self._mean_z()
        #pix width in angstroms
        pix_ang = self._survey.pix_ang
        #assume if pix_ang is provided, it will be used. (can add flag later)
        if pix_ang is not None:
            self._pix_kms = pix_ang * self._cosmo.velocity_from_wavelength(z)
        else:
            self._pix_kms =  self._survey.pix_kms
            assert self._pix_kms is not None, 'Must provide pix width in kms or ang'      

    def _get_res_kms(self):
        #get spectrograph reslution in km/s
        #mean redshift
        z = self._mean_z()
        #assume specifying resolution in km/s takes precedence
        if self._survey.res_kms is not None:
            self._res_kms = self._survey.res_kms
        else:
            res_ang = self._lc / self._survey.resolution
            self._res_kms = res_ang * self._cosmo.velocity_from_wavelength(z)
        
    def _get_zq_bin(self):
        # given wavelength range covered in bin, compute central tracer redshift and mean wavelength
        if not (self.lmin is None or self.lmax is None):
            # mean wavelenght of bin
            self._lc = np.sqrt(self.lmin * self.lmax)
            # redshift of quasar for which forest is centered in z
            lrc = np.sqrt(self._survey.lrmin * self._survey.lrmax)
            self._zq = (self._lc / lrc) - 1.0

    def _mean_z(self):
        """ given wavelength range covered in bin, compute central redshift"""

        return np.sqrt(self.lmin * self.lmax) / self._cosmo.LYA_REST - 1.0

    def _get_redshift_depth(self):
        """Depth of redshift bin, in km/s"""

        c_kms = self._cosmo.SPEED_LIGHT
        L_kms = c_kms*np.log(self.lmax/self.lmin)

        return L_kms
    
    def _get_redshift_limits(self):
        """Redshift limits of bin, computed after calling class instance with lmin, lmax."""

        self._zmin = self.lmin / self._cosmo.LYA_REST - 1
        self._zmax = self.lmax / self._cosmo.LYA_REST - 1

    def _get_forest_length(self):
        """Length of Lya forest, in km/s"""

        c_kms = self._cosmo.SPEED_LIGHT
        lmax_forest = self._survey.lrmax * (1 + self._zq)
        lmin_forest = self._survey.lrmin * (1 + self._zq)

        #this ensures the forest limits cannot be out of the defined bin.
        #lmin_forest = np.fmax(lmin_forest,self.lmin)
        #lmax_forest = np.fmin(lmax_forest,self.lmax)

        Lq_kms = c_kms*np.log(lmax_forest/lmin_forest)

        return Lq_kms
    
    def _get_forest_wave(self):

        lmax_forest = self._survey.lrmax * (1 + self._zq)
        lmin_forest = self._survey.lrmin * (1 + self._zq)
        nbins = int((lmax_forest - lmin_forest) / self._survey.pix_ang)
        self._forest_wave = np.linspace(lmin_forest,lmax_forest,nbins)

    def get_survey_volume(self):
        
        z = self._mean_z()
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)
        # _survey volume in units of (Mpc/h)^3
        volume_degkms = self._survey.area_deg2 * self._get_redshift_depth()
        volume_mpch = volume_degkms * dhmpc_ddeg**2 / dkms_dhmpc

        return volume_mpch

    def compute_eff_density_and_noise(self):
        """Compute effective density of lines of sight and eff. noise power.
            Terms aliasing_weights and effective_noise_power in McDonald & Eisenstein (2007)."""

        # length of forest in km/s
        forest_length = self._get_forest_length()
        # length of pixel in km/s
        pixel_length = self._pix_kms
        # resolution
        resolution = self._res_kms
        # central forest wavelength
        lambda_mean = self._lc
        # central redshift bin
        z_centre = self._mean_z()
        # quasar redshift that centres forest in bin
        z_tracer = self._zq
        # The code below is closer to the method described in the publication,
        # but the c++ code by Pat is more complicated. 
        # The code below evaluates all the quantities at the central redshift
        # of the bin, and uses a single quasar redshift assuming that all pixels
        # in the bin have restframe wavelength of the center of the forest.
        # Pat's code, instead, computes an average over both redshift of 
        # absorption and over quasar redshift.
        # weights instance
        self.weights = Weights(self._config,self._survey,self._cosmo,self._power_spec,self._spectrograph,
                               forest_length,pixel_length,resolution,lambda_mean,
                               z_centre,z_tracer,self._zmin,self._zmax)

        # lyman-alpha weights
        w_lya = self.weights.compute_weights()
        self._w_lya = w_lya

        # given weights, compute integrals in McDonald & Eisenstein (2007)
        int_1 = self.weights.compute_int_1(w_lya)
        int_2 = self.weights.compute_int_2(w_lya)
        int_3 = self.weights.compute_int_3(w_lya)

        # Pw2D in McDonald & Eisenstein (2007)
        self._aliasing_weights = int_2 / (int_1**2 * forest_length)
        # PNeff in McDonald & Eisenstein (2007)
        self._effective_noise_power = int_3 * pixel_length / (int_1**2 * forest_length)

        # quasar weights
        self._w_tracer = self.weights.compute_tracer_weights(self._tracer)   
        # quasar shot noise (deg^2 km/s)
        self._tracer_noise_power = 1 / self.weights.get_n_tracer()

    def _compute_total_lya_power(self,z,kt_deg,kp_kms):
        """Sum of 3D Lya power, aliasing and effective noise power in deg2kms"""
        # previously computed p2wd and pn_eff.
        p3d = self._power_spec.compute_p3d_kms(z,kt_deg,kp_kms,self._res_kms,self._pix_kms,'lya')
        aliasing = self._aliasing_weights * self._power_spec.compute_p1d_kms(z,kp_kms,self._res_kms,self._pix_kms)
        noise = self._effective_noise_power

        total_power = p3d + aliasing + noise

        return total_power

    def compute_3d_power_variance(self,k_hmpc,mu
                        ):
        """Variance of 3D Lya power, in units of (Mpc/h)^3.
            Note that here 0 < mu < 1.
            """
        
        #We should move to computing this in a vectorised fashion, rather than 
        #iterating over mu/k values. Then I wouldn't have to call the mu/k values
        # again.
        z = self._mean_z()

        # transform from comoving to observed coordinates
        dkms_dmpch = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)

        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / dkms_dmpch
        kt_deg = kt_hmpc * dhmpc_ddeg

        # get total power in units of observed coordinates 
        # To-do: get P_total(mag)
        total_power_degkms = self._compute_total_lya_power(z,kt_deg,kp_kms)
        # convert into units of (Mpc/h)^3
        total_power_hmpc = total_power_degkms * dhmpc_ddeg**2 / dkms_dmpch
        #survey volume
        vol_hmpc = self.get_survey_volume()
        # based on Eq 8 in Seo & Eisenstein (2003), but note that here we
        # use 0 < mu < 1 and they used -1 < mu < 1
        num_modes = vol_hmpc * k_hmpc**2 * self._power_spec.dk * self._power_spec.dmu / (2 * np.pi**2)
        power_variance = 2 * total_power_hmpc**2 / num_modes

        # self._w_lya = self.weights._p3d_w / (self.weights._p3d_w + power_variance)

        #If not per magnitude, return power var for mmax only. 
        # Otherwise as a function of m input.
        if not self.per_mag:
            power_variance = power_variance[-1]

        return power_variance
    
    def compute_n_pk(self,k,mu):
        """Compute power per node nP at given k (h/Mpc) and mu"""
        z = self._mean_z()

        # transform from comoving to observed coordinates
        dkms_dmpch = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)

        # decompose into line of sight and transverse components
        kp_hmpc = k * mu
        kt_hmpc = k * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / dkms_dmpch
        kt_deg = kt_hmpc * dhmpc_ddeg
                
        total_power_lya_degkms = self._compute_total_lya_power(z,kt_deg,kp_kms)
        noise_lya = total_power_lya_degkms - self._power_spec.compute_p3d_kms(z,kt_deg,kp_kms,
                                                                              self._res_kms
                                                                              ,self._pix_kms,'lya')
        #lya
        np_lya = self._power_spec.compute_p3d_kms(z,kt_deg,kp_kms,self._res_kms
                                                  ,self._pix_kms,'lya') / noise_lya[-1]

        # tracer
        np_tracer = self._power_spec.compute_p3d_kms(z,kt_deg,kp_kms,self._res_kms
                                                  ,self._pix_kms,self._tracer) / self._tracer_noise_power[-1]
        
        return np_lya, np_tracer

    def compute_tracer_power_variance(self,k_hmpc,mu):
        """The squared fractional error on the 3D bandpower of quasars, computed in h/Mpc.
            The result is the denominator of the fisher matrix calculation."""
        
        z = self._mean_z()

        vol_element = k_hmpc**2 * self._power_spec.dk * self._power_spec.dmu / (2 * np.pi**2)
        eff_vol = self._compute_tracer_eff_vol(k_hmpc,mu)
        p3d = self._power_spec.compute_p3d_hmpc_smooth(z, k_hmpc,
                                        mu, self._pix_kms, self._res_kms,
                                        self._tracer) 
        
        power_variance =  2 * p3d**2 / (eff_vol * vol_element)
        
        if not self.per_mag:
            power_variance = power_variance[-1]
        
        return power_variance

    def _compute_tracer_eff_vol(self,k_hmpc,mu):

        z = self._mean_z()
        #compute weights
        #computer covariance
        dkms_dmpch = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)

        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / dkms_dmpch
        kt_deg = kt_hmpc * dhmpc_ddeg

        volume_degkms = self._survey.area_deg2 * self._get_redshift_depth()
        volume_hmpc = volume_degkms * dhmpc_ddeg**2 / dkms_dmpch

        # compute power in km/s
        p3d_tracer = self._power_spec.compute_p3d_kms(z,kt_deg,kp_kms,self._res_kms
                                                  ,self._pix_kms,self._tracer)     

        # We assume weights module is initialised.
        #per (deg^2 km/s)
        shot_noise = self._tracer_noise_power
        #in h/Mpc^3
        eff_vol = volume_hmpc * (p3d_tracer / (p3d_tracer + shot_noise))**2

        return eff_vol
    
    def compute_cross_power_variance(self,k_hmpc,mu
                        ):
        """Variance of Lya-tracer cross-power, in units of (Mpc/h)^3.
            From eq.34 of McQuinn and White (2011).
            Note that here -1 < mu < 1.
           """
        z = self._mean_z()
        # decompose into line of sight and transverse components
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        kp_kms = kp_hmpc / dkms_dhmpc
        kt_deg = kt_hmpc * dhmpc_ddeg

        #lya auto
        total_power_lya = self._compute_total_lya_power(z,kt_deg,kp_kms)
        total_power_lya_hmpc = total_power_lya * dhmpc_ddeg**2 / dkms_dhmpc

        #cross
        cross_tracer = 'lya_' + self._tracer
        cross = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=cross_tracer)

        #tracer auto
        tracer_auto = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=self._tracer)
        
        #dn/ddeg2dkm/s to dn/dhmpc^3
        dn_dkmsddeg2_tracer = self.weights.get_n_tracer()
        dn_dhmpc3 = dn_dkmsddeg2_tracer / dhmpc_ddeg**2 * dkms_dhmpc
        noise = 1 / dn_dhmpc3

        #var cross
        var_p_cross = cross**2 + total_power_lya_hmpc * (tracer_auto + noise)

        self._w_cross = abs(cross) / (abs(cross) + var_p_cross)

        # survey volume in units of (Mpc/h)^3
        volume_mpch = self.get_survey_volume()

        #num modes
        num_modes = volume_mpch * k_hmpc**2 * self._power_spec.dk * self._power_spec.dmu / (2 * np.pi**2)

        # not sure if there should be a factor of 2 here.
        power_var = 2 * var_p_cross / num_modes
        if not self.per_mag:
            power_var = power_var[-1]

        return power_var

