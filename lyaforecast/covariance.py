import numpy as np
from lyaforecast.weights import Weights

class Covariance: 
    """Compute covariance matrix components for given tracers, for a given survey.
        Different redshift bins are treated as independent, and right
        now this object only deals with one redshift bin at a time."""
    
    TRACER_OPTIONS = ['qso','lbg','lae']
    LYA_TRACER_OPTIONS = ['qso','lbg']
    CROSS_TRACER_OPTIONS = ['lya_qso','lya_lbg','lya_lae']

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
        self._pix_kms = None
        self._res_kms = None
        self._num_modes = None
                
        # verbosity level - I'll remove this later.
        self.verbose = 1


    @property
    def pix_width_kms(self):
        """Pixel width in km/s"""
        return self._pix_kms
    
    @property
    def pix_res_kms(self):
        """Pixel width in km/s"""
        return self._res_kms
    
    @property
    def num_modes(self):
        """Number of modes in z bin, as a function of k (h/Mpc)"""
        return self._num_modes
    
    
    def __call__(self,lmin,lmax):
        """Load central wavelength, redshift, p3d_w and p1d_w, given wavelength range"""

        self.lmin = lmin
        self.lmax = lmax
        self._z_mean = self._get_z_mean()

        #conversions
        self._wavelength_to_velocity = self._cosmo.velocity_from_wavelength(self._z_mean)
        self._distance_to_velocity = self._cosmo.velocity_from_distance(self._z_mean)
        self._angle_to_distance = self._cosmo.distance_from_degrees(self._z_mean)


        #load central redshift/wavelength of bin
        self._get_zq_bin()
        #get pix width in kms
        self._get_pix_kms()
        #get resolution in kms (for mean wavelength)
        self._get_res_kms()
        #get redshift limits of bins
        self._get_redshift_limits()
        #load survey volume
        self._get_survey_volume()
        #load number of modes
        self._get_num_modes()
    
    
    def _get_z_mean(self):
        """ given wavelength range covered in bin, compute central redshift"""

        return np.sqrt(self.lmin * self.lmax) / self._cosmo.LYA_REST - 1.0
    
    def _get_pix_kms(self):
        """Get pixel width in km/s, whether angstrom or km/s is provided"""

        #pix width in angstroms
        pix_ang = self._survey.pix_ang
        #assume if pix_ang is provided, it will be used. (can add flag later)
        if pix_ang is not None:
            self._pix_kms = pix_ang * self._wavelength_to_velocity
        else:
            self._pix_kms =  self._survey.pix_kms
            assert self._pix_kms is not None, 'Must provide pix width in kms or ang'      

    def _get_res_kms(self):
        """Get spectrograph reslution in km/s"""

        #assume specifying resolution in km/s takes precedence
        if self._survey.res_kms is not None:
            self._res_kms = self._survey.res_kms
        else:
            res_ang = self._lc / self._survey.resolution
            self._res_kms = res_ang * self._wavelength_to_velocity
        
    def _get_zq_bin(self):
        """Given wavelength range covered in bin, compute central tracer redshift and mean wavelength"""

        if not (self.lmin is None or self.lmax is None):
            # mean wavelenght of bin
            self._lc = np.sqrt(self.lmin * self.lmax)
            # redshift of quasar for which forest is centered in z
            lrc = np.sqrt(self._survey.lrmin * self._survey.lrmax)
            self._zq = (self._lc / lrc) - 1.0

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

        lmax_forest = self._survey.lrmax * (1 + self._zq)
        lmin_forest = self._survey.lrmin * (1 + self._zq)

        Lq_kms = self._cosmo.SPEED_LIGHT * np.log(lmax_forest/lmin_forest)

        return Lq_kms
    
    def _get_forest_wave(self):
        """Observed wavelength range of Lya forest"""

        lmax_forest = self._survey.lrmax * (1 + self._zq)
        lmin_forest = self._survey.lrmin * (1 + self._zq)
        nbins = int((lmax_forest - lmin_forest) / self._survey.pix_ang)

        self._forest_wave = np.linspace(lmin_forest,lmax_forest,nbins)

    def _get_survey_volume(self):
        
        # Survey volume in units of (Mpc/h)^3
        volume_degkms = self._survey.area_deg2 * self._get_redshift_depth()
        volume_mpch = volume_degkms * self._angle_to_distance**2 / self._distance_to_velocity

        self._survey_volume_mpc = volume_mpch
    
    
    def _get_num_modes(self):
        
        #prefactor of Fisher matrix
        self._num_modes = (self._survey_volume_mpc * self._power_spec.k**2 
                                * self._power_spec.dk * self._power_spec.dmu 
                                    / (2 * np.pi**2))

    def compute_eff_density_and_noise(self):
        """ Compute effective density of lines of sight and eff. noise power.
        Terms aliasing_weights and effective_noise_power in McDonald & Eisenstein (2007). """

        # length of forest in km/s
        forest_length = self._get_forest_length()

        # The code below evaluates all the quantities at the central redshift
        # of the bin, and uses a single quasar redshift assuming that all pixels
        # in the bin have restframe wavelength of the center of the forest.
        # Pat's code, instead, computes an average over both redshift of 
        # absorption and over quasar redshift.

        # weights instance
        self._weights = Weights(self._config,self._survey,self._cosmo,self._power_spec,self._spectrograph,
                               forest_length,self._pix_kms,self._res_kms,self._lc,
                               self._z_mean,self._zq,self._zmin,self._zmax)

        # lyman-alpha weights
        w_lya = self._weights.compute_weights()
        self._w_lya = w_lya

        # given weights, compute integrals in McDonald & Eisenstein (2007)
        int_1 = self._weights.compute_int_1(w_lya)
        int_2 = self._weights.compute_int_2(w_lya)
        int_3 = self._weights.compute_int_3(w_lya)

        # Pw2D in McDonald & Eisenstein (2007)
        self._aliasing_weights = int_2 / (int_1**2 * forest_length)
        # PNeff in McDonald & Eisenstein (2007)
        self._effective_noise_power = int_3 * self._pix_kms / (int_1**2 * forest_length)

        # tracer weights
        self._w_tracer = self._weights.compute_tracer_weights(self._tracer)   
        # tracer shot noise (deg^2 km/s)
        self._tracer_noise_power = 1 / self._weights.get_n_tracer()


    def compute_total_power(self,k_hmpc,mu,tracer):
        
        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity
        kt_deg = kt_hmpc * self._angle_to_distance

        if tracer=='lya':
            return self._compute_total_power_lya(kt_deg,kp_kms)
        elif tracer in self.TRACER_OPTIONS:
            return self._compute_total_power_tracer(k_hmpc,mu)
        elif tracer in self.CROSS_TRACER_OPTIONS:
            return self._compute_total_power_cross(kt_deg,kp_kms,k_hmpc,mu,tracer)
        else:
            raise ValueError('Invalid tracer name given for covariance computation:', tracer)

    def _compute_total_power_lya(self,kt_deg,kp_kms):
        """Sum of 3D Lya power, aliasing and effective noise power in mpc/h^3"""

        p3d = self._power_spec.compute_p3d_kms(self._z_mean,kt_deg,kp_kms,self._res_kms,self._pix_kms,'lya')
        aliasing = (self._aliasing_weights[-1] * 
                    self._power_spec.compute_p1d_kms(self._z_mean,kp_kms,self._res_kms,self._pix_kms))
        noise = self._effective_noise_power[-1]

        total_power = p3d + aliasing + noise

        total_power_mpc = total_power * self._angle_to_distance**2 / self._distance_to_velocity

        return total_power_mpc
    
    def _compute_total_power_tracer(self,k_hmpc,mu):
        p3d = self._power_spec.compute_p3d_hmpc_smooth(self._z_mean, k_hmpc,
                                        mu, self._pix_kms, self._res_kms,
                                        self._tracer) 
        #in 1/deg2km/s
        density_deg2kms = self._weights.get_n_tracer()[-1]
        #in 1/mpc/h^3
        density_mpc3 = density_deg2kms / self._angle_to_distance**2 * self._distance_to_velocity
        
        total_power_mpc = p3d + 1 / density_mpc3

        return total_power_mpc
    
    def _compute_total_power_cross(self,kt_deg,kp_kms,k_hmpc,mu,tracer):
        """Compute observed power from cross-correlation of forests and galaxies/quasars.
            Should be no noise contributions in theory."""

        p3d = self._power_spec.compute_p3d_hmpc(self._z_mean,k_hmpc,mu,which=tracer)

        return p3d



    ### Below this will soon be degraded functions ###


    def compute_neff_2D_lya(self,k_hmpc,mu):
        """Effective 2D density of lya forest skewers, as defined in McQuinn and White 2014.
        1 / n_eff^2D = P_w + P_N / P1D"""

        # decompose into line of sight component
        kp_hmpc = k_hmpc * mu
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity
        
        p1d = self._power_spec.compute_p1d_kms(self._z_mean,kp_kms,self._res_kms,self._pix_kms)
        neff_2D_inv = self._aliasing_weights + self._effective_noise_power / p1d

        return 1 / neff_2D_inv
    
    def _compute_lya_eff_vol(self,k_hmpc,mu):
        """Effective 2D density of lya forest skewers, as defined in McQuinn and White 2014.
        1 / n_eff^2D = P_w + P_N / P1D"""

        # decompose into line of sight component
        kp_hmpc = k_hmpc * mu
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity

        neff_2d = self.compute_neff_2D_lya(k_hmpc,mu) / self._angle_to_distance**2 

        p3d = self._power_spec.compute_p3d_hmpc(self._z_mean, k_hmpc, mu, 'lya')

        p1d = (self._power_spec.compute_p1d_kms(self._z_mean,kp_kms,self._res_kms,self._pix_kms) 
                            / self._distance_to_velocity)
        
        eff_vol = self._survey_volume_mpc * (p3d / (p3d + p1d / neff_2d))**2

        return eff_vol

    
    def compute_aliasing(self,z,kt_deg,kp_kms):

        p1d = self._power_spec.compute_p1d_kms(z,kp_kms,self._res_kms,self._pix_kms)
        aliasing = self._aliasing_weights[-1] * p1d
        
        return aliasing 
                    

    def compute_3d_power_variance(self,k_hmpc,mu
                        ):
        """Variance of 3D Lya power, in units of (Mpc/h)^3.
            Note that here 0 < mu < 1.
            """
        
        #We should move to computing this in a vectorised fashion, rather than 
        #iterating over mu/k values. Then I wouldn't have to call the mu/k values
        # again.

        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity
        kt_deg = kt_hmpc * self._angle_to_distance

        # get total power in units of observed coordinates 
        total_power_degkms = self.self._compute_total_power_lya(self._z_mean,kt_deg,kp_kms)
        # convert into units of (Mpc/h)^3
        total_power_hmpc = total_power_degkms * self._angle_to_distance**2 / self._distance_to_velocity

        # based on Eq 8 in Seo & Eisenstein (2003), but note that here we
        # use 0 < mu < 1 and they used -1 < mu < 1
        num_modes = self._survey_volume_mpc * k_hmpc**2 * self._power_spec.dk * self._power_spec.dmu / (2 * np.pi**2)
        power_variance = 2 * total_power_hmpc**2 / num_modes
        
        # self._w_lya = self._weights._p3d_w / (self._weights._p3d_w + power_variance)

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
                
        total_power_lya_degkms = self.self._compute_total_power_lya(z,kt_deg,kp_kms)
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
        total_power_lya = self.self._compute_total_power_lya(z,kt_deg,kp_kms)
        total_power_lya_hmpc = total_power_lya * dhmpc_ddeg**2 / dkms_dhmpc

        #cross
        cross_tracer = 'lya_' + self._tracer
        cross = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=cross_tracer)

        #tracer auto
        tracer_auto = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=self._tracer)
        
        #dn/ddeg2dkm/s to dn/dhmpc^3
        dn_dkmsddeg2_tracer = self._weights.get_n_tracer()
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


    # def compute_tracer_cross_power_variance(self,k_hmpc,mu
    #                     ):
    #     """Variance of Lya-tracer cross-power, in units of (Mpc/h)^3.
    #         From eq.34 of McQuinn and White (2011).
    #         Note that here -1 < mu < 1.
    #        """
    #     z = self._mean_z()
    #     # decompose into line of sight and transverse components
    #     dkms_dhmpc = self._cosmo.velocity_from_distance(z)
    #     dhmpc_ddeg = self._cosmo.distance_from_degrees(z)
    #     kp_hmpc = k_hmpc * mu
    #     kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
    #     kp_kms = kp_hmpc / dkms_dhmpc
    #     kt_deg = kt_hmpc * dhmpc_ddeg

    #     #cross
    #     _tracer_name = self._tracer1 + '_' + self._tracer2
    #     tracer_cross = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=_tracer_name)

    #     #tracer auto
    #     tracer1_auto = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=self._tracer1)
    #     tracer2_auto = self._power_spec.compute_p3d_hmpc(z,k_hmpc,mu,which=self._tracer2)
        
    #     #dn/ddeg2dkm/s to dn/dhmpc^3
    #     dn_dkmsddeg2_tracer1 = self._weights.get_n_tracer(self._tracer1)
    #     dn_dhmpc3_tracer1 = dn_dkmsddeg2_tracer1 / dhmpc_ddeg**2 * dkms_dhmpc
    #     noise_tracer1 = 1 / dn_dhmpc3_tracer1

    #     dn_dkmsddeg2_tracer2 = self._weights.get_n_tracer(self._tracer2)
    #     dn_dhmpc3_tracer2 = dn_dkmsddeg2_tracer2 / dhmpc_ddeg**2 * dkms_dhmpc
    #     noise_tracer2 = 1 / dn_dhmpc3_tracer2

    #     #var cross
    #     var_tracer_cross = tracer_cross**2 + (tracer1_auto + noise_tracer1) * (tracer2_auto + noise_tracer2)

    #     self._w_cross = abs(tracer_cross) / (abs(tracer_cross) + var_tracer_cross)

    #     # survey volume in units of (Mpc/h)^3
    #     volume_mpch = self.get_survey_volume()

    #     #num modes
    #     num_modes = volume_mpch * k_hmpc**2 * self._power_spec.dk * self._power_spec.dmu / (2 * np.pi**2)

    #     # not sure if there should be a factor of 2 here.
    #     power_var = 2 * var_tracer_cross / num_modes
    #     if not self.per_mag:
    #         power_var = power_var[-1]

    #     return power_var



