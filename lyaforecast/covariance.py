import numpy as np

class Covariance:
    """Compute error-bars for Lyman alpha P(z,k,mu) for a given survey.
        Different redshift bins are treated as independent, and right
        now this object only deals with one redshift bin at a time."""
    def __init__(self, config, cosmo, survey, spectrograph, power_spectrum):

        # Cosmological model
        self.cosmo = cosmo
        # survey instance
        self.survey = survey
        # spectrograph instance
        self.spectrograph = spectrograph
        # power spectrum instance
        self.power_spec = power_spectrum

        #power spectrum calculation details
        _properties = config['power spectrum']
        self._k_min_hmpc = _properties.getfloat('k_min_hmpc', 1e-4)
        self._k_max_hmpc = _properties.getfloat('k_max_hmpc', 1)
        self._num_k_bins = _properties.getint('num_k_bins', 100)
        self._mu_min = _properties.getfloat('mu_min', 0)
        self._mu_max = _properties.getfloat('mu_max', 1)
        self._num_mu_bins = _properties.getint('num_mu_bins', 10)
        #True to ignore small-scale components of power spectra.
        self._linear = _properties.getboolean('linear power')
        
        #whether to iterate over magnitude instead of redshift
        self.per_mag = config['control'].getboolean('per mag')

        #Fourier mode to evaluate S/N weights
        self._get_eval_mode(config)

        #grids for evaluating power spectrum
        self.k = np.linspace(self._k_min_hmpc,self._k_max_hmpc, self._num_k_bins)
        self.logk = np.log(self.k)
        self._dk = self.k[1] - self.k[0]
        self.dlogk = self._dk / self.k

        #bin edges
        mu_edges = np.linspace(self._mu_min,self._mu_max, self._num_mu_bins)
        self.mu = mu_edges[:-1] + np.diff(mu_edges)/2     
        self.dmu = np.diff(mu_edges)[0]

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
        #pre-compute power spectra
        self._p3d_w = self._compute_p3d_kms(self.kt_w_deg,self.kp_w_kms)
        self._p1d_w = self._compute_p1d_kms(self.kp_w_kms)

    def _get_pix_kms(self):
        #get pix width in kms, whether angstrom or km/s is provided.
        z = self._mean_z()
        #pix width in angstroms
        pix_ang = self.survey.pix_ang
        #assume if pix_ang is provided, it will be used. (can add flag later)
        if pix_ang is not None:
            self._pix_kms = pix_ang * self.cosmo.velocity_from_wavelength(z)
        else:
            self._pix_kms =  self.survey.pix_kms
            assert self._pix_kms is not None, 'Must provide pix width in kms or ang'      

    def _get_res_kms(self):
        #get spectrograph reslution in km/s
        #mean redshift
        z = self._mean_z()
        #assume specifying resolution in km/s takes precedence
        if self.survey.res_kms is not None:
            self._res_kms = self.survey.res_kms
        else:
            res_ang = self._lc / self.survey.resolution
            self._res_kms = res_ang * self.cosmo.velocity_from_wavelength(z)
        
    def _get_zq_bin(self):
        # given wavelength range covered in bin, compute central redshift and mean wavelength
        if not (self.lmin is None or self.lmax is None):
            # mean wavelenght of bin
            self._lc = np.sqrt(self.lmin * self.lmax)
            # redshift of quasar for which forest is centered in z
            lrc = np.sqrt(self.survey.lrmin * self.survey.lrmax)
            self._zq = (self._lc / lrc) - 1.0

    def _get_eval_mode(self,config):
        """Mode at which to evaluate power spectrum PS in weighting (PS / PS + PN)"""
        ACCEPTED_OPTIONS =  ['bao','p1d']
        measurement_type = config['control'].get('measurement type')
        if measurement_type == 'bao':
            self.kt_w_deg = 2.4 # ~ 0.035 h/Mpc
            self.kp_w_kms = 0.00035 # ~ 0.035 h/Mpc
        elif measurement_type == 'p1d':
            self.kt_w_deg = 7 # 0.1 h/Mpc
            self.kp_w_kms = 0.001 # ~ 0.1 h/Mpc
        else:
            raise ValueError('measurement type must be chosen from:',ACCEPTED_OPTIONS)

    def _mean_z(self):
        """ given wavelength range covered in bin, compute central redshift"""
        return np.sqrt(self.lmin * self.lmax) / self.cosmo.LYA_REST - 1.0

    def _get_redshift_depth(self):
        """Depth of redshift bin, in km/s"""
        c_kms = self.cosmo.SPEED_LIGHT
        L_kms = c_kms*np.log(self.lmax/self.lmin)

        return L_kms

    def _get_forest_length(self):
        """Length of Lya forest, in km/s"""
        c_kms = self.cosmo.SPEED_LIGHT
        lmax_forest = self.survey.lrmax * (1 + self._zq)
        lmin_forest = self.survey.lrmin * (1 + self._zq)
        Lq_kms = c_kms*np.log(lmax_forest/lmin_forest)

        return Lq_kms
    
    def _get_forest_wave(self):
        lmax_forest = self.survey.lrmax * (1 + self._zq)
        lmin_forest = self.survey.lrmin * (1 + self._zq)
        nbins = int((lmax_forest - lmin_forest) / self.survey.pix_ang)
        self._forest_wave = np.linspace(lmin_forest,lmax_forest,nbins)

    def get_survey_volume(self):
        z = self._mean_z()
        dkms_dhmpc = self.cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self.cosmo.distance_from_degrees(z)
        # survey volume in units of (Mpc/h)^3
        volume_degkms = self.survey.area_deg2 * self._get_redshift_depth()
        volume_mpch = volume_degkms * dhmpc_ddeg**2 / dkms_dhmpc

        return volume_mpch


    def _compute_p1d_kms(self,kp_kms):
        """1D Lya power spectrum in observed coordinates,
            smoothed with pixel width and resolution."""
        z = self._mean_z()
        # get P1D before smoothing
        p1d_kms = self.power_spec.compute_p1d_palanque2013(z,kp_kms)
        # smoothing (pixelization and resolution)
        kernel = self.spectrograph.smooth_kernel_kms(self._pix_kms,self._res_kms,kp_kms)
        p1d_kms *= (kernel**2)

        return p1d_kms

    def _compute_p3d_kms(self,kt_deg,kp_kms):
        """3D Lya power spectrum in observed coordinates. 
            Power smoothed with pixel width and resolution.
            If self._linear=True, it will ignore small scale correction."""
        z = self._mean_z()
        # transform km/s to Mpc/h
        dkms_dhmpc = self.cosmo.velocity_from_distance(z)
        kp_hmpc = kp_kms * dkms_dhmpc
        # transform degrees to Mpc/h
        dhmpc_ddeg = self.cosmo.distance_from_degrees(z)
        kt_hmpc = kt_deg / dhmpc_ddeg
        # compute polar decomposition
        k_hmpc = np.sqrt(kp_hmpc**2 + kt_hmpc**2)
        mu = kp_hmpc / (k_hmpc + 1.e-10) #cg: what's the point of this addition?
        # compute power in Mpc/h (from power_spectrum module)
        p3d_hmpc = self.power_spec.compute_p3d_hmpc(z,k_hmpc,mu
                                                  ,self._k_min_hmpc,
                                                  self._k_max_hmpc,
                                                  self._linear)
        # convert power to observed units
        p3d_degkms = p3d_hmpc * dkms_dhmpc / dhmpc_ddeg**2
        # convert resolution to kms

        # smoothing (pixelization and resolution)
        kernel=self.spectrograph.smooth_kernel_kms(self._pix_kms,self._res_kms,kp_kms)
        p3d_degkms *= kernel**2

        return p3d_degkms
    
    def compute_p3d_hmpc(self,k_hmpc,mu):
        """3D Lya power, in units of (Mpc/h)^3, including pixel width and
            resolution smoothing."""
        z = self._mean_z()
        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self.cosmo.velocity_from_distance(z)
        kt_deg = kt_hmpc * self.cosmo.distance_from_degrees(z)
        # get 3D power in units of observed coordinates
        power_degkms = self._compute_p3d_kms(kt_deg,kp_kms)
        # convert into units of (Mpc/h)^3
        power_hmpc = power_degkms * self.cosmo.distance_from_degrees(z)**2 / self.cosmo.velocity_from_distance(z)

        return power_hmpc

    def _compute_int_1(self,weights):
        """Integral 1 in McDonald & Eisenstein (2007).
            It represents an effective density of quasars, and it depends
            on the current value of the weights, that in turn depend on I1.
            We solve these iteratively (converges very fast)."""
        # quasar number density
        dkms_dz = self.cosmo.SPEED_LIGHT / (1 + self._zq)
        dndm_degdz = self.survey.get_qso_lum_func(self._zq,self.survey.maglist)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = self.survey.maglist[1] - self.survey.maglist[0]
        integrand = dndm_degkms * weights
        # weighted density of quasars
        #I1 = np.sum(dndm_degkms * weights) * dm
        #move to using cumsum so we can plot as a function of magnitude
        int_1 = np.cumsum(integrand) * dm

        return int_1

    def _compute_int_2(self,weights):
        """Integral 2 in McDonald & Eisenstein (2007).
            It is used to set the level of aliasing."""
        # quasar number density
        dndm_degdz = self.survey.get_qso_lum_func(self._zq,self.survey.maglist)
        dkms_dz = self.cosmo.SPEED_LIGHT / (1+self._zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = self.survey.maglist[1]-self.survey.maglist[0]
        integrand = dndm_degkms * weights**2
        int_2 = np.cumsum(integrand) * dm
        
        return int_2

    def _compute_int_3(self,weights):
        """Integral 3 in McDonald & Eisenstein (2007).
            It is used to set the effective noise power."""
        # pixel noise variance (dimensionless)
        pixel_var = self._get_var_m()
        # quasar number density
        dndm_degdz = self.survey.get_qso_lum_func(self._zq,self.survey.maglist)
        dkms_dz = self.cosmo.SPEED_LIGHT / (1 + self._zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = self.survey.maglist[1] - self.survey.maglist[0]
        integrand = dndm_degkms * weights**2 * pixel_var
        int_3 = np.cumsum(integrand) * dm

        return int_3

    def get_np_eff(self,weights):
        """Effective density of pixels in deg km/s, n_p^eff in McDonald & Eisenstein (2007).
            It is used in constructing the weights as a function of mag."""
        # get effective density of quasars
        int_1 = self._compute_int_1(weights)
        # number of pixels in a forest
        num_pix = self._get_forest_length() / self._pix_kms
        np_eff = int_1 * num_pix

        return np_eff

    def _get_var_m(self):
        """Noise pixel variance as a function of magnitude (dimensionless)"""

        z = self._mean_z()
        #pixel width in angstroms
        pix_ang = self._pix_kms / self.cosmo.velocity_from_wavelength(z)

        # noise rms per pixel
        noise_rms = np.zeros_like(self.survey.maglist)
        for i,m in enumerate(self.survey.maglist):
            noise_rms[i] = self.spectrograph.get_pixel_rms_noise(m,self._zq,
                                                                 self._lc,pix_ang,
                                                                 self.survey.num_exp)
        noise_var = noise_rms**2

        return noise_var

    def _compute_noise_power_m(self,weights):
        """Effective noise power as a function of magnitude,
            referred to as P_N(m) in McDonald & Eisenstein (2007).
            Note this is a 3D power, not 1D, and it is used in 
            constructing the weights as a function of magnitude."""
        # pixel noise variance (dimensionless)
        pixel_var = self._get_var_m()
        # 3D effective density of pixels
        neff = self.get_np_eff(weights)
        noise_power = pixel_var / neff

        return noise_power

    def _compute_weights(self,weights):
        """Compute new weights as a function of magnitude, using P3D.
            This version of computing the weights is closer to the one
            described in McDonald & Eisenstein (2007)."""
        # 3D noise power as a function of magnitude
        noise_power = self._compute_noise_power_m(weights)
        # effective 3D density of quasars
        int_1 = self._compute_int_1(weights)
        int_2 = self._compute_int_2(weights)
        # 2D density of lines of sight (units of 1/deg^2)
        aliasing = int_2 / (int_1**2 * self._get_forest_length())
        # weights include aliasing as signal
        signal_power = self._p3d_w + self._p1d_w * aliasing
        weights = signal_power / (signal_power + noise_power)

        return weights

    #if this is the same why not just use this as default?
    def _compute_weights2(self,weights):
        """Compute new weights as a function of magnitude, using pixel var.
            This version of computing the weights is closer to the c++ code
            developed by Pat McDonald and used in official forecasts of DESI.
            It gives identical results to the Weights1 above."""
        # noise pixel variance as a function of magnitude (dimensionless)
        noise_var = self._get_var_m()
        # pixel variance from P1D (dimensionless)
        pix_var_1d = self._p1d_w / self._pix_kms
        # effective 3D density of pixels
        neff = self.get_np_eff(weights)
        # pixel variance from P3D (dimensionless)
        pix_var_3d = self._p3d_w * neff
        # signal variance (include P1D and P3D)
        sig_var = pix_var_3d + pix_var_1d
        weights = sig_var / (sig_var + noise_var)

        return weights

    def _initialise_weights(self):
        """Compute initial weights as a function of magnitude, using only
            P1D and noise variance."""
        # noise pixel variance as a function of magnitude (dimensionless)
        noise_var = self._get_var_m()
        # pixel variance from P1D (dimensionless)
        pix_var_1d = self._p1d_w / self._pix_kms
        weights = pix_var_1d / (pix_var_1d + noise_var)

        return weights

    def compute_weights(self):
        """Compute weights as a function of magnitude. 
            We do it iteratively since the weights depend on I1, and 
            I1 depends on the weights."""
        # compute first weights using only 1D and noise variance
        weights = self._initialise_weights()
        num_iter = 3
        for i in range(num_iter):
            if self.verbose > 1:
                print(i,'<w>',np.mean(weights))
            weights = self._compute_weights(weights)
            #weights = self.Weights2(weights)

        return weights

    def compute_eff_density_and_noise(self):
        """Compute effective density of lines of sight and eff. noise power.
            Terms Pw2D and PN_eff in McDonald & Eisenstein (2007)."""

        #if self.verbose>0:
        #    print('mean wave, mean rest wave, z qso =',lc,lrc,zq)

        # The code below is closer to the method described in the publication,
        # but the c++ code by Pat is more complicated. 
        # The code below evaluates all the quantities at the central redshift
        # of the bin, and uses a single quasar redshift assuming that all pixels
        # in the bin have restframe wavelength of the center of the forest.
        # Pat's code, instead, computes an average over both redshift of 
        # absorption and over quasar redshift.

        weights = self.compute_weights()

        # given weights, compute integrals in McDonald & Eisenstein (2007)
        int_1 = self._compute_int_1(weights)
        int_2 = self._compute_int_2(weights)
        int_3 = self._compute_int_3(weights)

        # length of forest in km/s
        forest_length = self._get_forest_length()
        # length of pixel in km/s
        pixel_length = self._pix_kms
        # Pw2D in McDonald & Eisenstein (2007)
        self._aliasing_weights = int_2 / (int_1**2 * forest_length)
        # PNeff in McDonald & Eisenstein (2007)
        self._effective_noise_power = int_3 * pixel_length / (int_1**2 * forest_length)

    def _compute_total_3d_power(self):
        """Sum of 3D Lya power, aliasing and effective noise power. 
            If Pw2D or PN_eff are not passed, it will compute them"""
        # figure out mean redshift of the mini-survey
        z = self._mean_z()
        # previously computed p2wd and pn_eff.
        total_power = self._p3d_w + self._aliasing_weights * self._p1d_w + self._effective_noise_power

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
        dkms_dmpch = self.cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self.cosmo.distance_from_degrees(z)

        # get total power in units of observed coordinates 
        # To-do: get P_total(mag)
        total_power_degkms = self._compute_total_3d_power()
        # convert into units of (Mpc/h)^3
        total_power_hmpc = total_power_degkms * dhmpc_ddeg**2 / dkms_dmpch
        # survey volume in units of (Mpc/h)^3
        volume_degkms = self.survey.area_deg2 * self._get_redshift_depth()
        volume_mpch = volume_degkms * dhmpc_ddeg**2 / dkms_dmpch
        # based on Eq 8 in Seo & Eisenstein (2003), but note that here we
        # use 0 < mu < 1 and they used -1 < mu < 1
        num_modes = volume_mpch * k_hmpc**2 * self._dk * self.dmu / 4 * np.pi**2
        power_variance = 2 * total_power_hmpc**2 / num_modes

        #If not per magnitude, return power var for mmax only. 
        # Otherwise as a function of m input.
        if not self.per_mag:
            power_variance = power_variance[-1]

        return power_variance
    
    def compute_qso_auto_power(self):
        return
    
    def compute_cross_power_variance(self,k_hmpc,mu
                        ):
        """Variance of Lya-QSO cross-power, in units of (Mpc/h)^3.
            From eq.34 of McQuinn and White (2011).
            Note that here -1 < mu < 1.
           """
        
        return 

