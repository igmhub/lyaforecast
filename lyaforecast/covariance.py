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
        self._linear = _properties.get('linear_power')


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
        self._dmu = np.diff(mu_edges)[0]

        # These will be dependent on redshift bins, 
        # which are passed during foreacast run (for now).
        self.lmin = None
        self.lmax = None
        self._aliasing_weights = None
        self._effecitve_noise_power = None

        # verbosity level - I'll remove this later.
        self.verbose = 1

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
        l = np.sqrt(self.lmin*self.lmax)
        z = l/self.cosmo.LYA_REST-1.0
        return z

    def _get_redshift_depth(self):
        """Depth of redshift bin, in km/s"""
        c_kms = self.cosmo.SPEED_LIGHT
        L_kms = c_kms*np.log(self.lmax/self.lmin)
        return L_kms

    def _get_forest_length(self):
        """Length of Lya forest, in km/s"""
        c_kms = self.cosmo.SPEED_LIGHT
        Lq_kms = c_kms*np.log(self.survey.lrmax/self.survey.lrmin)
        return Lq_kms

    def _compute_p1d_kms(self,kp_kms):
        """1D Lya power spectrum in observed coordinates,
            smoothed with pixel width and resolution."""
        z = self._mean_z()
        # get P1D before smoothing
        P1D_kms = self.power_spec.compute_p1d_palanque2013(z,kp_kms)
        # smoothing (pixelization and resolution)
        kernel = self.spectrograph.smooth_kernel_kms(self.survey.pix_kms,self.survey.res_kms,kp_kms)
        P1D_kms *= (kernel**2)
        return P1D_kms

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
        # smoothing (pixelization and resolution)
        kernel=self.spectrograph.smooth_kernel_kms(self.survey.pix_kms,self.survey.res_kms,kp_kms)
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


    def _compute_total_3d_power(self,kt_deg,kp_kms):
        """Sum of 3D Lya power, aliasing and effective noise power. 
            If Pw2D or PN_eff are not passed, it will compute them"""
        # figure out mean redshift of the mini-survey
        z = self._mean_z()
        # signal
        p3d = self._compute_p3d_kms(kt_deg,kp_kms)
        # P1D for aliasing
        p1d = self._compute_p1d_kms(kp_kms)
        # previously computed p2wd and pn_eff.
        total_power = p3d + self._aliasing_weights * p1d + self._effecitve_noise_power
        return total_power

    def compute_3d_power_variance(self,k_hmpc,mu
                        ):
        """Variance of 3D Lya power, in units of (Mpc/h)^3.
            Note that here 0 < mu < 1.
            If Pw2D or PN_eff are not passed, it will compute them"""
        
        #We should move to computing this in a vectorised fashion, rather than 
        #iterating over mu/k values. Then I wouldn't have to call the mu/k values
        # again.
        z = self._mean_z()

        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        dkms_dhmpc = self.cosmo.velocity_from_distance(z)
        kp_kms = kp_hmpc / dkms_dhmpc
        dhmpc_ddeg = self.cosmo.distance_from_degrees(z)
        kt_deg = kt_hmpc * dhmpc_ddeg

        # get total power in units of observed coordinates
        total_power_degkms = self._compute_total_3d_power(kt_deg,kp_kms)
        # convert into units of (Mpc/h)^3
        total_power_hmpc = total_power_degkms * dhmpc_ddeg**2 / dkms_dhmpc
        # survey volume in units of (Mpc/h)^3
        volume_degkms = self.survey.area_deg2 * self._get_redshift_depth()
        volume_hmpc = volume_degkms * dhmpc_ddeg**2 / dkms_dhmpc
        # based on Eq 8 in Seo & Eisenstein (2003), but note that here we
        # use 0 < mu < 1 and they used -1 < mu < 1
        num_modes = volume_hmpc * k_hmpc**2 * self._dk * self._dmu / 2 * np.pi**2
        power_variance = 2 * total_power_hmpc**2 / num_modes

        return power_variance

    def _compute_int_1(self,zq,mags,weights):
        """Integral 1 in McDonald & Eisenstein (2007).
            It represents an effective density of quasars, and it depends
            on the current value of the weights, that in turn depend on I1.
            We solve these iteratively (converges very fast)."""
        # quasar number density
        dkms_dz = self.cosmo.SPEED_LIGHT / (1 + zq)
        dndm_degdz = self.survey.get_qso_lum_func(zq,mags)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = mags[1] - mags[0]
        # weighted density of quasars
        I1 = np.sum(dndm_degkms*weights)*dm
        if self.verbose > 2:
            print('dkms_dz',dkms_dz)
            print('dndm_degdz',dndm_degdz)
            print('dndm_degkms',dndm_degkms)
            print('mags',mags)
            print('integrant',dndm_degkms*weights)
            print('I1',I1)
        return I1

    def _compute_int_2(self,zq,mags,weights):
        """Integral 2 in McDonald & Eisenstein (2007).
            It is used to set the level of aliasing."""
        # quasar number density
        dndm_degdz = self.survey.get_qso_lum_func(zq,mags)
        dkms_dz = self.cosmo.SPEED_LIGHT / (1+zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = mags[1]-mags[0]
        I2 = np.sum(dndm_degkms*weights*weights)*dm
        return I2

    def _compute_int_3(self,zq,lc,mags,weights):
        """Integral 3 in McDonald & Eisenstein (2007).
            It is used to set the effective noise power."""
        # pixel noise variance (dimensionless)
        varN = self._get_var_m(zq,lc,mags)
        # quasar number density
        dndm_degdz = self.survey.get_qso_lum_func(zq,mags)
        dkms_dz = self.cosmo.SPEED_LIGHT / (1+zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = mags[1]-mags[0]
        I3 = np.sum(dndm_degkms*weights*weights*varN)*dm
        return I3

    def _get_np_eff(self,zq,mags,weights):
        """Effective density of pixels in deg km/s, n_p^eff in McDonald & Eisenstein (2007).
            It is used in constructing the weights as a function of mag."""
        # get effective density of quasars
        I1 = self._compute_int_1(zq,mags,weights)
        # number of pixels in a forest
        Npix = self._get_forest_length() / self.survey.pix_kms
        np_eff = I1 * Npix
        if self.verbose > 2:
            print('I1',I1)
            print('Npix',Npix)
            print('np_eff',np_eff)
        return np_eff

    def _get_var_m(self,zq,lc,mags):
        """Noise pixel variance as a function of magnitude (dimensionless)"""
        z = (lc / self.cosmo.LYA_REST) - 1
        # pixel width in Angstroms
        pix_ang = self.survey.pix_kms / self.cosmo.velocity_from_wavelength(z)
        # noise rms per pixel
        noise_rms = np.zeros_like(mags)
        for i,m in enumerate(mags):
            noise_rms[i] = self.spectrograph.get_pixel_rms_noise(m,zq,lc,pix_ang)
        noise_var = noise_rms**2
        return noise_var

    def _compute_noise_power_m(self,zq,lc,mags,weights):
        """Effective noise power as a function of magnitude,
            referred to as P_N(m) in McDonald & Eisenstein (2007).
            Note this is a 3D power, not 1D, and it is used in 
            constructing the weights as a function of magnitude."""
        # pixel noise variance (dimensionless)
        varN = self._get_var_m(zq,lc,mags)
        # 3D effective density of pixels
        neff = self._get_np_eff(zq,mags,weights)
        PN = varN / neff
        if self.verbose > 2:
            print('noise variance',varN)
            print('neff',neff)
            print('PN',PN)
        return PN

    def _weights1(self,P3D_degkms,P1D_kms,zq,lc,mags,weights):
        """Compute new weights as a function of magnitude, using P3D.
            This version of computing the weights is closer to the one
            described in McDonald & Eisenstein (2007)."""
        # 3D noise power as a function of magnitude
        PN = self._compute_noise_power_m(zq,lc,mags,weights)
        # effective 3D density of quasars
        I1 = self._compute_int_1(zq,mags,weights)
        # 2D density of lines of sight (units of 1/deg^2)
        n2D_los = I1 * self._get_forest_length()
        # weights include aliasing as signal
        PS = P3D_degkms + P1D_kms / n2D_los
        weights = PS/(PS+PN)
        if self.verbose > 2:
            print('P3D',P3D_degkms)
            print('P1D',P1D_kms)
            print('PN',PN)
            print('I1',I1)
            print('n2D_los',n2D_los)
            print('PA',n2D_los*P1D_kms)
            print('PS',PS)
            print('weights',weights)
        return weights

    #if this is the same why not just use this as default?
    def _weights2(self,P3D_degkms,P1D_kms,zq,lc,mags,weights):
        """Compute new weights as a function of magnitude, using pixel var.
            This version of computing the weights is closer to the c++ code
            developed by Pat McDonald and used in official forecasts of DESI.
            It gives identical results to the Weights1 above."""
        # noise pixel variance as a function of magnitude (dimensionless)
        varN = self._get_var_m(zq,lc,mags)
        # pixel variance from P1D (dimensionless)
        var1D = P1D_kms / self.survey.pix_kms
        # effective 3D density of pixels
        neff = self._get_np_eff(zq,mags,weights)
        # pixel variance from P3D (dimensionless)
        var3D = P3D_degkms * neff
        # signal variance (include P1D and P3D)
        varS = var3D + var1D
        weights = varS/(varS+varN)
        if self.verbose > 2:
            print('P3D',P3D_degkms)
            print('P1D',P1D_kms)
            print('varN',varN)
            print('var1D',var1D)
            print('neff',neff)
            print('var3D',var3D)
            print('varS',varS)
            print('weights',weights)
        return weights

    def _initialise_weights(self,P1D_kms,zq,lc,mags):
        """Compute initial weights as a function of magnitude, using only
            P1D and noise variance."""
        # noise pixel variance as a function of magnitude (dimensionless)
        varN = self._get_var_m(zq,lc,mags)
        # pixel variance from P1D (dimensionless)
        var1D = P1D_kms / self.survey.pix_kms
        weights = var1D/(var1D+varN)
        if self.verbose > 2:
            print('P1D',P1D_kms)
            print('varN',varN)
            print('noise_rms',np.sqrt(varN))
            print('var1D',var1D)
            print('weights',weights)
        return weights

    def _compute_weights(self,P3D_degkms,P1D_kms,zq,lc,mags,Niter=3):
        """Compute weights as a function of magnitude. 
            We do it iteratively since the weights depend on I1, and 
            I1 depends on the weights."""
        # compute first weights using only 1D and noise variance
        weights = self._initialise_weights(P1D_kms,zq,lc,mags)
        for i in range(Niter):
            if self.verbose > 1:
                print(i,'<w>',np.mean(weights))
            weights = self._weights1(P3D_degkms,P1D_kms,zq,lc,mags,weights)
            #weights = self.Weights2(P3D_degkms,P1D_kms,zq,lc,mags,weights)
            if self.verbose > 2:
                print('weights',weights)
        return weights

    def compute_eff_density_and_noise(self):
        """Compute effective density of lines of sight and eff. noise power.
            Terms Pw2D and PN_eff in McDonald & Eisenstein (2007)."""
        # mean wavelenght of bin
        lc = np.sqrt(self.lmin * self.lmax)
        # redshift of quasar for which forest is centered in z
        lrc = np.sqrt(self.survey.lrmin * self.survey.lrmax)
        zq = (lc / lrc) - 1.0
        if self.verbose>0:
            print('mean wave, mean rest wave, z qso =',lc,lrc,zq)
        
        # evaluate P1D and P3D for weighting
        #the range of k values here will be the same as in the analysis.
        #Calum: these are computed at one specific kt, kp?
        #Calum: I'll leave it for now, since it's fast to compute p3d/p1d.
        p3d_w = self._compute_p3d_kms(self.kt_w_deg,self.kp_w_kms)
        p1d_w = self._compute_p1d_kms(self.kp_w_kms)

        # The code below is closer to the method described in the publication,
        # but the c++ code by Pat is more complicated. 
        # The code below evaluates all the quantities at the central redshift
        # of the bin, and uses a single quasar redshift assuming that all pixels
        # in the bin have restframe wavelength of the center of the forest.
        # Pat's code, instead, computes an average over both redshift of 
        # absorption and over quasar redshift.

        # set range of magnitudes used (same as in c++ code)
        mmin = self.survey.mag_min
        mmax = self.survey.mag_max
        # same binning as in c++ code
        dm = 0.025
        n_mag_bins = int((mmax-mmin)/dm)
        mags = np.linspace(mmin,mmax,n_mag_bins)
        # get weights (iteratively)
        weights = self._compute_weights(p3d_w,p1d_w,zq,lc,mags)

        # given weights, compute integrals in McDonald & Eisenstein (2007)
        int_1 = self._compute_int_1(zq,mags,weights)
        int_2 = self._compute_int_2(zq,mags,weights)
        int_3 = self._compute_int_3(zq,lc,mags,weights)

        # length of forest in km/s
        forest_length = self._get_forest_length()
        # length of pixel in km/s
        pixel_length = self.survey.pix_kms
        # effective 3D density of pixels
        self.np_eff = int_1*forest_length / pixel_length
        # Pw2D in McDonald & Eisenstein (2007)
        self._aliasing_weights = int_2 / (int_1 * forest_length)
        # PNeff in McDonald & Eisenstein (2007)
        self._effecitve_noise_power = int_3 * pixel_length / (int_1**2 * forest_length)