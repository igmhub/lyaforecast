"""Class to handle computation of weights for various tracers"""
import numpy as np 

class Weights:
    OPTIONS = ['lya','qso']
    def __init__(self,config,survey,cosmo,powerspec,spectrograph,
                 forest_length,pixel_length,resolution,lambda_mean,z_bin,z_qso,
                 zmin, zmax):
        
        self.weights = None
        self._config = config
        self._survey = survey
        self._cosmo = cosmo
        self._spectrograph = spectrograph
        self._powerspec = powerspec
        self._forest_length = forest_length
        self._pix_kms = pixel_length
        self._res_kms = resolution
        self._lambda_mean = lambda_mean
        self._z_bin = z_bin
        self._zq = z_qso
        self._zmin = zmin
        self._zmax = zmax

        #get k values to evaluate p_w ,z,kt_deg,kp_kms,res_kms,pix_kms,whic
        self._get_eval_mode()

    def _get_eval_mode(self):
        """Mode at which to evaluate power spectrum PS in weighting (PS / PS + PN)"""
        ACCEPTED_OPTIONS =  ['bao','p1d']
        measurement_type = self._config['control'].get('measurement type')
        if measurement_type == 'bao':
            self.kt_w_deg = 2.4 # ~ 0.035 h/Mpc
            self.kp_w_kms = 0.00035 # ~ 0.035 h/Mpc
        elif measurement_type == 'p1d':
            self.kt_w_deg = 7 # 0.1 h/Mpc
            self.kp_w_kms = 0.001 # ~ 0.1 h/Mpc
        else:
            raise ValueError('measurement type must be chosen from:',ACCEPTED_OPTIONS)
    
    def compute_weights(self,which='lya'):
        """Compute weights as a function of magnitude. 
            We do it iteratively since the weights depend on I1, and 
            I1 depends on the weights."""
        
        #Tracer is internally set, but just for posterity.
        assert which in self.OPTIONS, 'Tracer option invalid'

        #pre-compute power spectra for weighting
        self._p3d_w = self._powerspec.compute_p3d_kms(self._z_bin,self.kt_w_deg
                                                      ,self.kp_w_kms,self._res_kms,
                                                      self._pix_kms,which)
        self._p1d_w = self._powerspec.compute_p1d_kms(self._z_bin,self.kp_w_kms,
                                                      self._res_kms,self._pix_kms)
        
        if which=='lya':
            self._compute_weights = self._compute_weights_lya
            self._initialise_weights = self._initialise_weights_lya
        elif which=='qso':
            self._compute_weights = self._compute_weights_qso
            self._initialise_weights = self._initialise_weights_qso
        
        # compute first weights using only 1D and noise variance
        weights = self._initialise_weights()
        num_iter = 3
        for i in range(num_iter):
            weights = self._compute_weights(weights)

        return weights
    
    def _initialise_weights_lya(self):
        """Compute initial weights as a function of magnitude, using only
            P1D and noise variance."""
        # noise pixel variance as a function of magnitude (dimensionless)
        noise_var = self._get_pix_var_m()
        # pixel variance from P1D (dimensionless)
        pix_var_1d = self._p1d_w / self._pix_kms
        weights = pix_var_1d / (pix_var_1d + noise_var)

        return weights
    
    def _compute_weights_lya(self,weights):
        """Compute new weights as a function of magnitude, using P3D.
            This version of computing the weights is closer to the one
            described in McDonald & Eisenstein (2007)."""
        # 3D noise power as a function of magnitude
        noise_power = self._compute_noise_power_m(weights)
        # effective 3D density of quasars
        int_1 = self.compute_int_1(weights)
        int_2 = self.compute_int_2(weights)
        # 2D density of lines of sight (units of 1/deg^2)
        aliasing = int_2 / (int_1**2 * self._forest_length)
        # weights include aliasing as signal
        signal_power = self._p3d_w + self._p1d_w * aliasing
        weights = signal_power / (signal_power + noise_power)

        return weights
    
    def _initialise_weights_qso(self):
        """Compute initial quasar weights as a function of magnitude."""

        return np.ones_like(self._survey.maglist)

    # We're not computing an effective density, just n_3d, but easier to do it in this module.
    def _compute_weights_qso(self,weights):
        """Compute quasar density as a function of maximum magnitude"""

        zlist = np.linspace(self._zmin,self._zmax,self._survey.nzq)
        dz = zlist[1] - zlist[0]
        mags = self._survey.maglist
        #qso density / deg^2 (m)
        # dn_dzdm = np.zeros_like(mags)
        # for z in zlist:
        #      dn_dzdm += self._survey.get_qso_lum_func(z,mags) * dz
        dn_dzdm = self._survey.get_qso_lum_func(self._z_bin,mags)
        dkms_dz = self._cosmo.SPEED_LIGHT / (1 + self._z_bin)
        dn_dmdkms = dn_dzdm / dkms_dz

        dm = mags[1] - mags[0]
        dn_dkms = np.cumsum(dn_dmdkms * dm)

        return dn_dkms

    def compute_int_1(self,weights):
        """Integral 1 in McDonald & Eisenstein (2007).
            It represents an effective density of quasars, and it depends
            on the current value of the weights, that in turn depend on I1.
            We solve these iteratively (converges very fast)."""
        # quasar number density
        dkms_dz = self._cosmo.SPEED_LIGHT / (1 + self._zq)
        dndm_degdz = self._survey.get_qso_lum_func(self._zq,self._survey.maglist)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = self._survey.maglist[1] - self._survey.maglist[0]
        integrand = dndm_degkms * weights
        # weighted density of quasars
        #I1 = np.sum(dndm_degkms * weights) * dm
        #move to using cumsum so we can plot as a function of magnitude
        int_1 = np.cumsum(integrand*dm)

        return int_1

    def compute_int_2(self,weights):
        """Integral 2 in McDonald & Eisenstein (2007).
            It is used to set the level of aliasing."""
        # quasar number density
        dndm_degdz = self._survey.get_qso_lum_func(self._zq,self._survey.maglist)
        dkms_dz = self._cosmo.SPEED_LIGHT / (1+self._zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = self._survey.maglist[1]-self._survey.maglist[0]
        integrand = dndm_degkms * weights**2
        int_2 = np.cumsum(integrand*dm)
        
        return int_2

    def compute_int_3(self,weights):
        """Integral 3 in McDonald & Eisenstein (2007).
            It is used to set the effective noise power."""
        # pixel noise variance (dimensionless)
        pixel_var = self._get_pix_var_m()
        # quasar number density
        dndm_degdz = self._survey.get_qso_lum_func(self._zq,self._survey.maglist)
        dkms_dz = self._cosmo.SPEED_LIGHT / (1 + self._zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = self._survey.maglist[1] - self._survey.maglist[0]
        integrand = dndm_degkms * weights**2 * pixel_var
        int_3 = np.cumsum(integrand*dm)

        return int_3

    def get_np_eff_lya(self,weights):
        """Effective density of pixels in deg km/s, n_p^eff in McDonald & Eisenstein (2007).
            It is used in constructing the weights as a function of mag."""
        # get effective density of quasars
        int_1 = self.compute_int_1(weights)
        # number of pixels in a forest
        num_pix = self._forest_length / self._pix_kms
        np_eff = int_1 * num_pix

        return np_eff

    def get_np_eff_qso(self,weights):
        """Effective density of quasars in deg km/s.
            It is used in constructing the weights as a function of mag."""
        # get effective density of quasars
        int_1 = self._compute_weights_qso()
        # number of pixels in a forest
        num_pix = self._forest_length / self._pix_kms
        np_eff = int_1 * num_pix

        return np_eff
    
    def _get_pix_var_m(self):
        """Noise pixel variance as a function of magnitude (dimensionless)"""

        #pixel width in angstroms
        pix_ang = self._pix_kms / self._cosmo.velocity_from_wavelength(self._z_bin)

        # noise rms per pixel
        noise_rms = np.zeros_like(self._survey.maglist)
        for i,m in enumerate(self._survey.maglist):
            noise_rms[i] = self._spectrograph.get_pixel_rms_noise(m,self._zq,
                                                                 self._lambda_mean,pix_ang,
                                                                 self._survey.num_exp)
        noise_var = noise_rms**2

        return noise_var

    def _compute_noise_power_m(self,weights):
        """Effective noise power as a function of magnitude,
            referred to as P_N(m) in McDonald & Eisenstein (2007).
            Note this is a 3D power, not 1D, and it is used in 
            constructing the weights as a function of magnitude."""
        # pixel noise variance (dimensionless)
        pixel_var = self._get_pix_var_m()
        # 3D effective density of pixels
        neff = self.get_np_eff_lya(weights)
        noise_power = pixel_var / neff

        return noise_power