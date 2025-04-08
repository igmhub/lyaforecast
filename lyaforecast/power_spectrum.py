import numpy as np
from lyaforecast.analytic_biases import AnalyticBias

class PowerSpectrum:
    """Class to store power spectra models.
        Should only be used at the level of Fisher forecasts.
        Uses CAMB to generate linear power, and McDonald (2003) for Lya stuff.
        All units internally are in h/Mpc."""

    def __init__(self, config, cosmo, spectrograph):
        #define cosmology class
        self._cosmo = cosmo
        self._spectrograph = spectrograph
        self._growth_rate = self._cosmo.growth_rate
        #get analytical biases
        self.bias = AnalyticBias(self._cosmo)

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

        #grids for evaluating power spectrum
        self.k = np.linspace(self._k_min_hmpc,self._k_max_hmpc, self._num_k_bins)
        self.logk = np.log(self.k)
        self.dk = self.k[1] - self.k[0]
        self.dlogk = self.dk / self.k

        #bin edges
        mu_edges = np.linspace(self._mu_min,self._mu_max, self._num_mu_bins)
        self.mu = mu_edges[:-1] + np.diff(mu_edges)/2     
        self.dmu = np.diff(mu_edges)[0]

        # peak-only
        self._bao_only = config['control'].getboolean('bao only')

    def compute_linear_power_evol(self,z,k_hmpc):
        """Scale linear power, assuming EdS scale with redshift"""
        if z<1.8:
            print('Warning, going below z = 1.8 with EdS power scaling')
        if self._cosmo.z_ref<1.8:
            raise ValueError('Can not have z_ref below 1.8, input:',self._cosmo.z_ref)
        
        if self._bao_only:
            pk_zref = self._cosmo.get_pk_lin_peak(k_hmpc)
        else:
            pk_zref = self._cosmo.get_pk_lin(k_hmpc,self._k_min_hmpc,self._k_max_hmpc)

        eds = ((1+self._cosmo.z_ref)/(1+z))**2

        return pk_zref * eds
    
    def compute_p1d_kms(self,z,kp_kms,res_kms,pix_kms):
        """1D Lya power spectrum in observed coordinates,
            smoothed with pixel width and resolution."""
        # get P1D before smoothing
        p1d_kms = self.compute_p1d_palanque2013(z,kp_kms)
        # smoothing (pixelization and resolution)
        kernel = self._spectrograph.smooth_kernel_kms(pix_kms,res_kms,kp_kms)
        p1d_kms *= (kernel**2)

        return p1d_kms

    def compute_p3d_kms(self,z,kt_deg,kp_kms,res_kms,pix_kms,which='lya'):
        """3D Lya power spectrum in observed coordinates. 
            Power smoothed with pixel width and resolution.
            If self._linear=True, it will ignore small scale correction."""
        # transform km/s to Mpc/h
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        kp_hmpc = kp_kms * dkms_dhmpc
        # transform degrees to Mpc/h
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)
        kt_hmpc = kt_deg / dhmpc_ddeg
        # compute polar decomposition
        k_hmpc = np.sqrt(kp_hmpc**2 + kt_hmpc**2)
        mu = kp_hmpc / (k_hmpc + 1.e-10)

        # compute power in Mpc/h (from power_spectrum module)
        p3d_hmpc = self.compute_p3d_hmpc(z,k_hmpc,mu,which)
        # convert power to observed units
        p3d_degkms = p3d_hmpc * dkms_dhmpc / dhmpc_ddeg**2
        # convert resolution to kms

        # smoothing (pixelization and resolution)
        kernel = self._spectrograph.smooth_kernel_kms(pix_kms,res_kms,kp_kms)
        p3d_degkms *= kernel**2

        return p3d_degkms

    def compute_p3d_hmpc(self,z,k_hmpc,mu,which='lya'):
        """3D power spectrum P_F(z,k,mu). 
        If linear=True, it will ignore small scale correction."""
        # get linear power at zrefs
        k = np.fmax(k_hmpc,self._k_min_hmpc)
        k = np.fmin(k,self._k_max_hmpc)
        # compute redshift-evolved linear matter power spectrum
        pk_zref = self.compute_linear_power_evol(z,k_hmpc)
        # get flux scale-dependent biasing (or only linear term)
        b = self.bias.compute_bias(z,k,mu,self._linear,which)

        return pk_zref * b
    
    def compute_p3d_hmpc_smooth(self,z,k_hmpc,mu,pix_kms,res_kms,which='lya'):
        """Smooth power spectrum (convert to k space then back.)"""

        #conversions
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        kp_hmpc = k_hmpc * mu
        kp_kms = kp_hmpc / dkms_dhmpc    

        # get linear power at zrefs
        p3d_hmpc = self.compute_p3d_hmpc(z,k_hmpc,mu,which)
        p3d_degkms = p3d_hmpc * dkms_dhmpc / dhmpc_ddeg**2

        kernel = self._spectrograph.smooth_kernel_kms(pix_kms,res_kms,kp_kms)
        p3d_degkms *= kernel**2

        p3d_hmpc = p3d_degkms * dhmpc_ddeg**2 / dkms_dhmpc

        return p3d_hmpc


    def compute_p1d_palanque2013(self,z,k_kms):
        """Fitting formula for 1D P(z,k) from Palanque-Delabrouille et al. (2013).
            Wavenumbers and power in units of km/s. Corrected to be flat at low-k"""
        # numbers from Palanque-Delabrouille (2013)
        A_F = 0.064
        n_F = -2.55
        alpha_F = -0.1
        B_F = 3.55
        beta_F = -0.28
        k0 = 0.009
        z0 = 3.0
        n_F_z = n_F + beta_F * np.log((1+z)/(1+z0))
        # this function would go to 0 at low k, instead of flat power
        k_min=k0*np.exp((-0.5*n_F_z-1)/alpha_F)
        k_kms = np.fmax(k_kms,k_min)
        exp1 = 3 + n_F_z + alpha_F * np.log(k_kms/k0)
        toret = np.pi * A_F / k0 * pow(k_kms/k0, exp1-1) * pow((1+z)/(1+z0), B_F)
        return toret
    
    #currently un-used
    def compute_p1d_hmpc(self,z,k_hmpc,res_hmpc=None,pix_hmpc=None):
        """Analytical P1D, in units of h/Mpc instead of km/s."""
        # transform to km/s
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        k_kms = k_hmpc / dkms_dhmpc
        # get analytical P1D from Palanque-Delabrouille (2013)
        power_kms = self.compute_p1d_palanque2013(z,k_kms)
        power_hmpc = power_kms / dkms_dhmpc
        if res_hmpc:
            # smooth with Gaussian
            power_hmpc *= np.exp(-pow(k_hmpc*res_hmpc,2))
        if pix_hmpc:
            # smooth with Top Hat
            kpix = np.fmax(k_hmpc*pix_hmpc,1.e-5)
            power_hmpc *= pow(np.sin(kpix/2)/(kpix/2),2)

        return power_hmpc