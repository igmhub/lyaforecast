import numpy as np
from lyaforecast.analystic_biases import AnalyticBias

class PowerSpectrum:
    """Class to store power spectra models.
        Should only be used at the level of Fisher forecasts.
        Uses CAMB to generate linear power, and McDonald (2003) for Lya stuff.
        All units internally are in h/Mpc."""

    def __init__(self, cosmo):
        #define cosmology class
        self.cosmo = cosmo
        #isotropic linear matter power
        #self.linear_power = cosmo.get_pk_lin_interp(self.kmin,self.kmax,1000)
        self.bias = AnalyticBias()

    def compute_linear_power_evol(self,z,k_hmpc,k_min,k_max):
        """Scale linear power, assuming EdS scale with redshift"""
        if z<1.8:
            #raise ValueError('Can not use EdS to go below z = 1.8')
            print('Warning, going below z = 1.8 with EdS power scaling')
        if self.cosmo.z_ref<1.8:
            raise ValueError('Can not have z_ref below 1.8, input:',self.cosmo.z_ref)
        pk_zref = self.cosmo.get_pk_lin(k_hmpc,k_min,k_max)
        eds = ((1+self.cosmo.z_ref)/(1+z))**2
        return pk_zref * eds

    def compute_p3d_hmpc(self,z,k_hmpc,mu,k_min,k_max,
                         linear=False,which='lya'):
        """3D power spectrum P_F(z,k,mu). 
        If linear=True, it will ignore small scale correction."""
        # get linear power at zrefs
        k = np.fmax(k_hmpc,k_min)
        k = np.fmin(k,k_max)
        # compute redshift-evolved linear matter power spectrum
        pk_zref = self.compute_linear_power_evol(z,k_hmpc,k_min,k_max)
        # get flux scale-dependent biasing (or only linear term)
        b = self.bias.compute_bias(z,k,mu,linear,which)

        return pk_zref * b

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
        dkms_dhmpc = self.cosmo.velocity_from_distance(z)
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