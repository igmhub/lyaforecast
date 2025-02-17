import numpy as np

class McDonald2003:
    """Class to store analytic formulae for biases of Lya P3D, including non-linear corrections.
        These will later be handled by ForestFlow, currently parameter values are out-of-date."""

    def _get_non_linear_corr(self,k_hMpc):
        """Non-linear correction"""
        k_nl = 6.40
        alpha_nl = 0.569 
        return pow(k_hMpc / k_nl, alpha_nl)

    def _get_pressure_corr(self,k_hMpc):
        """Pressure correction"""
        k_p = 15.3
        alpha_p = 2.01
        return pow(k_hMpc / k_p, alpha_p)

    def _get_non_linear_velo(self,k_hMpc, mu):
        """Non-linear velocities"""
        k_v0 = 1.220
        alpha_v = 1.50
        k_vv = 0.923
        alpha_vv = 0.451
        kpar = k_hMpc * mu
        kv = k_v0 * pow(1 + k_hMpc / k_vv, alpha_vv)
        return pow(kpar/kv, alpha_v)
    
    def _get_density_bias(self,z):
        """Linear density bias as a function of redshift, 
            values from DESI Collaboration et al., 2024b"""
        #alpha=1.25
        alpha = 2.9
        bias_zref = -0.1078
        zref = 2.33
        return bias_zref * ((1 + z)/(1 + zref))**alpha

    def _get_beta_rsd(self,z):
        """Linear RSD anisotropy parameter as a function of redshift,
        values from DESI Collaboration et al., 2024b"""
        alpha = 0.0
        beta_zref = 1.743
        zref = 2.33
        return beta_zref*((1 + z)/(1 + zref))**alpha

    def _small_scale_correction(self,k_hmpc,mu):
        """Analytic formula for small-scales correction to Lyman alpha P3D(z,k,mu) 
            from McDonald (2003). 
            Values computed at z=2.25, it would be great to have z-evolution.
            Values are cosmology dependent, but we ignore it here.
            Wavenumbers in h/Mpc. """
        # this will go in the exponent
        texp = (self._get_non_linear_corr(k_hmpc) 
                - self._get_pressure_corr(k_hmpc)
                - self._get_non_linear_velo(k_hmpc,mu))
        
        return np.exp(texp)

    def compute_bias(self,z,k_hmpc,mu,linear=True):
        """Analytic formula for scale-dependent bias of Lyman alpha P3D(z,k,mu) 
            from McDonald (2003), including Kaiser and small scale correction.  
            Basically, it retursn P_F(k,mu) / P_lin(k,mu)
            Values computed at z=2.25, it would be great to have z-evolution.
            Values are cosmology dependent, but we ignore it here.
            If linear=True, return only Kaiser.
            Wavenumbers in h/Mpc. """
        b_delta = self._get_density_bias(z)
        beta = self._get_beta_rsd(z)
        kaiser = (b_delta * (1+ beta * mu**2))**2
        if linear:
            return kaiser
        else:
            return kaiser * self._small_scale_correction(k_hmpc,mu)