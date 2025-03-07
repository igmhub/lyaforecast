import numpy as np

class AnalyticBias:
    """Class to store analytic formulae for biases of Lya P3D, including non-linear corrections.
        These will later be handled by ForestFlow, currently parameter values are out-of-date."""
    POWER_OPTIONS = ['lya','qso']

    def __init__(self,cosmo):
        self._cosmo = cosmo
        self._growth_rate = self._cosmo.growth_rate

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
    
    def _get_density_bias(self,z,which):
        """Linear density bias as a function of redshift, 
            values from DESI Collaboration et al., 2024b"""
        #alpha=1.25
        if which=='lya':
            alpha = 2.9
            bias_zref = -0.1078
            zref = 2.33
        elif which=='qso':
            alpha = 1.44
            bias_zref = 3.4
            zref = 2.33
        else:
            raise ValueError(f'invalid biasing: {which}, select from: {self.POWER_OPTIONS}')
        
        return bias_zref * ((1 + z)/(1 + zref))**alpha

    def _get_beta_rsd(self,z,which):
        """Linear RSD anisotropy parameter as a function of redshift,
        values from DESI Collaboration et al., 2024b"""
        if which=='lya':
            alpha = 0.0
            zref = 2.33
            beta_zref = 1.743
        elif which=='qso':
            alpha = 0.0
            zref = 2.33
            beta_zref = self._growth_rate/self._get_density_bias(zref,which)
        else:
            raise ValueError(f'invalid biasing: {which}, select from: {self.POWER_OPTIONS}')
        
        return beta_zref*((1 + z)/(1 + zref))**alpha

    def _small_scale_correction(self,k_hmpc,mu,which):
        """Analytic formula for small-scales correction to Lyman alpha P3D(z,k,mu) 
            from McDonald (2003). 
            Values computed at z=2.33, it would be great to have z-evolution.
            Values are cosmology dependent, but we ignore it here.
            Wavenumbers in h/Mpc. """
        if which=='lya':
            texp = (self._get_non_linear_corr(k_hmpc) 
                - self._get_pressure_corr(k_hmpc)
                - self._get_non_linear_velo(k_hmpc,mu))
            return np.exp(texp)
        else:
            return 1

    def compute_bias(self,z,k_hmpc,mu,linear=True,which='lya'):
        """Analytic formula for scale-dependent bias of Lyman alpha P3D(z,k,mu),
             including Kaiser and small scale correction.  
            Basically, it retursn P_F(k,mu) / P_lin(k,mu)
            Values computed at z=2.33, it would be great to have z-evolution.
            Values are cosmology dependent, but we ignore it here.
            If linear=True, return only Kaiser.
            Wavenumbers in h/Mpc. """
        if which=='lyaqso':
            b = self._get_density_bias(z,'lya') * self._get_density_bias(z,'qso')
            rsd = (1 + self._get_beta_rsd(z,'lya') * mu**2) * (1 + self._get_beta_rsd(z,'qso') * mu**2)
            kaiser = b * rsd    
        else:
            b = self._get_density_bias(z,which)
            rsd = self._get_beta_rsd(z,which)
            kaiser = (b * (1 + rsd * mu**2))**2        

        if linear:
            return kaiser
        else:
            return kaiser * self._small_scale_correction(k_hmpc,mu,which)
