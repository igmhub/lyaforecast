import numpy as np

#to-do, make this less hard-coded
class AnalyticBias:
    """Class to store analytic formulae for biases of Lya P3D, including non-linear corrections.
        These will later be handled by ForestFlow, currently parameter values are out-of-date."""
    OPTIONS = ['lya','qso','lbg']

    def __init__(self,cosmo):
        self._cosmo = cosmo
        self._growth_rate = self._cosmo.growth_rate
        self._zref = self._cosmo.z_ref

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
            values from DESI Collaboration et al., 2025"""
        if which=='lya':
            alpha = 2.9
            bias_zref = -0.1352
            zref = 2.33
        elif which=='qso':
            alpha = 1.44
            bias_zref = 3.54
            zref = 2.33
        elif which=='lbg':
            #From Vanina et al. 2024
            alpha = 1.44
            bias_zref = 3.48
            zref = 2.9
        else:
            raise ValueError(f'invalid biasing: {which}, select from: {self.OPTIONS}')
        
        return bias_zref * ((1 + z)/(1 + zref))**alpha

    def _get_beta_rsd(self,z,which):
        """Linear RSD anisotropy parameter as a function of redshift,
        values from DESI Collaboration et al., 2025"""
        if which=='lya':
            alpha = 0.0
            zref = 2.33
            beta_zref = 1.45
        elif which=='qso':
            alpha = 0.0
            zref = 2.33
            beta_zref = self._growth_rate/self._get_density_bias(zref,which)
        elif which=='lbg':
            alpha = 0.0
            zref = 2.7
            beta_zref = self._growth_rate/self._get_density_bias(zref,which)

            #to fix: growth-rate here is estimated at zref set by camb config, that won't be the same as both 2.33 and 2.7.

        else:
            raise ValueError(f'invalid biasing: {which}, select from: {self.OPTIONS}')
        
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
        
        tracers = which.split('_')

        if len(tracers) > 1:
            b = self._get_density_bias(z,tracers[0]) * self._get_density_bias(z,tracers[1])
            rsd = (1 + self._get_beta_rsd(z,tracers[0]) * mu**2) * (1 + self._get_beta_rsd(z,tracers[1]) * mu**2)
            kaiser = b * rsd    
        else:
            b = self._get_density_bias(z,which)**2
            rsd = (1 + self._get_beta_rsd(z,which) * mu**2)**2
            kaiser = b * rsd    
        if linear:
            #currently only option
            return kaiser
        else:
            return kaiser * self._small_scale_correction(k_hmpc,mu,which)
