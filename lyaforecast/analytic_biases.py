import numpy as np
from scipy.interpolate import interp1d


# to-do, make this less hard-coded
class AnalyticBias:
    """Class to store analytic formulae for biases of Lya P3D, including non-linear corrections.
        These will later be handled by ForestFlow, currently parameter values are out-of-date."""
    OPTIONS = ['lya', 'qso', 'elgqso', 'lbg', 'lae']
    # _tracer_bias = None  # replace by functions, called in _get_density_bias
    _zbin_index = None

    def __init__(self, cosmo):
        self._cosmo = cosmo
        self._growth_rate_func = interp1d(self._cosmo.z_bins,self._cosmo.growth_rate_zbins,kind='linear', bounds_error=False, fill_value='extrapolate')
        self._zref = self._cosmo.z_ref
        self._density_bias_func = dict()

    def _get_non_linear_corr(self, k_hMpc):
        """Non-linear correction"""
        k_nl = 6.40
        alpha_nl = 0.569
        return pow(k_hMpc / k_nl, alpha_nl)

    def _get_pressure_corr(self, k_hMpc):
        """Pressure correction"""
        k_p = 15.3
        alpha_p = 2.01
        return pow(k_hMpc / k_p, alpha_p)

    def _get_non_linear_velo(self, k_hMpc, mu):
        """Non-linear velocities"""
        k_v0 = 1.220
        alpha_v = 1.50
        k_vv = 0.923
        alpha_vv = 0.451
        kpar = k_hMpc * mu
        kv = k_v0 * pow(1 + k_hMpc / k_vv, alpha_vv)
        return pow(kpar/kv, alpha_v)

    def _get_density_bias(self, z, which):
        """Linear density bias as a function of redshift,
            values from DESI Collaboration et al., 2025"""

        if which in self._density_bias_func.keys() :
            # use externally provided bias function
            return self._density_bias_func[which](z)
        elif which == 'lya' :
            alpha = 2.9
            bias_zref = -0.1352
            zref = 2.33
        elif which == 'qso' or which == 'elgqso':
            alpha = 1.44
            bias_zref = 3.54
            zref = 2.33
        elif which == 'lbg':
            # From Vanina et al. 2024
            alpha = 1.44
            bias_zref = 3.48
            zref = 2.9
        elif which == 'lae':
            # From Vanina et al. 2024
            alpha = 1.44
            bias_zref = 2.2
            zref = 2.9
        else:
            raise ValueError(f'invalid biasing: {which}, select from: {self.OPTIONS}')

        return bias_zref * ((1 + z)/(1 + zref))**alpha

    def _get_beta_rsd(self, z, which):
        """Linear RSD anisotropy parameter as a function of redshift,
        values from DESI Collaboration et al., 2025"""
        if which == 'lya':
            alpha = 0.0
            zref = 2.33
            beta_zref = 1.45
            return beta_zref*((1 + z)/(1 + zref))**alpha
        else :
            return self._growth_rate_func(z)/self._get_density_bias(z, which)


    def _small_scale_correction(self, k_hmpc, mu, which):
        """Analytic formula for small-scales correction to Lyman alpha P3D(z,k,mu)
            from McDonald (2003).
            Values computed at z=2.33, it would be great to have z-evolution.
            Values are cosmology dependent, but we ignore it here.
            Wavenumbers in h/Mpc. """
        if which == 'lya':
            texp = (
                self._get_non_linear_corr(k_hmpc)
                - self._get_pressure_corr(k_hmpc)
                - self._get_non_linear_velo(k_hmpc, mu)
            )
            return np.exp(texp)
        else:
            return 1

    def compute_bias(self, z, k_hmpc, mu, corr, linear=True):
        """Analytic formula for scale-dependent bias of Lyman alpha P3D(z,k,mu),
             including Kaiser and small scale correction.
            Basically, it retursn P_F(k,mu) / P_lin(k,mu)
            Values computed at z=2.33, it would be great to have z-evolution.
            Values are cosmology dependent, but we ignore it here.
            If linear=True, return only Kaiser.
            Wavenumbers in h/Mpc. """
        tracers = corr.split('_')
        assert len(tracers) == 2, 'corr must be of the form tracer1_tracer2'

        kaiser = 1
        for i, t in enumerate(tracers):
            if t.find("lya") == 0 :
                t="lya"
            kaiser *= self._get_density_bias(z, t) * (1 + self._get_beta_rsd(z, t) * mu**2)
        return kaiser

    def set_density_bias_func(self,tracer,bias_func) :
        if not tracer in self.OPTIONS :
            raise ValueError(f"in set_density_bias_func, tracer name must be in {self.OPTIONS}")
        self._density_bias_func[tracer]=bias_func
