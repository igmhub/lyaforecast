import numpy as np
from scipy.interpolate import interp1d


# to-do, make this less hard-coded
class AnalyticBias:
    """Analytic formulae for biases and non-linear corrections of Lya P3D.

    Parameters are out-of-date and will later be handled by ForestFlow.
    """

    OPTIONS = ['lya', 'qso', 'elgqso', 'lbg', 'lae']
    _zbin_index = None

    def __init__(self, cosmo):
        """
        Parameters
        ----------
        cosmo : CosmoCamb
            Cosmological model instance providing growth rate information.
        """
        self._cosmo = cosmo
        self._growth_rate_func = interp1d(self._cosmo.z_bins,self._cosmo.growth_rate_zbins,kind='linear', bounds_error=False, fill_value='extrapolate')
        self._zref = self._cosmo.z_ref
        self._density_bias_func = dict()

    def _get_non_linear_corr(self, k_hMpc):
        """Non-linear clustering correction to Lya P3D.

        Parameters
        ----------
        k_hMpc : float or array_like
            Wavenumber in h/Mpc.

        Returns
        -------
        float or ndarray
            Multiplicative non-linear correction exponent.
        """
        k_nl = 6.40
        alpha_nl = 0.569
        return pow(k_hMpc / k_nl, alpha_nl)

    def _get_pressure_corr(self, k_hMpc):
        """Jeans pressure correction to Lya P3D.

        Parameters
        ----------
        k_hMpc : float or array_like
            Wavenumber in h/Mpc.

        Returns
        -------
        float or ndarray
            Multiplicative pressure correction exponent.
        """
        k_p = 15.3
        alpha_p = 2.01
        return pow(k_hMpc / k_p, alpha_p)

    def _get_non_linear_velo(self, k_hMpc, mu):
        """Non-linear velocity (fingers-of-god) correction to Lya P3D.

        Parameters
        ----------
        k_hMpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float or array_like
            Cosine of the angle to the line of sight.

        Returns
        -------
        float or ndarray
            Multiplicative velocity correction exponent.
        """
        k_v0 = 1.220
        alpha_v = 1.50
        k_vv = 0.923
        alpha_vv = 0.451
        kpar = k_hMpc * mu
        kv = k_v0 * pow(1 + k_hMpc / k_vv, alpha_vv)
        return pow(kpar/kv, alpha_v)

    def _get_density_bias(self, z, which):
        """Linear density bias b(z) for a given tracer.

        Values from DESI Collaboration et al., 2025.

        Parameters
        ----------
        z : float
            Redshift.
        which : str
            Tracer name; must be one of ``OPTIONS``.

        Returns
        -------
        float
            Linear density bias at redshift z.
        """
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
        """RSD anisotropy parameter beta = f/b as a function of redshift.

        Values from DESI Collaboration et al., 2025.

        Parameters
        ----------
        z : float
            Redshift.
        which : str
            Tracer name; must be one of ``OPTIONS``.

        Returns
        -------
        float
            Beta parameter at redshift z.
        """
        if which == 'lya':
            alpha = 0.0
            zref = 2.33
            beta_zref = 1.45
            return beta_zref*((1 + z)/(1 + zref))**alpha
        else :
            return self._growth_rate_func(z)/self._get_density_bias(z, which)

    def _small_scale_correction(self, k_hmpc, mu, which):
        """Analytic small-scale correction to Lya P3D from McDonald (2003).

        Values computed at z=2.33; cosmology-dependent effects are ignored.

        Parameters
        ----------
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float or array_like
            Cosine of the angle to the line of sight.
        which : str
            Tracer name.

        Returns
        -------
        float or ndarray
            Multiplicative small-scale correction factor.
        """
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
        """Total scale-dependent bias P_F / P_lin, including Kaiser and small-scale terms.

        Parameters
        ----------
        z : float
            Redshift.
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float or array_like
            Cosine of the angle to the line of sight.
        corr : str
            Correlation name of the form 'tracer1_tracer2'.
        linear : bool, optional
            If True, return only the Kaiser term (no small-scale correction).

        Returns
        -------
        float or ndarray
            Total bias factor P_F / P_lin.
        """
        tracers = corr.split('_')
        assert len(tracers) == 2, 'corr must be of the form tracer1_tracer2'

        kaiser = 1
        for i, t in enumerate(tracers):
            if t.find("lya") == 0 :
                t="lya"
            kaiser *= self._get_density_bias(z, t) * (1 + self._get_beta_rsd(z, t) * mu**2)
        return kaiser

    def set_density_bias_func(self, tracer, bias_func):
        """Register an external bias interpolator for a given tracer.

        Parameters
        ----------
        tracer : str
            Tracer name; must be one of ``OPTIONS``.
        bias_func : callable
            Function b(z) returning the linear density bias at redshift z.
        """
        if not tracer in self.OPTIONS :
            raise ValueError(f"in set_density_bias_func, tracer name must be in {self.OPTIONS}")
        self._density_bias_func[tracer]=bias_func
