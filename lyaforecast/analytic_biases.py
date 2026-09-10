from scipy.interpolate import interp1d


# to-do, make this less hard-coded
class AnalyticBias:
    """Density biases and Kaiser redshift-space distortions for tracer P3D."""

    OPTIONS = ['lya', 'qso', 'elgqso', 'lbg', 'lae']

    def __init__(self, cosmo):
        """
        Parameters
        ----------
        cosmo : CosmoCamb
            Cosmological model instance providing growth rate information.
        """
        self._cosmo = cosmo
        self._growth_rate_func = interp1d(
            self._cosmo.z_bins, self._cosmo.growth_rate_zbins,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        self._density_bias_func = dict()

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

    def compute_bias(self, z, k_hmpc, mu, corr, linear=True):
        """Total Kaiser bias P_F / P_lin for a tracer pair.

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
            Retained for configuration compatibility; the model uses Kaiser terms only.

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
        if tracer not in self.OPTIONS :
            raise ValueError(f"in set_density_bias_func, tracer name must be in {self.OPTIONS}")
        self._density_bias_func[tracer]=bias_func
