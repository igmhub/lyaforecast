"""Class to handle computation of weights for various tracers."""
import numpy as np


class Weights:
    """Compute optimal weights and noise integrals for Lya and discrete tracers.

    Implements the weighting scheme of McDonald & Eisenstein (2007) to estimate
    effective densities and noise powers as a function of magnitude.
    """

    def __init__(
        self, config, maglist, cosmo, powerspec, spectrograph,
        forest_length, pixel_length, resolution, lambda_mean, z_bin, z_qso, zmin, zmax,
        lya_tracer=None, discrete_tracer=None
    ):
        """
        Parameters
        ----------
        config : configparser.ConfigParser
            Parsed configuration object.
        maglist : array_like
            Grid of apparent magnitudes over which to integrate.
        cosmo : CosmoCamb
            Cosmological model instance.
        powerspec : PowerSpectrum
            Power spectrum model instance.
        spectrograph : Spectrograph
            Spectrograph noise model instance.
        forest_length : float
            Length of the Lya forest in km/s.
        pixel_length : float
            Pixel width in km/s.
        resolution : float
            Spectrograph resolution in km/s.
        lambda_mean : float
            Mean observed wavelength of the forest in Angstroms.
        z_bin : float
            Mean redshift of the current redshift bin.
        z_qso : float
            Central quasar redshift corresponding to the bin centre.
        zmin : float
            Minimum redshift of the bin.
        zmax : float
            Maximum redshift of the bin.
        lya_tracer : Tracer, optional
            Lya forest tracer (required for Lya integrals).
        discrete_tracer : Tracer, optional
            Discrete tracer (required for tracer density).
        """
        self.weights = None
        self._config = config
        self.maglist = maglist

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
        self._lya_tracer = lya_tracer
        self._discrete_tracer = discrete_tracer

        # optional behaviour (defaults reproduce the central-redshift treatment)
        control = config['control']
        # double integral over quasar redshift and forest pixels within the bin
        self._integrate_forest_pixels = control.getboolean(
            'integrate forest pixels', fallback=False)
        self._num_zq_points = control.getint('num zq integration points', fallback=128)
        self._num_lambda_points = control.getint('num forest wavelength points', fallback=64)
        # include aliasing (P1D x P_w) as signal when computing the weights
        # (McDonald & Eisenstein 2007); without it the updates tend to weights
        # that omit the P1D variance limiting sparse forests
        self._weights_include_aliasing = control.getboolean(
            'weights include aliasing', fallback=True)

        # grid and noise cube of the integrated mode, built on first use
        self._forest_grid_ready = False
        self._pixel_noise_var = None

        # get k values to evaluate p_w
        self._get_eval_mode()

    def _get_eval_mode(self):
        """Set the (kt, kp) evaluation point for signal/noise weighting.

        The mode is chosen based on the ``measurement type`` config key:
        'bao' uses large-scale modes (~0.035 h/Mpc); 'p1d' uses smaller scales (~0.1 h/Mpc).
        """
        ACCEPTED_OPTIONS = ['bao', 'p1d']
        measurement_type = self._config['control'].get('measurement type')
        if measurement_type == 'bao':
            self.kt_w_deg = 2.4  # ~ 0.035 h/Mpc
            self.kp_w_kms = 0.00035  # ~ 0.035 h/Mpc
        elif measurement_type == 'p1d':
            self.kt_w_deg = 7  # 0.1 h/Mpc
            self.kp_w_kms = 0.001  # ~ 0.1 h/Mpc
        else:
            raise ValueError('measurement type must be chosen from:', ACCEPTED_OPTIONS)

    def compute_weights(self, corr):
        """Compute optimal Lya weights as a function of magnitude via iteration.

        Iterates because weights depend on I1 and I1 depends on the weights.

        Parameters
        ----------
        corr : str
            Correlation name used to evaluate the 3D and 1D power spectra.

        Returns
        -------
        ndarray
            Optimal weights w(m) of shape (n_mag,).
        """
        # pre-compute power spectra for weighting
        self._p3d_w = self._powerspec.compute_p3d_kms_smooth(
            self._z_bin, self.kt_w_deg, self.kp_w_kms, self._res_kms, self._pix_kms, corr)

        self._p1d_w = self._powerspec.compute_p1d_kms(
            self._z_bin, self.kp_w_kms, self._res_kms, self._pix_kms, corr)

        # compute first weights using only 1D and noise variance
        weights = self._initialise_weights_lya()

        num_iter = 3
        for i in range(num_iter):
            weights = self._compute_weights_lya(weights)

        return weights

    def _initialise_weights_lya(self):
        """Compute initial Lya weights using only P1D and pixel noise variance.

        Returns
        -------
        ndarray
            Initial weights w(m) of shape (n_mag,).
        """
        # noise pixel variance as a function of magnitude (dimensionless)
        noise_var = self._get_pix_var_m()
        # pixel variance from P1D (dimensionless)
        pix_var_1d = self._p1d_w / self._pix_kms
        weights = pix_var_1d / (pix_var_1d + noise_var)

        return weights

    def _compute_weights_lya(self, weights):
        """Update Lya weights using P3D as the signal term (McDonald & Eisenstein 2007).

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Updated weights w(m) of shape (n_mag,).
        """
        # 3D noise power as a function of magnitude
        noise_power = self._compute_noise_power_m(weights)
        # effective 3D density of quasars
        int_1 = self.compute_int_1(weights)
        int_2 = self.compute_int_2(weights)
        # 2D density of lines of sight (units of 1/deg^2)
        aliasing = int_2 / (int_1**2 * self._forest_length)
        # weights include aliasing as signal
        if self._weights_include_aliasing:
            # p1d_w [km/s] x aliasing [deg^2] has the units of p3d_w [deg^2 km/s]
            signal_power = self._p3d_w + self._p1d_w * aliasing
        else:
            signal_power = self._p3d_w

        weights = signal_power / (signal_power + noise_power)

        return weights

    def _initialise_weights_qso(self):
        """Compute initial quasar (FKP) weights: uniform over magnitudes.

        Returns
        -------
        ndarray
            Initial weights of shape (n_mag,), all ones.
        """
        return np.ones_like(self.maglist)

    def _compute_weights_qso(self, weights):
        """FKP weighting of discrete tracers (quasars).

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Updated FKP weights of shape (n_mag,).
        """
        np_eff = self.compute_int_1(weights)
        weights = self._p3d_w / (self._p3d_w + 1 / np_eff)

        return weights

    def _get_dn_dkmsdm(self, z, m, tracer):
        """Convert tracer dn/dzdm (deg^{-2}) to dn/d(km/s)dm.

        Parameters
        ----------
        z : float
            Redshift.
        m : array_like
            Apparent magnitudes.
        tracer : Tracer
            Tracer instance providing the dn/dzdm interpolator.

        Returns
        -------
        ndarray
            dn/d(km/s)/dm in (km/s)^{-1} deg^{-2} per unit magnitude.
        """
        dkms_dz = self._cosmo.SPEED_LIGHT / (1 + z)

        # number density
        dndm_degdzdm = tracer.get_dn_dzdm(z, m)

        dn_degkmsdm = dndm_degdzdm / dkms_dz

        return dn_degkmsdm

    def compute_int_1(self, weights):
        """Compute integral I1 from McDonald & Eisenstein (2007): effective pixel density.

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Cumulative I1(m) of shape (n_mag,) in deg^{-2} (km/s)^{-1}.
        """
        assert self._lya_tracer is not None, "Lya tracer must be set to compute I1"
        # quasar number density
        dn_dmdegkms = self._get_dn_dkmsdm(self._zq, self.maglist, self._lya_tracer)
        dm = self.maglist[1] - self.maglist[0]
        integrand = dn_dmdegkms * weights

        # move to using cumsum so we can plot as a function of magnitude
        int_1 = np.sum(integrand) * dm

        return int_1

    def compute_int_2(self, weights):
        """Compute integral I2 from McDonald & Eisenstein (2007): aliasing normalisation.

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Cumulative I2(m) of shape (n_mag,) in deg^{-2} (km/s)^{-1}.
        """
        assert self._lya_tracer is not None, "Lya tracer must be set to compute I2"
        # quasar number density
        dn_dmdegkms = self._get_dn_dkmsdm(self._zq, self.maglist, self._lya_tracer)
        dm = self.maglist[1] - self.maglist[0]
        integrand = dn_dmdegkms * weights**2
        int_2 = np.sum(integrand) * dm

        return int_2

    def compute_int_3(self, weights):
        """Compute integral I3 from McDonald & Eisenstein (2007): effective noise power.

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Cumulative I3(m) of shape (n_mag,) in noise variance units.
        """
        assert self._lya_tracer is not None, "Lya tracer must be set to compute I3"
        # pixel noise variance (dimensionless)
        pixel_var = self._get_pix_var_m()
        # quasar number density
        dn_dmdegkms = self._get_dn_dkmsdm(self._zq, self.maglist, self._lya_tracer)
        dm = self.maglist[1] - self.maglist[0]

        integrand = dn_dmdegkms * weights**2 * pixel_var
        int_3 = np.sum(integrand) * dm

        return int_3

    def get_np_eff_lya(self, weights):
        """Compute effective 1D pixel density n_p^eff (McDonald & Eisenstein 2007).

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Effective pixel density of shape (n_mag,) in deg^{-2} (km/s)^{-1}.
        """
        int_1 = self.compute_int_1(weights)
        # number of pixels in a forest
        num_pix = self._forest_length / self._pix_kms
        np_eff = int_1 * num_pix

        return np_eff

    def get_n_tracer(self, tracer):
        """Compute the cumulative tracer number density as a function of magnitude limit.

        Parameters
        ----------
        tracer : Tracer
            Discrete tracer whose dn/dzdm is integrated.

        Returns
        -------
        ndarray
            Cumulative n(< m) of shape (n_mag,) in deg^{-2} (km/s)^{-1}.
        """
        mags = self.maglist
        dn_dkmsdm = self._get_dn_dkmsdm(self._z_bin, self.maglist, tracer)
        dm = mags[1] - mags[0]
        dn_dkms = np.cumsum(dn_dkmsdm * dm)

        return dn_dkms

    def compute_tracer_weights(self, corr, tracer):
        """Compute FKP-like weights for a discrete tracer.

        Parameters
        ----------
        corr : str
            Correlation name used to evaluate P3D.
        tracer : Tracer
            Discrete tracer instance.

        Returns
        -------
        ndarray
            Tracer weights w(m) of shape (n_mag,).
        """
        p3d = self._powerspec.compute_p3d_kms_smooth(
            self._z_bin, self.kt_w_deg, self.kp_w_kms, self._res_kms, self._pix_kms, corr)

        p_n = 1 / self.get_n_tracer(tracer)
        weights = p3d / (p3d + p_n)

        return weights

    def _get_pix_var_m(self):
        """Return the pixel noise variance (dimensionless) as a function of magnitude.

        Returns
        -------
        ndarray
            Noise variance sigma_N^2(m) of shape (n_mag,).
        """
        assert self._lya_tracer is not None, "Lya tracer must be set to compute pixel variance"
        # pixel width in angstroms
        pix_ang = self._pix_kms / self._cosmo.velocity_from_wavelength(self._z_bin)

        # noise rms per pixel
        noise_rms = np.zeros_like(self.maglist)
        for i, m in enumerate(self.maglist):
            noise_rms[i] = self._spectrograph.get_pixel_rms_noise(
                m, self._zq, self._lambda_mean, pix_ang, self._lya_tracer.num_exp)
        noise_var = noise_rms**2

        return noise_var

    def _compute_noise_power_m(self, weights):
        """Compute the 3D effective noise power P_N(m) (McDonald & Eisenstein 2007).

        Parameters
        ----------
        weights : ndarray
            Current weights w(m) of shape (n_mag,).

        Returns
        -------
        ndarray
            Effective 3D noise power of shape (n_mag,).
        """
        # pixel noise variance (dimensionless)
        pixel_var = self._get_pix_var_m()
        # 3D effective density of pixels
        neff = self.get_np_eff_lya(weights)
        noise_power = pixel_var / neff

        return noise_power

    # ------------------------------------------------------------------
    # Integrated forest-pixel mode: double integral over the quasar
    # redshift z_q and over the forest pixels lying inside the bin.
    # ------------------------------------------------------------------

    def _setup_forest_pixel_grid(self):
        """Build the (z_q, forest pixel, magnitude) integration grid for the bin.

        Works in u = ln(lambda_obs). The bin is [u1, u2] = ln[lambda_min, lambda_max]
        and the forest is [a, b] = ln[lrmin, lrmax] in the rest frame. A quasar at
        y = ln(1 + z_q) has its forest at u in [y + a, y + b], so it overlaps the bin
        for y in [u1 - b, u2 - a], further limited to the quasar redshift range
        [zq_min_forest, zq_max_forest] of the tracer.

        The z_q integral is a midpoint rule with ``num zq integration points`` points
        uniform in y = ln(1 + z_q), with dz_q = (1 + z_q) dy. For each z_q, the
        pixels are ``num forest wavelength points`` midpoints uniform in u over the
        exact overlap of the forest with the bin, so each pixel carries
        dv = c * overlap / n_lambda (km/s) and pixels outside the bin are never used.

        If the redshift ranges do not overlap (y_high <= y_low) there is no forest
        coverage: ``_has_forest_coverage`` is False, all quadrature weights vanish,
        and the integrals N1, N2, N3 are zero.

        Side effects: sets ``_zq_grid`` (n_zq,), ``_delta_zq`` (n_zq,),
        ``_log_lambda_pix``, ``_lambda_pix`` and ``_z_pix`` (n_zq, n_lambda),
        ``_overlap_length_kms`` and ``_delta_v_kms`` (n_zq,), ``_dn_dzq_dm``
        (n_zq, n_mag) in deg^-2 per unit z_q per magnitude, ``_bin_length_kms``,
        and ``_integration_measure`` (n_zq, 1, n_mag) in deg^-2 (the quadrature weight
        dn/dz_q dm * dz_q * dm * dv / L_bin).
        """
        assert self._lya_tracer is not None, "Lya tracer must be set for forest integration"
        assert self._num_zq_points >= 1 and self._num_lambda_points >= 1, \
            "Number of integration points must be positive"

        speed_light = self._cosmo.SPEED_LIGHT
        num_zq = self._num_zq_points
        num_lambda = self._num_lambda_points

        # bin edges and rest-frame forest edges in u = ln(lambda)
        log_lambda_min = np.log(self._cosmo.LYA_REST * (1 + self._zmin))
        log_lambda_max = np.log(self._cosmo.LYA_REST * (1 + self._zmax))
        log_rest_min = np.log(self._lya_tracer.lrmin)
        log_rest_max = np.log(self._lya_tracer.lrmax)
        self._bin_length_kms = speed_light * (log_lambda_max - log_lambda_min)

        # range of y = ln(1+z_q) for which the forest overlaps the bin
        log_1pzq_low = max(
            log_lambda_min - log_rest_max, np.log(1 + self._lya_tracer.zq_min_forest))
        log_1pzq_high = min(
            log_lambda_max - log_rest_min, np.log(1 + self._lya_tracer.zq_max_forest))
        self._has_forest_coverage = bool(log_1pzq_high > log_1pzq_low)

        # midpoint rule in y = ln(1+z_q); dz_q = (1+z_q) dy
        if self._has_forest_coverage:
            delta_log_1pzq = (log_1pzq_high - log_1pzq_low) / num_zq
        else:
            delta_log_1pzq = 0.0
        log_1pzq_grid = log_1pzq_low + (np.arange(num_zq) + 0.5) * delta_log_1pzq
        self._zq_grid = np.exp(log_1pzq_grid) - 1
        self._delta_zq = (1 + self._zq_grid) * delta_log_1pzq

        # part of each forest inside the bin, in u = ln(lambda); the forest of
        # a quasar at y spans [y + a, y + b]
        overlap_low = np.maximum(log_lambda_min, log_1pzq_grid + log_rest_min)
        overlap_high = np.minimum(log_lambda_max, log_1pzq_grid + log_rest_max)
        overlap_length = np.maximum(overlap_high - overlap_low, 0.0)
        self._overlap_length_kms = speed_light * overlap_length
        self._delta_v_kms = self._overlap_length_kms / num_lambda

        # pixel centres: midpoints uniform in u within the overlap, (n_zq, n_lambda)
        pixel_fraction = (np.arange(num_lambda) + 0.5) / num_lambda
        self._log_lambda_pix = (
            overlap_low[:, np.newaxis] + overlap_length[:, np.newaxis] * pixel_fraction
        )
        self._lambda_pix = np.exp(self._log_lambda_pix)
        self._z_pix = self._lambda_pix / self._cosmo.LYA_REST - 1

        # quasar density per unit z_q and magnitude on the (n_zq, n_mag) grid;
        # get_dn_dzdm masks with boolean arrays that need identical shapes
        zq_2d, mag_2d = np.meshgrid(self._zq_grid, self.maglist, indexing='ij')
        self._dn_dzq_dm = self._lya_tracer.get_dn_dzdm(zq_2d, mag_2d)

        # quadrature weight of each (z_q, m) pair and pixel, normalised by the bin
        # length so that the integrals N_n play the role of I_n * L_q (deg^-2)
        delta_mag = self.maglist[1] - self.maglist[0]
        self._integration_measure = (
            self._dn_dzq_dm
            * self._delta_zq[:, np.newaxis] * delta_mag
            * self._delta_v_kms[:, np.newaxis]
            / self._bin_length_kms
        )[:, np.newaxis, :]

        self._forest_grid_ready = True

    @property
    def has_forest_coverage(self):
        """bool: whether some allowed z_q has forest pixels inside the bin.

        False if the quasar redshift range of the sample does not overlap the bin, in
        which case the quadrature measure of the integrated mode vanishes identically.
        The integration grid is built on first access.
        """
        if not self._forest_grid_ready:
            self._setup_forest_pixel_grid()

        return self._has_forest_coverage

    def _get_pixel_noise_var_grid(self):
        """Return the pixel noise variance on the (z_q, pixel, magnitude) grid.

        The pixel width in Angstroms follows the pixel wavelength,
        pix_ang(lambda) = pix_kms * lambda / c, so that every pixel has the same
        velocity width. The result is computed once and cached.

        Returns
        -------
        ndarray
            Dimensionless noise variance sigma_N^2(m, z_q, lambda) of shape
            (n_zq, n_lambda, n_mag).
        """
        if self._pixel_noise_var is None:
            if not self._forest_grid_ready:
                self._setup_forest_pixel_grid()

            # pixel width in Angstroms for each pixel, (n_zq, n_lambda, 1)
            pix_ang = self._pix_kms * self._lambda_pix / self._cosmo.SPEED_LIGHT

            noise_rms = self._spectrograph.get_pixel_rms_noise_array(
                self.maglist[np.newaxis, np.newaxis, :],
                self._zq_grid[:, np.newaxis, np.newaxis],
                self._lambda_pix[:, :, np.newaxis],
                pix_ang[:, :, np.newaxis],
                self._lya_tracer.num_exp
            )
            self._pixel_noise_var = noise_rms**2

        return self._pixel_noise_var

    def compute_weights_integrated(self, corr):
        """Compute optimal Lya weights w(z_q, lambda, m) for the integrated forest mode.

        Same iteration as :meth:`compute_weights`, but the weights depend on the
        quasar redshift, the pixel wavelength and the magnitude, so the pixel noise
        variance uses the S/N at (m, z_q, lambda_pix). The initial weights use only
        P1D and the pixel variance; three iterations of w = P_S / (P_S + P_N) follow.

        Parameters
        ----------
        corr : str
            Correlation name used to evaluate the 3D and 1D power spectra.

        Returns
        -------
        ndarray
            Weights of shape (n_zq, n_lambda, n_mag); all zero if the bin has no
            forest coverage.
        """
        if not self._forest_grid_ready:
            self._setup_forest_pixel_grid()

        # pre-compute power spectra for weighting, as in the central mode
        self._p3d_w = self._powerspec.compute_p3d_kms_smooth(
            self._z_bin, self.kt_w_deg, self.kp_w_kms, self._res_kms, self._pix_kms, corr)

        self._p1d_w = self._powerspec.compute_p1d_kms(
            self._z_bin, self.kp_w_kms, self._res_kms, self._pix_kms, corr)

        pixel_noise_var = self._get_pixel_noise_var_grid()

        if not self._has_forest_coverage:
            return np.zeros_like(pixel_noise_var)

        # initial weights from P1D and the pixel noise variance only
        pix_var_1d = self._p1d_w / self._pix_kms
        weights = pix_var_1d / (pix_var_1d + pixel_noise_var)

        num_iter = 3
        for i in range(num_iter):
            weights = self._compute_weights_lya_integrated(weights)

        return weights

    def _compute_weights_lya_integrated(self, weights):
        """Update the integrated-mode weights with P3D (plus optional aliasing) as signal.

        The noise power of a pixel is P_N = sigma_N^2 * pix_kms / N1. If
        ``weights include aliasing`` is set, the signal is
        P_S = P3D_w + P1D_w * N2 / N1^2 (McDonald & Eisenstein 2007, Eq. 19),
        with N1 and N2 from the current weights.

        Parameters
        ----------
        weights : ndarray
            Current weights of shape (n_zq, n_lambda, n_mag).

        Returns
        -------
        ndarray
            Updated weights of shape (n_zq, n_lambda, n_mag).
        """
        int_1, int_2, _, _ = self.compute_forest_integrals_integrated(weights)

        # noise power per pixel, (n_zq, n_lambda, n_mag), deg^2 km/s
        noise_power = self._get_pixel_noise_var_grid() * self._pix_kms / int_1

        if self._weights_include_aliasing:
            aliasing = int_2 / int_1**2
            signal_power = self._p3d_w + self._p1d_w * aliasing
        else:
            signal_power = self._p3d_w

        return signal_power / (signal_power + noise_power)

    def compute_forest_integrals_integrated(self, weights):
        """Compute the integrals N1, N2, N3 and the effective redshift in integrated mode.

        N_n = (1 / L_bin) sum_{z_q, lambda, m} dn/dz_q dm dz_q dm dv w^n
        (times sigma_N^2 for n = 3), in deg^-2 (n = 1, 2) and deg^-2 times noise
        variance (n = 3), with L_bin = c ln(lambda_max / lambda_min). They play the
        role of I_n * L_q of the central mode, so that the aliasing weight is
        N2 / N1^2 and the effective noise power is N3 * pix_kms / N1^2. The
        effective redshift is the N1-integrand-weighted mean of
        z_pix = lambda_pix / lambda_Lya - 1.

        Parameters
        ----------
        weights : ndarray
            Weights of shape (n_zq, n_lambda, n_mag).

        Returns
        -------
        int_1 : float
            N1 in deg^-2.
        int_2 : float
            N2 in deg^-2.
        int_3 : float
            N3 in deg^-2 (noise variance units).
        z_eff : float
            Effective redshift of the bin; NaN if N1 = 0 (no forest coverage).
        """
        if not self._forest_grid_ready:
            self._setup_forest_pixel_grid()

        pixel_noise_var = self._get_pixel_noise_var_grid()

        integrand_1 = self._integration_measure * weights
        integrand_2 = integrand_1 * weights
        integrand_3 = integrand_2 * pixel_noise_var

        int_1 = np.sum(integrand_1)
        int_2 = np.sum(integrand_2)
        int_3 = np.sum(integrand_3)

        # effective redshift of the pixels contributing to N1
        if int_1 > 0:
            z_eff = np.sum(self._z_pix[:, :, np.newaxis] * integrand_1) / int_1
        else:
            z_eff = np.nan

        return int_1, int_2, int_3, z_eff
