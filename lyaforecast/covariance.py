import numpy as np
from lyaforecast.weights import Weights


class Covariance:
    """Compute covariance matrix components for given tracers in a single redshift bin.

    Implements the McDonald & Eisenstein (2007) formalism. Different redshift bins
    are treated as independent; this object handles one bin at a time, updated
    by calling the instance.
    """

    forest_length = None
    _lc = None  # mean wavelength of forest
    _zq = None  # quasar redshift
    _pix_kms = None  # pixel width in km/s
    _res_kms = None  # spectrograph resolution in km/s

    TRACER_OPTIONS = ['elgqso', 'qso', 'lbg', 'lae']
    LYA_TRACER_OPTIONS = ['elgqso', 'qso', 'lbg']

    def __init__(
        self, config, cosmo, survey, spectrograph, power_spectrum, corr, tracer1, tracer2=None
    ):
        """
        Parameters
        ----------
        config : configparser.ConfigParser
            Parsed configuration object.
        cosmo : CosmoCamb
            Cosmological model instance.
        survey : Survey
            Survey instance.
        spectrograph : Spectrograph or None
            Spectrograph noise model; None for discrete-only correlations.
        power_spectrum : PowerSpectrum
            Power spectrum model instance.
        corr : str
            Correlation name of the form 'tracer1_tracer2'.
        tracer1 : Tracer
            First tracer instance.
        tracer2 : Tracer, optional
            Second tracer; defaults to tracer1 if not provided.
        """
        # Config
        self._config = config
        # Cosmological model
        self._cosmo = cosmo
        # survey instance
        self._survey = survey
        # spectrograph instance
        self._spectrograph = spectrograph
        # power spectrum instance
        self._power_spec = power_spectrum
        # whether to iterate over magnitude instead of redshift
        self.per_mag = self._config['control'].getboolean('per mag')

        # tracer instances
        self._tracer1 = tracer1
        self._tracer2 = tracer2 if tracer2 is not None else tracer1
        # tracer name
        self._tracer1_name = (
            tracer1.name if tracer1.type == 'discrete' else tracer1.background_tracer
        )
        if self._tracer1_name not in self.TRACER_OPTIONS:
            raise ValueError(f'Please choose from accepted tracers: {self.TRACER_OPTIONS}')

        self._tracer2_name = (
            tracer2.name if tracer2 and tracer2.type == 'discrete' else tracer2.background_tracer
        )
        if self._tracer2_name not in self.TRACER_OPTIONS:
            raise ValueError(f'Please choose from accepted tracers: {self.TRACER_OPTIONS}')

        if tracer1.type == 'continuous' and tracer2.type == 'continuous':
            self._corr_type = 'lya_auto'
        elif tracer1.type == 'discrete' and tracer2.type == 'discrete':
            self._corr_type = 'tracer_auto'
        else:
            self._corr_type = 'cross'

        self.tracer_corr = f'{self._tracer1_name}_{self._tracer2_name}'
        self.corr = corr

        # These will be dependent on redshift bins,
        # which are passed during forecast run.
        self.lmin = None
        self.lmax = None
        self._aliasing_weights = None
        self._effective_noise_power = None
        self._num_modes = None

        self.verbose = 1

    @property
    def pix_width_kms(self):
        """Pixel width in km/s.

        Returns
        -------
        float
        """
        return self._pix_kms

    @property
    def pix_res_kms(self):
        """Spectrograph resolution in km/s.

        Returns
        -------
        float
        """
        return self._res_kms

    @property
    def num_modes(self):
        """Number of Fourier modes as a function of k for the current redshift bin.

        Returns
        -------
        ndarray
        """
        return self._num_modes

    def __call__(self, zmin, zmax):
        """Configure the covariance for a new redshift bin defined by [zmin, zmax].

        Parameters
        ----------
        zmin : float
            Minimum redshift of the bin.
        zmax : float
            Maximum redshift of the bin.
        """
        self.lmin = self._cosmo.LYA_REST * (1 + zmin)
        self.lmax = self._cosmo.LYA_REST * (1 + zmax)
        self._z_mean = np.sqrt(self.lmin * self.lmax) / self._cosmo.LYA_REST - 1.0
        self._zmin = self.lmin / self._cosmo.LYA_REST - 1
        self._zmax = self.lmax / self._cosmo.LYA_REST - 1

        # conversions
        self._wavelength_to_velocity = self._cosmo.velocity_from_wavelength(self._z_mean)
        self._distance_to_velocity = self._cosmo.velocity_from_distance(self._z_mean)
        self._angle_to_distance = self._cosmo.distance_from_degrees(self._z_mean)

        if self._corr_type != 'tracer_auto':
            lya_tracer = self._tracer1 if self._tracer1.type == 'continuous' else self._tracer2

            # load central redshift/wavelength of bin
            self._get_zq_bin(lya_tracer)
            # get pix width in kms
            self._get_pix_kms(lya_tracer)
            # get resolution in kms (for mean wavelength)
            self._get_res_kms()
            # get observed wavelength range of forest
            self.forest_length = self._get_forest_length(lya_tracer)

        # load survey volume
        self._get_survey_volume()
        # load number of modes
        self._get_num_modes()

    def _get_pix_kms(self, lya_tracer):
        """Set the pixel width in km/s from the tracer configuration.

        Parameters
        ----------
        lya_tracer : Tracer
            Lya forest tracer providing pix_ang or pix_kms.
        """
        pix_ang = lya_tracer.pix_ang
        if pix_ang is not None:
            self._pix_kms = pix_ang * self._wavelength_to_velocity
        else:
            self._pix_kms = lya_tracer.pix_kms
            assert self._pix_kms is not None, 'Must provide pix width in kms or ang'

    def _get_res_kms(self):
        """Set the spectrograph resolution in km/s from survey config or wavelength."""
        if self._survey.res_kms is not None:
            self._res_kms = self._survey.res_kms
        else:
            res_ang = self._lc / self._survey.resolution
            self._res_kms = res_ang * self._wavelength_to_velocity

    def _get_zq_bin(self, lya_tracer):
        """Compute the mean quasar redshift and forest wavelength for the bin.

        Parameters
        ----------
        lya_tracer : Tracer
            Lya forest (continuous) tracer instance.
        """
        assert lya_tracer.type == 'continuous', 'Can only compute zq for continuous tracers'

        if not (self.lmin is None or self.lmax is None):
            # mean wavelength of bin
            self._lc = np.sqrt(self.lmin * self.lmax)
            # redshift of quasar for which forest is centered in z
            lrc = np.sqrt(lya_tracer.lrmin * lya_tracer.lrmax)
            self._zq = (self._lc / lrc) - 1.0

    def _get_redshift_depth(self):
        """Return the line-of-sight depth of the current redshift bin in km/s.

        Returns
        -------
        float
            Depth L in km/s.
        """
        c_kms = self._cosmo.SPEED_LIGHT
        L_kms = c_kms*np.log(self.lmax/self.lmin)

        return L_kms

    def _get_forest_length(self, lya_tracer):
        """Return the Lya forest length for a quasar at z_qso in km/s.

        Parameters
        ----------
        lya_tracer : Tracer
            Lya forest tracer defining rest-frame wavelength limits.

        Returns
        -------
        float
            Forest length in km/s.
        """
        assert lya_tracer.type == 'continuous', 'Can only compute zq for continuous tracers'

        lmax_forest = lya_tracer.lrmax * (1 + self._zq)
        lmin_forest = lya_tracer.lrmin * (1 + self._zq)

        Lq_kms = self._cosmo.SPEED_LIGHT * np.log(lmax_forest/lmin_forest)

        return Lq_kms

    def _get_forest_wave(self, lya_tracer):
        """Compute the observed wavelength grid spanning the Lya forest.

        Parameters
        ----------
        lya_tracer : Tracer
            Lya forest tracer defining rest-frame wavelength limits.
        """
        assert lya_tracer.type == 'continuous', 'Can only compute zq for continuous tracers'

        lmax_forest = lya_tracer.lrmax * (1 + self._zq)
        lmin_forest = lya_tracer.lrmin * (1 + self._zq)
        nbins = int((lmax_forest - lmin_forest) / self._survey.pix_ang)

        self._forest_wave = np.linspace(lmin_forest, lmax_forest, nbins)

    def _get_survey_volume(self):
        """Compute the survey volume in (Mpc/h)^3 for the current redshift bin."""
        volume_degkms = self._survey.area_deg2 * self._get_redshift_depth()
        volume_mpch = volume_degkms * self._angle_to_distance**2 / self._distance_to_velocity

        self._survey_volume_mpc = volume_mpch

    def _get_num_modes(self):
        """Compute the number of Fourier modes as a function of k for the current bin."""
        self._num_modes = (
            self._survey_volume_mpc * self._power_spec.k**2
            * self._power_spec.dk * self._power_spec.dmu / (2 * np.pi**2)
        )

    def compute_eff_density_and_noise(self, corr_name):
        """Compute the effective density of lines of sight and noise power for the current bin.

        Evaluates the aliasing weights (P_w) and effective noise power (P_N^eff) from
        McDonald & Eisenstein (2007).

        Parameters
        ----------
        corr_name : str
            Correlation name passed to the Weights instance for P3D/P1D evaluation.
        """
        lya_tracer = None
        discrete_tracer = None

        if self._tracer1.type == 'discrete' :
            discrete_tracer = self._tracer1
        else :
            lya_tracer = self._tracer1

        if self._tracer2.type == 'discrete' :
            discrete_tracer = self._tracer2
        else  :
            lya_tracer = self._tracer2

        self._weights = Weights(
            self._config, self._survey.maglist, self._cosmo, self._power_spec, self._spectrograph,
            self.forest_length, self._pix_kms, self._res_kms, self._lc,
            self._z_mean, self._zq, self._zmin, self._zmax, lya_tracer, discrete_tracer
        )

        if lya_tracer is not None :
            # These only matter for lya
            w_lya = self._weights.compute_weights(corr_name)
            self._w_lya = w_lya

            # given weights, compute integrals in McDonald & Eisenstein (2007)
            int_1 = self._weights.compute_int_1(w_lya)
            int_2 = self._weights.compute_int_2(w_lya)
            int_3 = self._weights.compute_int_3(w_lya)

            # Pw2D in McDonald & Eisenstein (2007)
            self._aliasing_weights = int_2 / (int_1**2 * self.forest_length)
            # PNeff in McDonald & Eisenstein (2007)
            self._effective_noise_power = int_3 * self._pix_kms / (int_1**2 * self.forest_length)

        # tracer weights
        self._w_tracer = self._weights.compute_tracer_weights(
            f'{self._tracer1_name}_{self._tracer2_name}', self._tracer1)
        # tracer shot noise (deg^2 km/s)
        self._tracer_noise_power = 1 / self._weights.get_n_tracer(self._tracer1)

    def compute_total_power(self, k_hmpc, mu, corr):
        """Return the total observed power (signal + noise) for the given (k, mu) and correlation.

        Parameters
        ----------
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.
        corr : str
            Correlation name of the form 'tracer1_tracer2'.

        Returns
        -------
        ndarray
            Total observed power P_tot in (Mpc/h)^3.
        """
        # decompose into line of sight and transverse components
        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity
        kt_deg = kt_hmpc * self._angle_to_distance

        tracers = corr.split('_')
        if 'lya' in tracers[0] and tracers[0] == tracers[1]:
            return self._compute_total_power_lya(kt_deg, kp_kms, corr)
        elif tracers[0] == tracers[1]:
            assert corr == self.tracer_corr, 'Mismatch in tracer names'
            return self._compute_total_power_tracer(k_hmpc, mu, corr)
        else:
            return self._compute_total_power_cross(kt_deg, kp_kms, k_hmpc, mu, corr)

    def _compute_total_power_lya(self, kt_deg, kp_kms, corr):
        """Total Lya power: signal + aliasing + effective noise, in (Mpc/h)^3.

        Parameters
        ----------
        kt_deg : float or array_like
            Transverse wavenumber in deg^{-1}.
        kp_kms : float or array_like
            Line-of-sight wavenumber in s/km.
        corr : str
            Correlation name.

        Returns
        -------
        ndarray
            Total observed Lya power in (Mpc/h)^3.
        """
        p3d = self._power_spec.compute_p3d_kms_smooth(
            self._z_mean, kt_deg, kp_kms, self._res_kms, self._pix_kms, corr)
        aliasing = (
            self._aliasing_weights[-1] *
            self._power_spec.compute_p1d_kms(
                self._z_mean, kp_kms, self._res_kms, self._pix_kms, corr)
        )
        noise = self._effective_noise_power[-1]

        total_power = p3d + aliasing + noise

        total_power_mpc = total_power * self._angle_to_distance**2 / self._distance_to_velocity

        return total_power_mpc

    def _compute_total_power_tracer(self, k_hmpc, mu, corr):
        """Total discrete tracer power: signal + shot noise, in (Mpc/h)^3.

        Parameters
        ----------
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.
        corr : str
            Correlation name.

        Returns
        -------
        ndarray
            Total observed tracer power in (Mpc/h)^3.
        """
        p3d = self._power_spec.compute_p3d_hmpc(self._z_mean, k_hmpc, mu, corr)
        # in 1/deg2km/s
        density_deg2kms = self._weights.get_n_tracer(self._tracer1)[-1]
        # in 1/mpc/h^3
        density_mpc3 = density_deg2kms / self._angle_to_distance**2 * self._distance_to_velocity

        total_power_mpc = p3d + 1 / density_mpc3

        return total_power_mpc

    def _compute_total_power_cross(self, kt_deg, kp_kms, k_hmpc, mu, corr):
        """Total Lya x tracer cross-power (no noise contribution), in (Mpc/h)^3.

        Parameters
        ----------
        kt_deg : float or array_like
            Transverse wavenumber in deg^{-1}.
        kp_kms : float or array_like
            Line-of-sight wavenumber in s/km.
        k_hmpc : float or array_like
            Total wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.
        corr : str
            Correlation name.

        Returns
        -------
        ndarray
            Cross-power in (Mpc/h)^3.
        """
        p3d = self._power_spec.compute_p3d_hmpc_smooth(
            self._z_mean, k_hmpc, mu, self._pix_kms, self._res_kms, corr)

        return (p3d)

    def compute_neff_2D_lya(self, k_hmpc, mu):
        """Effective 2D density of Lya skewers (McQuinn & White 2011).

        1 / n_eff^2D = P_w + P_N / P1D.

        Parameters
        ----------
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.

        Returns
        -------
        ndarray
            Effective 2D line-of-sight density in deg^{-2}.
        """
        # decompose into line of sight component
        kp_hmpc = k_hmpc * mu
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity

        p1d = self._power_spec.compute_p1d_kms(
            self._z_mean, kp_kms, self._res_kms, self._pix_kms, self.corr)
        neff_2D_inv = self._aliasing_weights + self._effective_noise_power / p1d

        return 1 / neff_2D_inv

    def _compute_lya_eff_vol(self, k_hmpc, mu):
        """Effective survey volume for Lya (McQuinn & White 2011).

        Parameters
        ----------
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.

        Returns
        -------
        ndarray
            Effective volume in (Mpc/h)^3.
        """
        # decompose into line of sight component
        kp_hmpc = k_hmpc * mu
        # transform from comoving to observed coordinates
        kp_kms = kp_hmpc / self._distance_to_velocity

        neff_2d = self.compute_neff_2D_lya(k_hmpc, mu) / self._angle_to_distance**2

        p3d = self._power_spec.compute_p3d_hmpc(self._z_mean, k_hmpc, mu, 'lya_lya')

        p1d = (
            self._power_spec.compute_p1d_kms(
                self._z_mean, kp_kms, self._res_kms, self._pix_kms, self.corr)
            / self._distance_to_velocity
        )

        eff_vol = self._survey_volume_mpc * (p3d / (p3d + p1d / neff_2d))**2

        return eff_vol

    def compute_aliasing(self, z, kt_deg, kp_kms):
        """Compute the 3D aliasing power at (kt_deg, kp_kms).

        Parameters
        ----------
        z : float
            Redshift (not used directly, passed for interface consistency).
        kt_deg : float or array_like
            Transverse wavenumber in deg^{-1}.
        kp_kms : float or array_like
            Line-of-sight wavenumber in s/km.

        Returns
        -------
        ndarray
            Aliasing power in observed units.
        """
        p1d = self._power_spec.compute_p1d_kms(z, kp_kms, self._res_kms, self._pix_kms, self.corr)
        aliasing = self._aliasing_weights[-1] * p1d

        return aliasing

    def _compute_tracer_eff_vol(self, k_hmpc, mu):
        """Effective FKP volume for a discrete tracer in (Mpc/h)^3.

        Parameters
        ----------
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.

        Returns
        -------
        ndarray
            Effective volume in (Mpc/h)^3.
        """
        z = self._mean_z()
        dkms_dmpch = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)

        kp_hmpc = k_hmpc * mu
        kt_hmpc = k_hmpc * np.sqrt(1.0-mu**2)
        kp_kms = kp_hmpc / dkms_dmpch
        kt_deg = kt_hmpc * dhmpc_ddeg

        volume_degkms = self._survey.area_deg2 * self._get_redshift_depth()
        volume_hmpc = volume_degkms * dhmpc_ddeg**2 / dkms_dmpch

        p3d_tracer = self._power_spec.compute_p3d_kms_smooth(
            z, kt_deg, kp_kms, self._res_kms, self._pix_kms, self.corr)

        shot_noise = self._tracer_noise_power
        eff_vol = volume_hmpc * (p3d_tracer / (p3d_tracer + shot_noise))**2

        return eff_vol

    def compute_n_pk(self, k, mu):
        """Compute the signal-to-noise nP at (k, mu) for Lya and tracer.

        Parameters
        ----------
        k : float or array_like
            Wavenumber in h/Mpc.
        mu : float
            Cosine of the angle to the line of sight.

        Returns
        -------
        np_lya : ndarray
            nP for the Lya auto-correlation.
        np_tracer : ndarray
            nP for the discrete tracer auto-correlation.
        """
        z = self._mean_z()

        dkms_dmpch = self._cosmo.velocity_from_distance(z)
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)

        kp_hmpc = k * mu
        kt_hmpc = k * np.sqrt(1.0-mu**2)
        kp_kms = kp_hmpc / dkms_dmpch
        kt_deg = kt_hmpc * dhmpc_ddeg

        total_power_lya_degkms = self._compute_total_power_lya(z, kt_deg, kp_kms)
        noise_lya = total_power_lya_degkms - self._power_spec.compute_p3d_kms_smooth(
            z, kt_deg, kp_kms, self._res_kms, self._pix_kms, 'lya_lya')
        np_lya = self._power_spec.compute_p3d_kms_smooth(
            z, kt_deg, kp_kms, self._res_kms, self._pix_kms, 'lya_lya') / noise_lya[-1]

        np_tracer = self._power_spec.compute_p3d_kms_smooth(
            z, kt_deg, kp_kms, self._res_kms, self._pix_kms, self.corr
        ) / self._tracer_noise_power[-1]

        return np_lya, np_tracer
