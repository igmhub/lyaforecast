import numpy as np
from lyaforecast.analytic_biases import AnalyticBias


class PowerSpectrum:
    """Model for 1D and 3D power spectra used in Fisher forecasts.

    Uses CAMB for the linear matter power and McDonald (2003) analytic biases for the
    Lya flux power. All units are internally in h/Mpc unless otherwise noted.
    """

    def __init__(self, config, cosmo, spectrograph):
        """
        Parameters
        ----------
        config : configparser.ConfigParser
            Parsed configuration object with a ``[power spectrum]`` section.
        cosmo : CosmoCamb
            Cosmological model instance.
        spectrograph : dict or Spectrograph
            Spectrograph instance(s) keyed by correlation name.
        """
        # define cosmology class
        self._cosmo = cosmo
        self._spectrograph = spectrograph
        self._growth_rate = self._cosmo.growth_rate
        # get analytical biases
        self.bias = AnalyticBias(self._cosmo)

        # power spectrum calculation details
        _properties = config['power spectrum']
        self._k_min_hmpc = _properties.getfloat('k_min_hmpc', 1e-4)
        self._k_max_hmpc = _properties.getfloat('k_max_hmpc', 1)
        self._num_k_bins = _properties.getint('num_k_bins', 100)
        self._mu_min = _properties.getfloat('mu_min', 0)
        self._mu_max = _properties.getfloat('mu_max', 1)
        self._num_mu_bins = _properties.getint('num_mu_bins', 10)
        # True to ignore small-scale components of power spectra.
        self._linear = _properties.getboolean('linear power')

        # grids for evaluating power spectrum
        self.k = np.linspace(self._k_min_hmpc, self._k_max_hmpc, self._num_k_bins)
        self.logk = np.log(self.k)
        self.dk = self.k[1] - self.k[0]
        self.dlogk = self.dk / self.k

        # bin edges
        mu_edges = np.linspace(self._mu_min, self._mu_max, self._num_mu_bins+1)
        self.mu = mu_edges[:-1] + np.diff(mu_edges)/2
        self.dmu = np.diff(mu_edges)[0]

    def compute_linear_power_evol(self, z, k_hmpc):
        """Return the linear matter power scaled from z_ref to z using EdS growth.

        Parameters
        ----------
        z : float
            Target redshift (must be >= 1.8).
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.

        Returns
        -------
        ndarray
            P_lin(z, k) in (Mpc/h)^3.
        """
        if z < 1.8:
            print('Warning, going below z = 1.8 with EdS power scaling')
        if self._cosmo.z_ref < 1.8:
            raise ValueError('Can not have z_ref below 1.8, input:', self._cosmo.z_ref)

        pk_zref = self._cosmo.get_pk_lin(k_hmpc, self._k_min_hmpc, self._k_max_hmpc)

        eds = ((1+self._cosmo.z_ref)/(1+z))**2

        return pk_zref * eds

    def compute_p1d_kms(self, z, kp_kms, res_kms, pix_kms, corr):
        """1D Lya power spectrum in observed (km/s) coordinates, smoothed by pixel and resolution.

        Parameters
        ----------
        z : float
            Redshift.
        kp_kms : float or array_like
            Line-of-sight wavenumber in s/km.
        res_kms : float
            Spectrograph resolution in km/s.
        pix_kms : float
            Pixel width in km/s.
        corr : str
            Correlation name used to select the spectrograph instance.

        Returns
        -------
        ndarray
            Smoothed P1D in (km/s) units.
        """
        # get P1D before smoothing
        p1d_kms = self.compute_p1d_palanque2013(z, kp_kms)
        # smoothing (pixelization and resolution)

        kernel = self._spectrograph[corr].smooth_kernel_kms(pix_kms, res_kms, kp_kms)
        p1d_kms *= (kernel**2)

        return p1d_kms

    def compute_p3d_kms_smooth(self, z, kt_deg, kp_kms, res_kms, pix_kms, corr):
        """3D power spectrum in observed (deg, km/s) coordinates, smoothed by pixel and resolution.

        Parameters
        ----------
        z : float
            Redshift.
        kt_deg : float or array_like
            Transverse wavenumber in deg^{-1}.
        kp_kms : float or array_like
            Line-of-sight wavenumber in s/km.
        res_kms : float
            Spectrograph resolution in km/s.
        pix_kms : float
            Pixel width in km/s.
        corr : str
            Correlation name (e.g. 'lya_lya', 'lya_qso').

        Returns
        -------
        ndarray
            Smoothed P3D in deg^2 km/s units.
        """
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
        p3d_hmpc = self.compute_p3d_hmpc(z, k_hmpc, mu, corr)
        # convert power to observed units
        p3d_degkms = p3d_hmpc * dkms_dhmpc / dhmpc_ddeg**2

        # smoothing (pixelization and resolution)
        tracers = corr.split('_')
        for t in tracers:
            if 'lya' in t:
                kernel = self._spectrograph[corr].smooth_kernel_kms(pix_kms, res_kms, kp_kms)
                p3d_degkms *= kernel

        return p3d_degkms

    def compute_p3d_hmpc(self, z, k_hmpc, mu, corr):
        """3D flux power spectrum P_F(z, k, mu) in (Mpc/h)^3.

        Parameters
        ----------
        z : float
            Redshift.
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float or array_like
            Cosine of the angle to the line of sight.
        corr : str
            Correlation name used to compute the bias.

        Returns
        -------
        ndarray
            P3D in (Mpc/h)^3.
        """
        # get linear power at zrefs
        k = np.fmax(k_hmpc, self._k_min_hmpc)
        k = np.fmin(k, self._k_max_hmpc)
        # compute redshift-evolved linear matter power spectrum
        pk_zref = self.compute_linear_power_evol(z, k_hmpc)
        # get flux scale-dependent biasing (or only linear term)
        b = self.bias.compute_bias(z, k, mu, corr, self._linear)

        return pk_zref * b

    def compute_p3d_hmpc_smooth(self, z, k_hmpc, mu, pix_kms, res_kms, corr):
        """P3D in (Mpc/h)^3, smoothed via conversion to observed coordinates and back.

        Parameters
        ----------
        z : float
            Redshift.
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        mu : float or array_like
            Cosine of the angle to the line of sight.
        pix_kms : float
            Pixel width in km/s.
        res_kms : float
            Spectrograph resolution in km/s.
        corr : str
            Correlation name.

        Returns
        -------
        ndarray
            Smoothed P3D in (Mpc/h)^3.
        """
        # conversions
        dhmpc_ddeg = self._cosmo.distance_from_degrees(z)
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        kp_hmpc = k_hmpc * mu
        kp_kms = kp_hmpc / dkms_dhmpc

        # get linear power at zrefs
        p3d_hmpc = self.compute_p3d_hmpc(z, k_hmpc, mu, corr)

        if 'lya' not in corr:
            return p3d_hmpc

        tracers = corr.split('_')
        p3d_degkms = p3d_hmpc * dkms_dhmpc / dhmpc_ddeg**2
        for t in tracers:
            if 'lya' in t:
                kernel = self._spectrograph[corr].smooth_kernel_kms(pix_kms, res_kms, kp_kms)
                p3d_degkms *= kernel

        p3d_hmpc = p3d_degkms * dhmpc_ddeg**2 / dkms_dhmpc

        return p3d_hmpc

    def compute_p1d_palanque2013(self, z, k_kms):
        """Fitting formula for P1D(z, k) from Palanque-Delabrouille et al. (2013).

        Corrected to be flat at low-k instead of going to zero.

        Parameters
        ----------
        z : float
            Redshift.
        k_kms : float or array_like
            Wavenumber in s/km.

        Returns
        -------
        ndarray
            1D Lya power spectrum in (km/s) units.
        """
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
        k_min = k0*np.exp((-0.5*n_F_z-1)/alpha_F)
        k_kms = np.fmax(k_kms, k_min)
        exp1 = 3 + n_F_z + alpha_F * np.log(k_kms/k0)
        toret = np.pi * A_F / k0 * pow(k_kms/k0, exp1-1) * pow((1+z)/(1+z0), B_F)
        return toret

    def compute_p1d_hmpc(self, z, k_hmpc, res_hmpc=None, pix_hmpc=None):
        """Analytical P1D in h/Mpc units (currently unused).

        Parameters
        ----------
        z : float
            Redshift.
        k_hmpc : float or array_like
            Wavenumber in h/Mpc.
        res_hmpc : float, optional
            Gaussian resolution scale in Mpc/h; applies exp(-k^2 res^2) smoothing.
        pix_hmpc : float, optional
            Pixel size in Mpc/h; applies sinc^2 smoothing.

        Returns
        -------
        ndarray
            P1D in (Mpc/h) units.
        """
        # transform to km/s
        dkms_dhmpc = self._cosmo.velocity_from_distance(z)
        k_kms = k_hmpc / dkms_dhmpc
        # get analytical P1D from Palanque-Delabrouille (2013)
        power_kms = self.compute_p1d_palanque2013(z, k_kms)
        power_hmpc = power_kms / dkms_dhmpc
        if res_hmpc:
            # smooth with Gaussian
            power_hmpc *= np.exp(-pow(k_hmpc * res_hmpc, 2))
        if pix_hmpc:
            # smooth with Top Hat
            kpix = np.fmax(k_hmpc * pix_hmpc, 1.e-5)
            power_hmpc *= pow(np.sin(kpix/2)/(kpix/2), 2)

        return power_hmpc
