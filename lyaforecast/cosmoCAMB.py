import numpy as np
import camb
from lyaforecast.utils import get_file


class CosmoCamb:
    """Compute cosmological functions using CAMB."""

    SPEED_LIGHT = 2.998e5  # km/s
    LYA_REST = 1215.67  # Angstroms

    def __init__(self, ini, z_ref=None, z_centres=[]):
        """
        Parameters
        ----------
        ini : str
            Path to a CAMB .ini configuration file.
        z_ref : float, optional
            Reference redshift for P_lin evaluation.
        z_centres : array_like, optional
            Redshift bin centres for per-bin sigma8 and growth rate.
        """
        config = get_file(ini)
        self._pars = camb.read_ini(str(config))

        # Set effective redshift of survey
        if z_ref is None:
            if self._pars.Transfer.PK_num_redshifts == 0:
                raise ValueError(
                    "You must specify at least one reference"
                    "redshift to evaluate the power spectrum."
                )
            else:
                # This will raise an error with CAMB anyway, but here for clarity.
                assert len(self._pars.Transfer.PK_redshifts) == self._pars.Transfer.PK_num_redshifts
                print('Using first transfer redshift as z_ref')
                z_ref = self._pars.Transfer.PK_redshifts[0]
        else:
            self._pars.Transfer.PK_redshifts = [z_ref]
            self._pars.Transfer.PK_num_redshifts = 1

        self.z_ref = z_ref
        self.results = camb.get_results(self._pars)

        self.growth_rate = self.results.get_fsigma8()[0] / self.results.get_sigma8()[0]
        self.sigma8 = self.results.get_sigma8()[0]

        z_bins = np.array(z_centres)
        z_bins[::-1].sort()

        self._pars.Transfer.PK_redshifts = list(z_bins)
        self._pars.Transfer.PK_num_redshifts = len(z_bins)
        self.results_bins = camb.get_results(self._pars)
        self.z_bins = np.array(z_centres)
        self.sigma8_zbins = np.array(self.results_bins.get_sigma8())[::-1]
        self.growth_factor_ratios = self.sigma8_zbins / self.sigma8
        self.growth_rate_zbins = np.array(self.results_bins.get_fsigma8())[::-1] / self.sigma8_zbins

    def get_pk_lin(self, k, kmin=1.e-4, kmax=1.e1, npoints=1000):
        """Return the linear matter power spectrum at z_ref, interpolated onto k.

        Parameters
        ----------
        k : float or array_like
            Wavenumber in h/Mpc.
        kmin : float, optional
            Minimum wavenumber for the CAMB matter power spectrum grid.
        kmax : float, optional
            Maximum wavenumber for the CAMB matter power spectrum grid.
        npoints : int, optional
            Number of points in the CAMB matter power spectrum grid.

        Returns
        -------
        ndarray
            Linear power spectrum P(k) in (Mpc/h)^3.
        """
        kh, _, pk = self.results.get_matter_power_spectrum(
            minkh=kmin, maxkh=kmax, npoints=npoints)
        return np.interp(k, kh, pk[0, :])

    def velocity_from_distance(self, z):
        """Conversion factor from Mpc/h to km/s at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            dv/dr in (km/s) / (Mpc/h).
        """
        return (self.results.hubble_parameter(z) / self._pars.H0 / (1 + z)) * 100.0

    def velocity_from_wavelength(self, z):
        """Conversion factor from observed wavelength to km/s at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            dv/dlambda in (km/s) / Angstrom.
        """
        return self.SPEED_LIGHT / self.LYA_REST / (1+z)

    def distance_from_wavelength(self, z):
        """Conversion factor from observed wavelength to Mpc/h at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            dr/dlambda in (Mpc/h) / Angstrom.
        """
        return self.velocity_from_wavelength(z) / self.velocity_from_distance(z)

    def distance_from_degrees(self, z):
        """Conversion factor from degrees to Mpc/h at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            dr/dtheta in (Mpc/h) / degree.
        """
        dmpc_drad = self.results.angular_diameter_distance(z) * (1+z)
        return dmpc_drad * (np.pi/180.0) * (self._pars.H0 / 100.0)
