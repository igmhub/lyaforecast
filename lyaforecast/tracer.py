import numpy as np
from scipy.interpolate import RectBivariateSpline,interp1d
from lyaforecast.utils import get_file

TRACER_OPTIONS = ['qso', 'elgqso', 'lbg', 'lae']
CONTINUOUS_TRACERS = ['lya']


class Tracer:
    """Tracer description holding the dn/dzdm distribution and bias information."""

    background_tracer = None

    def __init__(self, config):
        """
        Parameters
        ----------
        config : configparser.SectionProxy
            Configuration section for this tracer.
        """
        self.config = config
        self.simple_name = config.get('tracer')
        self.type = config.get('tracer_type')

        # tracer density
        self.tracer_density = self.config.getfloat('target density')

        # tracer magnitude range
        self.mag_min = self.config.getfloat('min_band_mag', None)
        self.mag_max = self.config.getfloat('max_band_mag', None)

        self.background_tracer = None
        tracer_dzdz_file = get_file(self.config.get('dn dz'))

        if self.type == 'continuous':
            self.init_continuous_tracer(tracer_dzdz_file)
        elif self.type == 'discrete':
            self.init_discrete_tracer(tracer_dzdz_file)
        else:
            raise ValueError(f'Unknown tracer type: {self.type}')

        if self.background_tracer is not None:
            self.name = f'{self.simple_name}({self.background_tracer})'
        else:
            self.name = self.simple_name

        self.bias_func = None

        bias_z_str = self.config.get("bias z")
        if bias_z_str is not None :
            bias_val_str = self.config.get("bias val")
            if bias_val_str is None :
                raise(KeyError("need either both 'bias z' and 'bias val' or none in tracer config"))
            bias_z = np.array([float(val) for val in bias_z_str.split(" ")])
            bias_val = np.array([float(val) for val in bias_val_str.split(" ")])
            print("bias z=",bias_z)
            print("bias val=",bias_val)
            self.bias_func = interp1d(
                bias_z, bias_val, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )

    def get_dn_dzdm(self, z, m):
        """Return the tracer number density per unit redshift and magnitude.

        Parameters
        ----------
        z : float or array_like
            Redshift.
        m : float or array_like
            Apparent magnitude (same shape as z for non-grid evaluation).

        Returns
        -------
        ndarray
            dn/dzdm in deg^{-2} per unit z per unit magnitude.
        """
        points = self._tracer_dndz(z, m, grid=False)
        points[m > self._mmax] = 1e-20
        points[m < self._mmin] = 1e-20

        return points

    def init_discrete_tracer(self, tracer_dzdz_file):
        """Initialise the dn/dzdm interpolator for a discrete (galaxy/quasar) tracer.

        Parameters
        ----------
        tracer_dzdz_file : Path
            Path to the flat text file containing (z, m, dn/dzdm/deg^2) columns.
        """
        if self.simple_name not in TRACER_OPTIONS:
            raise ValueError(f'Unknown discrete tracer: {self.simple_name}')

        self._tracer_dndz = self._setup_dndzdm_tracer(tracer_dzdz_file)

    def init_continuous_tracer(self, tracer_dzdz_file):
        """Initialise the dn/dzdm interpolator for a continuous (Lya forest) tracer.

        Parameters
        ----------
        tracer_dzdz_file : Path
            Path to the flat text file containing (z, m, dn/dzdm/deg^2) columns.
        """
        if self.simple_name not in CONTINUOUS_TRACERS:
            raise ValueError(f'Unknown continuous tracer: {self.simple_name}')

        self.background_tracer = self.config.get('background_tracer')
        if self.background_tracer not in TRACER_OPTIONS:
            raise ValueError(f'Unknown background tracer: {self.background_tracer}')

        # definition of forest (lrmin,lrmax)
        self.lrmin = self.config.getfloat('min_rest_frame_lya')
        self.lrmax = self.config.getfloat('max_rest_frame_lya')

        # number of exposures (for snr of forest)
        self.num_exp = self.config.getfloat('num exposures')

        # pixel width (in angstroms or kms)
        self.pix_kms = self.config.getfloat('pix_width_kms', None)
        self.pix_ang = self.config.getfloat('pix_width_ang', None)

        self._tracer_dndz = self._setup_dndzdm_lya(tracer_dzdz_file)

    def _setup_dndzdm_lya(self, file):
        """Build a bivariate spline interpolator for Lya background-source dn/dzdm.

        Parameters
        ----------
        file : Path
            Text file with columns (z, m, dn/dzdm/deg^2).

        Returns
        -------
        RectBivariateSpline
            Interpolator for dn/dzdm in deg^{-2} per unit z per unit magnitude.
        """
        z, m, tdNdmdzddeg2 = np.loadtxt(file, unpack=True)

        if self.mag_min is not None :
            print(f"Enforcing m>={self.mag_min}")
            tdNdmdzddeg2 *= (m>=self.mag_min)

        if self.mag_max is not None :
            print(f"Enforcing m<={self.mag_max}")
            tdNdmdzddeg2 *= (m<=self.mag_max)

        # scale density of quasars to desired number. By default given staright from QLF
        if self.tracer_density is not None:
            z_min_lya = 2.15
            current_total_density = np.sum(tdNdmdzddeg2*(z > z_min_lya))
            print(
                f"Scaling lya dndzdm from a total density (z>={z_min_lya}) of "
                f"{current_total_density:.1f} to {self.tracer_density:.1f}/deg2"
            )
            tdNdmdzddeg2 *= (self.tracer_density/current_total_density)

        z = np.unique(z)
        m = np.unique(m)

        # This assumes entries are evenly spaced.
        dz = z[1] - z[0]
        dm = m[1] - m[0]

        tdNdmdzddeg2 /= (dz*dm)
        tdNdmdzddeg2 = np.reshape(tdNdmdzddeg2, [len(z), len(m)])

        # figure out allowed redshift range (will check out of bounds)
        self._zmin = z[0]
        self._zmax = z[-1]
        # figure out allowed magnitude range (will check out of bounds)
        self._mmin = m[0]
        self._mmax = m[-1]

        interpolator = RectBivariateSpline(
            z, m, tdNdmdzddeg2, bbox=[self._zmin, self._zmax, self._mmin, self._mmax],
            kx=2, ky=2
        )

        return interpolator

    def _setup_dndzdm_tracer(self, file):
        """Build a bivariate spline interpolator for a discrete tracer dn/dzdm.

        Parameters
        ----------
        file : Path
            Text file with columns (z, m, dn/dzdm/deg^2).

        Returns
        -------
        RectBivariateSpline
            Interpolator for dn/dzdm in deg^{-2} per unit z per unit magnitude.
        """
        z_arr, m_arr, tdNdmdzddeg2 = np.loadtxt(file, unpack=True)
        z = np.unique(z_arr)
        m = np.unique(m_arr)

        # scale density of quasars to desired number. By default taken straight from QLF
        if self.tracer_density is not None:
            # re-scale based on lya qso requirements
            if self.simple_name == 'qso':
                current_total_density = np.sum(tdNdmdzddeg2.reshape(z.size, m.size)[z > 2.15])
            else:
                current_total_density = np.sum(tdNdmdzddeg2.reshape(z.size, m.size))
            print(
                f"Scaling dndzdm tracer from a total density of "
                f"{current_total_density} to {self.tracer_density}/deg2"
            )
            tdNdmdzddeg2 *= (self.tracer_density/current_total_density)

        # This assumes entries are evenly spaced.
        dz = z[1] - z[0]
        dm = m[1] - m[0]

        tdNdmdzddeg2 /= (dz*dm)
        tdNdmdzddeg2 = np.reshape(tdNdmdzddeg2, [len(z), len(m)])

        # figure out allowed redshift range (will check out of bounds)
        self._zmin = z[0]
        self._zmax = z[-1]
        # figure out allowed magnitude range (will check out of bounds)
        self._mmin = m[0]
        self._mmax = m[-1]

        interpolator = RectBivariateSpline(
            z, m, tdNdmdzddeg2, bbox=[self._zmin, self._zmax, self._mmin, self._mmax],
            kx=2, ky=2
        )
        return interpolator
