import numpy as np
import camb
from lyaforecast.utils import get_file

class CosmoCamb:
    """Compute cosmological functions using CAMB"""
    
    SPEED_LIGHT = 2.998e5 #km/s
    LYA_REST = 1215.67 #Angstroms

    def __init__(self,ini,z_ref):
        """Setup cosmological model.
        Reference z is set in config with other parameters"""

        config = get_file(ini)
        self._pars = camb.read_ini(str(config))

        # Set effective redshift of survey
        if z_ref is None:
            if self._pars.Transfer.PK_num_redshifts == 0:
                raise ValueError("You must specify at least one reference"
                                "redshift to evaluate the power spectrum.")
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

    def get_pk_lin(self,k,kmin=1.e-4,kmax=1.e1,npoints=1000):
        """Return linear power interpolator in units of h/Mpc, at zref"""
        kh,_,pk = self.results.get_matter_power_spectrum(minkh=kmin,
                                                maxkh=kmax,npoints=npoints)
        return np.interp(k,kh,pk[0,:])

    def velocity_from_distance(self,z):
        """Conversion factor from Mpc/h to km/s, at redshift z."""
        breakpoint()
        return (self.results.hubble_parameter(z) / self._pars.H0 / (1 + z)) * 100.0

    def velocity_from_wavelength(self,z):
        """Conversion factor from lambda_obs to km/s, at redshift z."""
        return self.SPEED_LIGHT / self.LYA_REST / (1+z) 

    def distance_from_wavlength(self,z):
        """Conversion factor from lambda_obs to Mpc/h, at redshift z."""
        return self.velocity_from_wavelength(z) / self.velocity_from_distance(z)

    def distance_from_degrees(self,z):
        """Conversion factor from degrees to Mpc/h, at redshift z."""
        dmpc_drad = self.results.angular_diameter_distance(z) * (1+z)
        #print('dMpc_drad',dMpc_drad)
        return dmpc_drad * (np.pi/180.0) * (self._pars.H0 / 100.0)

