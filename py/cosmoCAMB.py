import numpy as np
import scipy.interpolate
import camb
from camb import model, initialpower

class Cosmology(object):
    """Compute cosmological functions using CAMB.

       Plenty of room for improvement, for now just something to get started."""

    def __init__(self,pk_zref=None):
        """Setup cosmological model.

            If pk_zref is set, it will compute linear power at z=pk_zref."""
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=67.5,ombh2=0.022,omch2=0.122)
        self.pars.InitPower.set_params(As=2e-9,ns=0.965, r=0)
        self.pk_zref=pk_zref
        if self.pk_zref:
            self.pars.set_matter_power(redshifts=[self.pk_zref], kmax=10.0)
            # compute and store linear power spectrum (at zref)
            self.pars.NonLinear = model.NonLinear_none
            self.results = camb.get_results(self.pars)
        else:
            # compute only background expansion
            self.results = camb.get_results(self.pars)

    def LinPk_hMpc(self,kmin,kmax,npoints):
        """Return linear power interpolator in units of h/Mpc, at zref"""
        if self.pk_zref:
            kh,_,pk = self.results.get_matter_power_spectrum(minkh=kmin,
                                                        maxkh=kmax,npoints=npoints)
            return scipy.interpolate.interp1d(kh,pk[0,:])
        else:
            print('if you want LinPk_hMpc, initialize Cosmology with pk_zref')
            raise SystemExit

    def dkms_dhMpc(self,z):
        """Convertion factor from Mpc/h to km/s, at redshift z."""
        return self.results.hubble_parameter(z)/self.pars.H0/(1+z)*100

