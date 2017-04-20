import numpy as np
import scipy.interpolate
import analytic_bias_McD2003 as bM03
import camb
from camb import model, initialpower

class TheoryLyaP3D(object):
  """Make predictions for Lyman alpha 3D P(z,k,mu).
    Should only be used at the level of Fisher forecasts.
    Uses CAMB to generate linear power, and McDonald (2003) for Lya stuff.
    All units internally are in h/Mpc."""

  def __init__(self):
    # get linear power spectrum at z=2.25 from CAMB, in h/Mpc
    self.zref=2.25

    #one massive neutrino and helium set using BBN consistency
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=0.06)
    pars.InitPower.set_params(As=2e-9,ns=0.965, r=0)
    pars.set_matter_power(redshifts=[self.zref], kmax=10.0)

    #compute and store linear power spectrum (at zref)
    pars.NonLinear = model.NonLinear_none
    self.results = camb.get_results(pars)
    self.kmin=1.e-4
    self.kmax=1.e1
    kh,_,pk = self.results.get_matter_power_spectrum(minkh=self.kmin,
                                                     maxkh=self.kmax,npoints=1000)
    self.linPk = scipy.interpolate.interp1d(kh,pk[0,:])

  def FluxP3D_hMpc(self,z,k_hMpc,mu,linear=False):
    """If linear=True, ignore small scale correction"""
    # get linear power at zref
    k = np.fmax(k_hMpc,self.kmin)
    k = np.fmin(k,self.kmax)
    P = self.linPk(k)
    # get flux scale-dependent biasing (or only linear term)
    b = bM03.bias_hMpc_McD2003(k,mu,linear)
    # get (approximated) redshift evolution
    zevol = pow( (1+z)/(1+self.zref), 3.8)
    return P * b * zevol

