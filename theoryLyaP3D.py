import numpy as np
from classy import Class
import analytic_bias_McD2003 as bM03
import cosmoCLASS as cosmo

class TheoryLyaP3D(object):
  """Class to make predictions for Lyman alpha 3D P(z,k,mu).
    Should only be used at the level of Fisher forecasts.
    Uses CLASS to generate linear power, and McDonald (2003) for Lya stuff.
    All units internally are in h/Mpc."""

  def __init__(self):
    # get linear power spectrum at z=2.25 from CLASS, in h/Mpc
    self.zref=2.25
    self.kmin=0.0001
    self.kmax=10.0
    self.cosmo = cosmo.Cosmology()
    self.Plin=self.cosmo.LinearPower_hMpc(self.zref,self.kmin,self.kmax)

  def FluxP3D_hMpc(self,z,k_hMpc,mu,linear=False):
    """If linear=True, ignore small scale correction"""
    # get linear power at zref
    k = np.fmax(k_hMpc,self.kmin)
    k = np.fmin(k,self.kmax)
    P = self.Plin(k)
    # get flux scale-dependent biasing (or only linear term)
    b = bM03.bias_hMpc_McD2003(k,mu,linear)
    # get (approximated) redshift evolution
    zevol = pow( (1+z)/(1+self.zref), 3.8)
    return P * b * zevol

