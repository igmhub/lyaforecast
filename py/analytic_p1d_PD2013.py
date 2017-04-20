import numpy as np

def P1D_z_kms_PD2013(z,k_kms):
  """Fitting formula for 1D P(z,k) from Palanque-Delabrouille et al. (2013).
    Wavenumbers and power in units of km/s. Corrected to be flat at low-k"""
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
  k_min=k0*np.exp((-0.5*n_F_z-1)/alpha_F)
  k_kms = np.fmax(k_kms,k_min)
  exp1 = 3 + n_F_z + alpha_F * np.log(k_kms/k0)
  toret = 3.141592 * A_F / k0 * pow(k_kms/k0, exp1-1) * pow((1+z)/(1+z0), B_F)
  return toret
