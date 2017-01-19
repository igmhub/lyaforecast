import numpy as np
from classy import Class
from scipy.interpolate import InterpolatedUnivariateSpline

class Cosmology(object):
  """Class to compute cosmological functions using CLASS"""

  def __init__(self):
    """Setup cosmological model, at some point passing parameters"""

    self.params = {                     
      'output': 'mPk',                     
      'non linear':'no',
      'lensing': 'no',
      'P_k_max_1/Mpc': 150.0,                  
      'A_s': 2.3e-9,
      'n_s': 0.9624, 
      'h': 0.6711,                                                
      'omega_b': 0.022068,                                    
      'omega_cdm': 0.12029}    

    # Create an instance of the CLASS wrapper
    self.cosmo = Class()  
    # Set the parameters to the cosmological code
    self.cosmo.set(self.params) 
    # Run CLASS
    self.cosmo.compute()

  def LinearPower_hMpc(self,z,kmin,kmax):
    self.cosmo.set({'z_pk' : z})
    self.cosmo.compute()
    k_hMpc = np.logspace(np.log10(kmin),np.log10(kmax),1000)
    h = self.params['h']
    k_Mpc = k_hMpc * h
    # power spectrum of matter at z, in units of Mpc
    P_Mpc =[self.cosmo.pk(kk,z) for kk in k_Mpc]
    P_hMpc = np.asarray(P_Mpc) / pow(h,3)
    # interpolation object
    return InterpolatedUnivariateSpline(k_hMpc,P_hMpc)

  def dkms_dhMpc(self,z):
    # CLASS returns H(z) in units of 1/Mpc, you need to multiply by c_kms
    H = self.cosmo.Hubble(z)*2.9979e5
    h = self.params['h']
    return H/h/(1+z)

