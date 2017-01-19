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

  def dkms_dhMpc_z(self,z):
    # CLASS returns H(z) in units of 1/Mpc, you need to multiply by c_kms
    H = self.cosmo.Hubble(z)*2.9979e5
    h = self.cosmo.h()
    return H/h/(1+z)

  def dkms_dlobs_z(self,z):
    c_kms = 2.9979e5
    lya_A = 1215.67
    return c_kms / lya_A / (1+z)

  def dhMpc_dlobs_z(self,z):
    return self.dkms_dlobs_z(z) / self.dkms_dhMpc_z(z)

  def dhMpc_ddeg_z(self,z):
    # angular diameter distance in CLASS is physical Mpc
    DA_Mpc = self.cosmo.angular_distance(z)
    h = self.cosmo.h()
    # get comoving distance
    dhMpc_drad = DA_Mpc
    dhMpc_drad = DA_Mpc * (1+z)
    dhMpc_drad = DA_Mpc * (1+z) * h
    # convert from radians to degrees
    ddeg_drad = 180.0 / 3.141592
    return dhMpc_drad / ddeg_drad

  def hMpc_zmin_zmax(self,zmin,zmax):
    Lmin_Mpc = self.cosmo.angular_distance(zmin)*(1+zmin)
    Lmax_Mpc = self.cosmo.angular_distance(zmax)*(1+zmax)
    return (Lmax_Mpc-Lmin_Mpc) * self.cosmo.h()

