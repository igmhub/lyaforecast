import numpy as np
from classy import Class
from scipy.interpolate import InterpolatedUnivariateSpline

def LinearPower_hMpc(z,kmin,kmax):
  print 'in LinearPower_hMpc, z',z
  params = {                     
    'output': 'mPk',                     
    'z_pk' : z,
    'non linear':'no',
    'lensing': 'no',
    'P_k_max_1/Mpc': 150.0,                  
    'A_s': 2.3e-9,
    'n_s': 0.9624, 
    'h': 0.6711,                                                
    'omega_b': 0.022068,                                    
    'omega_cdm': 0.12029}    

  # Create an instance of the CLASS wrapper
  cosmo = Class()  
  # Set the parameters to the cosmological code
  cosmo.set(params) 
  # Run CLASS
  cosmo.compute()

  k_hMpc = np.logspace(np.log10(kmin),np.log10(kmax),1000)
  h = params['h']
  k_Mpc = k_hMpc * h
  # power spectrum of matter at z, in units of Mpc
  P_Mpc =[cosmo.pk(kk,z) for kk in k_Mpc]
  P_hMpc = np.asarray(P_Mpc) / pow(h,3)
  # interpolation object
  return InterpolatedUnivariateSpline(k_hMpc,P_hMpc)

