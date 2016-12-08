# adapted from a notebook by Francisco Javier Sanchez Lopez (UCI, Dec 2016)

import numpy as np
import matplotlib.pyplot as plt
from classy import Class

params = {                     
  'output': 'mPk',                     
  'non linear':'yes',
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

# Run the whole code. Depending on your output, it will call the
# CLASS modules more or less fast. For instance, without any
# output asked, CLASS will only compute background quantities,
# thus running almost instantaneously.
# This is equivalent to the beginning of the `main` routine of CLASS,
# with all the struct_init() methods called.
cosmo.compute()
z=0
k=np.logspace(-4,1,1000)
# power spectrum of matter at z=0, in units of Mpc
pk =[cosmo.pk(kk,z) for kk in k]

plt.ylim(1e3,1e5)
plt.xlim(1e-4,1e0)
plt.xlabel("k [1/Mpc]")
plt.ylabel("P(k) [Mpc^3]")
plt.title("Density power at z=0")
plt.loglog(k,pk)
plt.show()


