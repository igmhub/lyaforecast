import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB as cCAMB 

# setup cosmological model
zref=2.25
cosmo = cCAMB.Cosmology(zref)
# compute linear power at zref
kmin=1.e-4
kmax=1.e1
linPk = cosmo.LinPk_hMpc(kmin,kmax,1000)
# plot power 
k = np.logspace(np.log10(kmin),np.log10(0.99*kmax),1000)
plt.loglog(k, linPk(k))
plt.xlabel('k [h/Mpc]')
plt.ylabel('P(k) [Mpc/h]')
plt.title('Linear density power at z ='+str(zref))
plt.show()

