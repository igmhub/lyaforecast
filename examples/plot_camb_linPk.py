import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower

# setup cosmological model
pars = camb.CAMBparams()
#one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9,ns=0.965, r=0)
#setup to compute power at z=0,2,5
pars.set_matter_power(redshifts=[0.,2.,5.], kmax=10.0)

#compute linear power spectrum
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh,zs,pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=200)
s8 = np.array(results.get_sigma8())

for i, z in enumerate(zs):
    plt.loglog(kh, pk[i,:],label='z = '+str(int(z)))
plt.xlabel('k [h/Mpc]')
plt.ylabel('P(k) [Mpc/h]')
plt.legend(loc='lower left')
plt.title('Matter power at different redshifts')
plt.show()

