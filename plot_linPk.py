import numpy as np
import matplotlib.pyplot as plt
import cosmoCLASS


kmin=0.0001
kmax=10.0
k=np.logspace(-4,1,1000)

cosmo=cosmoCLASS.Cosmology()
plt.loglog(k,cosmo.LinearPower_hMpc(0.0,kmin,kmax)(k),label='z=0.0')
plt.loglog(k,cosmo.LinearPower_hMpc(2.0,kmin,kmax)(k),label='z=2.0')
plt.loglog(k,cosmo.LinearPower_hMpc(4.0,kmin,kmax)(k),label='z=4.0')

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [
  r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need
  r'\sisetup{detect-all}',   # this to force siunitx to actually use your fonts
  r'\usepackage{helvet}',    # set the normal font here
  r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
  r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

plt.legend()
plt.ylim(1e2,1e6)
plt.xlim(1e-4,1e0)
plt.xlabel(r'k [h $\rm{Mpc}^{-1}$]',fontsize=15)
plt.ylabel(r'P(k) [$\rm{Mpc}^3 \rm{h}^{-3}$]',fontsize=15)
plt.title(r'Density power spectrum',fontsize=15)
plt.savefig('linPk.png')
plt.show()

