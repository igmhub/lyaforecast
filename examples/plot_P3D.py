import numpy as np
import matplotlib.pyplot as plt
import theoryLyaP3D as tP3D

theory=tP3D.TheoryLyaP3D()

z=3.0
k=np.logspace(-4,0.9,1000)
pk_0=theory.FluxP3D_hMpc(z,k,np.zeros_like(k))
pk_1=theory.FluxP3D_hMpc(z,k,np.ones_like(k))
pk_0_lin=theory.FluxP3D_hMpc(z,k,np.zeros_like(k),linear=True)
pk_1_lin=theory.FluxP3D_hMpc(z,k,np.ones_like(k),linear=True)

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

plt.ylim(1e0,2e3)
plt.xlim(1e-3,1e0)
plt.xlabel(r'k [h $\rm{Mpc}^{-1}$]',fontsize=15)
plt.ylabel(r'$\rm{P}_{\rm{F}}(k)$ [$\rm{Mpc}^3 \rm{h}^{-3}$]',fontsize=15)
plt.title(r'Flux power at z='+str(z),fontsize=15)
plt.loglog(k,pk_0,label=r'$\mu=0$')
plt.loglog(k,pk_1,label=r'$\mu=1$')
plt.loglog(k,pk_0_lin,label=r'$\mu=0$, lin')
plt.loglog(k,pk_1_lin,label=r'$\mu=1$, lin')
plt.legend(loc='lower left',fontsize=12)
plt.show()
