import numpy as np
import matplotlib.pyplot as plt
import linear_power_CLASS as lpC
#import analytic_bias_McD2003 as bM03

z=3.0
kmin=0.0001
kmax=10.0
power=lpC.LinearPower_hMpc(z,kmin,kmax)
print 'P(k=1.0)',power(1.0)
k=np.logspace(-4,1,1000)
pk=power(k)

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

plt.ylim(1e2,1e5)
plt.xlim(1e-4,1e0)
plt.xlabel(r'k [h $\rm{Mpc}^{-1}$]',fontsize=15)
plt.ylabel(r'P(k) [$\rm{Mpc}^3 \rm{h}^{-3}$]',fontsize=15)
plt.title(r'Density power at z='+str(z),fontsize=15)
plt.loglog(k,pk)
#plt.loglog(k,pk*bM03.bias_hMpc_McD2003(k,np.zeros_like(k)))
#plt.loglog(k,pk*bM03.bias_hMpc_McD2003(k,np.ones_like(k)))
plt.show()


