import numpy as np
import matplotlib.pyplot as plt
import qso_LF as qLF 

LF = qLF.QuasarLF()

mmin,mmax = LF.QL_range_mag()
zmin,zmax = LF.QL_range_z()

m = np.linspace(mmin,mmax,100)

plt.semilogy(m,LF.QL_dNdzdmddeg2(2.0,m),label='z=2.0')
plt.semilogy(m,LF.QL_dNdzdmddeg2(2.5,m),label='z=2.5')
plt.semilogy(m,LF.QL_dNdzdmddeg2(3.0,m),label='z=3.0')
plt.semilogy(m,LF.QL_dNdzdmddeg2(3.5,m),label='z=3.5')
plt.semilogy(m,LF.QL_dNdzdmddeg2(4.0,m),label='z=4.0')
plt.semilogy(m,LF.QL_dNdzdmddeg2(4.5,m),label='z=4.5')
plt.semilogy(m,LF.QL_dNdzdmddeg2(5.0,m),label='z=5.0')

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

plt.legend(loc=2,fontsize=15)
plt.ylim(1e-3,1e1)
plt.xlim(mmin,mmax)
plt.xlabel('r mag',fontsize=15)
plt.ylabel(r'dN / dz / dmag / $\rm{ddeg}^2$',fontsize=15)
plt.savefig('QuasarLF.png')
plt.show()

