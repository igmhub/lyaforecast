import numpy as np
import matplotlib.pyplot as plt
import qso_LF as qLF 

# get QLF object
LF = qLF.QuasarLF()

# get limits
mmin,mmax = LF.QL_range_mag()
zmin,zmax = LF.QL_range_z()

m = np.linspace(mmin,mmax,100)

# plot QL at different z
plt.semilogy(m,LF.QL_dNdzdmddeg2(2.0,m),label='z=2.0')
plt.semilogy(m,LF.QL_dNdzdmddeg2(2.5,m),label='z=2.5')
plt.semilogy(m,LF.QL_dNdzdmddeg2(3.0,m),label='z=3.0')
plt.semilogy(m,LF.QL_dNdzdmddeg2(3.5,m),label='z=3.5')
plt.semilogy(m,LF.QL_dNdzdmddeg2(4.0,m),label='z=4.0')
plt.semilogy(m,LF.QL_dNdzdmddeg2(4.5,m),label='z=4.5')
plt.semilogy(m,LF.QL_dNdzdmddeg2(5.0,m),label='z=5.0')

plt.legend(loc=2,fontsize=15)
plt.ylim(1e-3,1e1)
plt.xlim(mmin,mmax)
plt.xlabel('g mag',fontsize=15)
plt.ylabel(r'dN / dz / dmag / $\rm{ddeg}^2$',fontsize=15)
plt.show()

