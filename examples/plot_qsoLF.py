import numpy as np
import matplotlib.pyplot as plt
import qso_LF as qLF 

# get QLF object
LF = qLF.QuasarLF()

# get limits
mmin,mmax = LF.range_mag()
zmin,zmax = LF.range_z()

m = np.linspace(mmin,mmax,100)

# plot QL at different z
plt.semilogy(m,LF.dNdzdmddeg2(2.0,m)[0],label='z=2.0')
plt.semilogy(m,LF.dNdzdmddeg2(2.5,m)[0],label='z=2.5')
plt.semilogy(m,LF.dNdzdmddeg2(3.0,m)[0],label='z=3.0')
plt.semilogy(m,LF.dNdzdmddeg2(3.5,m)[0],label='z=3.5')
plt.semilogy(m,LF.dNdzdmddeg2(4.0,m)[0],label='z=4.0')
plt.semilogy(m,LF.dNdzdmddeg2(4.5,m)[0],label='z=4.5')
plt.semilogy(m,LF.dNdzdmddeg2(5.0,m)[0],label='z=5.0')

plt.legend(loc=2,fontsize=15)
plt.ylim(1e-2,1e2)
plt.xlim(mmin,mmax)
plt.xlabel('g mag',fontsize=15)
plt.ylabel(r'dN / dz / dmag / $\rm{ddeg}^2$',fontsize=15)
plt.show()

