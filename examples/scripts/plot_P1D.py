import numpy as np
import matplotlib.pyplot as plt
import analytic_p1d_PD2013 as aP1D

# s/km units
k=np.logspace(-4,-1,1000)
pk_z2=aP1D.P1D_z_kms_PD2013(2.0,k)
pk_z3=aP1D.P1D_z_kms_PD2013(3.0,k)
pk_z4=aP1D.P1D_z_kms_PD2013(4.0,k)

#plt.ylim(1e0,1e5)
plt.xlim(1e-4,0.1)
plt.xlabel(r'k [s $\rm{km}^{-1}$]',fontsize=15)
plt.ylabel(r'P(k) [km $\rm{s}^{-1}$]',fontsize=15)
plt.title(r'1D flux power',fontsize=15)
plt.loglog(k,pk_z4,label=r'z=4.0')
plt.loglog(k,pk_z3,label=r'z=3.0')
plt.loglog(k,pk_z2,label=r'z=2.0')
plt.legend(fontsize=15)
plt.show()

