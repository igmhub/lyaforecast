import numpy as np
import matplotlib.pyplot as plt
import argparse
import qso_LF as qLF 

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="""Plot QLF""")
parser.add_argument('--dndzdmag-file', type=str, required=True, help="qso density filename, like data/nzr_qso.dat")
args = parser.parse_args()

# get QLF object
LF = qLF.QuasarLF(args.dndzdmag_file)

# get limits
m = np.linspace(17,23,100)

# plot QL at different z
plt.semilogy(m,LF.dNdzdmddeg2(2.0,m)[0],label='z=2.0')
plt.semilogy(m,LF.dNdzdmddeg2(2.5,m)[0],label='z=2.5')
plt.semilogy(m,LF.dNdzdmddeg2(3.0,m)[0],label='z=3.0')
plt.semilogy(m,LF.dNdzdmddeg2(3.5,m)[0],label='z=3.5')
plt.semilogy(m,LF.dNdzdmddeg2(4.0,m)[0],label='z=4.0')

plt.legend(loc=2,fontsize=15)
plt.ylim(1e-2,1e2)
#plt.xlim(mmin,mmax)
plt.xlabel('r mag',fontsize=15)
plt.ylabel(r'dN / dz / dmag / $\rm{ddeg}^2$',fontsize=15)
plt.show()

