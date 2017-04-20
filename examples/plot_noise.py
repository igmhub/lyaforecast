import numpy as np
import matplotlib.pyplot as plt
import spectrograph as sp 

def plot_quasar(spec,z,mag,lobs):
  label='z='+str(z)+', mag='+str(mag)
  Nl = len(l)
  noise = np.empty(Nl)
  for i in range(Nl):
    noise[i] = spec.PixelNoiseRMS(mag,z,l[i],1.0) 
  plt.plot(l,noise,label=label)

band='r'
spec = sp.Spectrograph(band)
lmin,lmax = spec.range_lobs_A()
l = np.linspace(lmin,lmax,1000)
plot_quasar(spec,z=2.5,mag=22.0,lobs=l)
plot_quasar(spec,z=2.5,mag=22.5,lobs=l)
plot_quasar(spec,z=3.0,mag=22.0,lobs=l)
plot_quasar(spec,z=3.0,mag=22.5,lobs=l)

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

plt.legend(fontsize=15)
plt.ylim(0,2.0)
plt.xlim(3600,6000)
plt.xlabel('wavelength [A]',fontsize=15)
plt.ylabel('pixel noise',fontsize=15)
if band is 'r':
    plt.title('Lya SNR as a function of quasar (z, r mag)')
elif band is 'g':
    plt.title('Lya SNR as a function of quasar (z, g mag)')
else:
    print('bad band in plot_noise',band)
    raise SystemExit
plt.show()

