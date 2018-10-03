import numpy as np
import matplotlib.pyplot as plt
import forecast as fc
import decimal
import argparse

def z_str(z):
    return "{:.3g}".format(decimal.Decimal(z))

def mu_str(mu):
    return "{:.2g}".format(decimal.Decimal(mu))

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="""Plot P3D signal-to-noise""")
parser.add_argument('--snr-files', type=str, nargs= "*", required=True, help="list of input snr filenames, like data/sn-spec-lya-20180907-r22-t4000-nexp4.dat")
parser.add_argument('--dndzdmag-file', type=str, required=True, help="qso density filename, like data/nzr_qso.dat")
parser.add_argument('--z-bin-centers', type=str, required=False, default="1.96,2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55", help="comma separated list of redshifts")
parser.add_argument('--total-density', type=float, default=None, required=False, help="normalize the qso dndwdmag to this total density (/deg2)")

args = parser.parse_args()

forecast = fc.FisherForecast(snr_filenames=args.snr_files,
        dndzdmag_filename=args.dndzdmag_file,total_density=args.total_density)

# define binning for plots
kmin_hMpc=0.01
kmax_hMpc=1.0
ks_hMpc=np.linspace(kmin_hMpc,kmax_hMpc,100)
Nk=len(ks_hMpc)
dk_hMpc=ks_hMpc[1]-ks_hMpc[0]
print('Nk =',Nk)
print('dk =',dk_hMpc)

# make plot for a fixed mu, as a function of z
forecast.verbose=0
lmin=3501.0
lmax=3701.0
plt.figure(figsize=(12,6))
while lmin < 5000:
    print(lmin,'<l<',lmax)    
    forecast.lmin=lmin
    forecast.lmax=lmax
    z = forecast.mean_z()
    mu = 0.9
    dmu = 0.2
    P3D = np.array([forecast.FluxP3D_hMpc(k,mu) for k in ks_hMpc])
    #P3D = forecast.LyaP3D.FluxP3D_hMpc(z,k_hMpc,mu) 
    VarP3D = np.array([forecast.VarFluxP3D_hMpc(k,mu,dk_hMpc,dmu) for k in ks_hMpc])    
    #VarP3D = forecast.VarFluxP3D_hMpc_kmu(k_hMpc,mu,dk_hMpc,dmu)
    #print('P3D =',P3D)
    #print('VarP3D =',VarP3D)    
    plt.plot(ks_hMpc,P3D/np.sqrt(VarP3D),label='z ='+z_str(z))
    #plt.plot(ks_hMpc,np.sqrt(VarP3D),':')
    lmin = lmax 
    lmax += 200
plt.legend(loc='best')
plt.title('mu ='+mu_str(mu)+', dk ='+str(dk_hMpc)+', dmu = '+mu_str(dmu))
plt.xlabel('k [h/Mpc]')
plt.ylabel('S/N')
plt.savefig('LyaP3D_mu'+mu_str(mu)+'.pdf')

# make plot for a fixed z, as a function of mu
mus=np.linspace(0.1,0.9,5)
dmu=mus[1]-mus[0]
print('mus',mus)
forecast.lmin=3700.0
forecast.lmax=3900.0
z = forecast.mean_z()
plt.figure(figsize=(12,6))
for mu in mus:
    P3D = np.array([forecast.FluxP3D_hMpc(k,mu) for k in ks_hMpc])
    #P3D = forecast.LyaP3D.FluxP3D_hMpc(z,k_hMpc,mu) 
    VarP3D = np.array([forecast.VarFluxP3D_hMpc(k,mu,dk_hMpc,dmu) for k in ks_hMpc])    
    #VarP3D = forecast.VarFluxP3D_hMpc_kmu(k_hMpc,mu,dk_hMpc,dmu)
    #print('P3D =',P3D)
    #print('VarP3D =',VarP3D)    
    plt.plot(ks_hMpc,P3D/np.sqrt(VarP3D),label='mu ='+mu_str(mu))
plt.legend(loc='best')
plt.title('z ='+z_str(z)+', dk ='+str(dk_hMpc)+', dmu = '+mu_str(dmu))
plt.xlabel('k [h/Mpc]')
plt.ylabel('S/N')
plt.savefig('LyaP3D_z'+z_str(z)+'.pdf')
