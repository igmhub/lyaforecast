#!/usr/bin/env python


import numpy as np
import forecast as fc
import argparse

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   description="""Fisher forecast""")
parser.add_argument('--snr-files', type=str, nargs= "*", required=True, help="list of input snr filenames, like sn-spec-lya-20180907-r22-t4000-nexp4.dat")
parser.add_argument('--dndzdmag-file', type=str, required=True, help="qso density filename, like nzr_qso.dat")
parser.add_argument('--mag', type=str, required=False, default=None,help="min,max magnitude range")

args = parser.parse_args()

forecast = fc.FisherForecast(snr_filenames=args.snr_files,dndzdmag_filename=args.dndzdmag_file)
# set redshift range
forecast.lmin=3600.
forecast.lmax=5400.

print("args.mag=",args.mag)
if args.mag is not None :
    tmp=args.mag.split(",")
    forecast.mag_min=float(tmp[0])
    forecast.mag_max=float(tmp[1])
    print("QSO magnitude range=",forecast.mag_min,forecast.mag_max)
    
# this uses Luminosity, density, noise model
np_eff,Pw2D,PN_eff=forecast.EffectiveDensityAndNoise()


mu_bin_edges = np.linspace(0,1.,60)
k      = np.linspace(0.01,1,100) # h/Mpc
dk     = k[1]-k[0] # h/Mpc
lk     = np.log(k)
dlk    = dk/k

fisher_matrix = np.zeros((2,2))

sdmu=0
for mu_index in range(mu_bin_edges.size-1) :
    
    mu  = (mu_bin_edges[mu_index+1]+mu_bin_edges[mu_index])/2.
    dmu = mu_bin_edges[mu_index+1]-mu_bin_edges[mu_index]
    
    p3d    = np.array([forecast.FluxP3D_hMpc(kk,mu) for kk in k]) # (Mpc/h)**3
    varp3d = np.array([forecast.VarFluxP3D_hMpc(kk,mu,dk,dmu=dmu,Pw2D=Pw2D,PN_eff=PN_eff) for kk in k]) # (Mpc/h)**6
    # can certainly do much much better to get the smooth_p3d than this
    coef=np.polyfit(lk,np.log(p3d),8)
    smooth_p3d = np.exp(np.poly1d(coef)(lk))
    
    # I am looking only at the wiggles
    model      = p3d-smooth_p3d
    dmodel     = np.zeros(k.size)
    dmodel[1:] = model[1:]-model[:-1]
    dmodeldlk  = dmodel/dlk
    
    # dmodel_d_log_alpha_parallel = ?
    # k = sqrt( kpar**2 + kperp**2) = sqrt( apar**2*k**2*mu2 + aper**2*k**2*(1-mu2))= k*sqrt( apar**2*mu2 + aper**2*(1-mu2))
    # dk/dapar       = mu2 * k
    # dlog(k)/dapar  = mu2
    # dlog(k)/daperp = (1-mu2)
    # dmodel/dapar   = dmodel/dlog(k)*dlog(k)/dapar  = dmodeldlk * mu2
    # dmodel/daperp  = dmodel/dlog(k)*dlog(k)/daperp = dmodeldlk * (1-mu2)
    
    h=np.array([mu**2,(1-mu**2)])
    hh=np.outer(h,h)
    fisher_matrix += hh*np.sum(dmodeldlk**2/varp3d)
    

print(fisher_matrix)
cov = np.linalg.inv(fisher_matrix)
print("sigma apar=",np.sqrt(cov[0,0]))
print("sigma aperp=",np.sqrt(cov[1,1]))
