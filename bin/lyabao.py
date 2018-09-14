#!/usr/bin/env python


import numpy as np
import forecast as fc
import argparse

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   description="""Fisher forecast""")
parser.add_argument('--snr-files', type=str, nargs= "*", required=True, help="list of input snr filenames, like sn-spec-lya-20180907-r22-t4000-nexp4.dat")
parser.add_argument('--dndzdmag-file', type=str, required=True, help="qso density filename, like nzr_qso.dat")
parser.add_argument('--z-bin-centers', type=str, required=False, default="1.96,2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55", help="comma separated list of redshifts")

args = parser.parse_args()

forecast = fc.FisherForecast(snr_filenames=args.snr_files,dndzdmag_filename=args.dndzdmag_file)


zz = np.array(args.z_bin_centers.split(",")).astype(float)
dz = np.zeros(zz.size)
dz[1:-1] = (zz[2:]-zz[:-2])/2.
dz[0]   = zz[1]-zz[0]
dz[-1]  = zz[-1]-zz[-2]

sigma_log_da = np.zeros(zz.size)
sigma_log_dh = np.zeros(zz.size)
corr_coef    = np.zeros(zz.size)

for iz in range(zz.size) :
    z1=zz[iz]-dz[iz]/2
    z2=zz[iz]+dz[iz]/2
    print("{} {} {} {}".format(iz,zz[iz],z1,z2))


    # set redshift range
    forecast.lmin=forecast.cosmo.lya_A*(1+z1)
    forecast.lmax=forecast.cosmo.lya_A*(1+z2)
    
    # this uses Luminosity, density, noise model
    np_eff,Pw2D,PN_eff=forecast.EffectiveDensityAndNoise()


    mu_bin_edges = np.linspace(0,1.,10)
            
    # need linear
    k      = np.linspace(0.01,1,100) # h/Mpc
    dk     = (k[1]-k[0]) # h/Mpc
    dlk    = dk/k
    lk     = np.log(k)
    
    # There is no need to add a marginalization on additive polynomial coefficients
    # because I subtract a high degree polynomial on P(k) to keep only the BAO wiggles
    # (such that the derivatives of the model wrt the alphas are by construction orthogonal to
    # the polynomial coefficients so no marginalization is needed)
    
    fisher_matrix = np.zeros((2,2))
    
    
    for mu_index in range(mu_bin_edges.size-1) :

        mu  = (mu_bin_edges[mu_index+1]+mu_bin_edges[mu_index])/2.
        dmu = mu_bin_edges[mu_index+1]-mu_bin_edges[mu_index]

        p3d    = np.array([forecast.FluxP3D_hMpc(kk,mu) for kk in k]) # (Mpc/h)**3
        varp3d = np.array([forecast.VarFluxP3D_hMpc(kk,mu,dk,dmu=dmu,Pw2D=Pw2D,PN_eff=PN_eff) for kk in k]) # (Mpc/h)**6

        # compute a smooth version of p3d
        # not sure how to do much better than a polynomial fit
        x=np.log(k)
        y=np.log(p3d)
        x -= np.mean(x)
        x /= (np.max(x)-np.min(x))
        w=np.ones(x.size)
        w[:3] *= 1.e8 
        coef=np.polyfit(x,y,8,w=w)
        pol=np.poly1d(coef)
        smooth_p3d = np.exp(pol(x))

        # I am looking only at the wiggles
        model      = p3d-smooth_p3d
        
        # gaussian damping
        kp = mu*k
        kt = np.sqrt(1-mu*mu)*k
        SigNLp = 3.26 # Mpc/h
        SigNLt = 3.26 # Mpc/h
        model     *= np.exp(-(SigNLp*kp)**2/2-(SigNLt*kt)**2/2)

        # derivative of model wrt to log(k)
        dmodel     = np.zeros(k.size)
        dmodel[1:] = model[1:]-model[:-1]
        dmodeldlk  = dmodel/dlk
        
        # k = sqrt( kp**2 + kt**2)
        #   = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
        #   = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
        # dk/dap         = mu2 * k
        # dlog(k)/dap    = mu2
        # dlog(k)/dat    = (1-mu2)
        # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
        # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
        
        h=[mu**2,(1-mu**2)]
        fisher_matrix += np.outer(h,h)*np.sum(dmodeldlk**2/varp3d)
    
    
    cov = np.linalg.inv(fisher_matrix)
    sigma_log_dh[iz] = np.sqrt(cov[0,0])
    sigma_log_da[iz] = np.sqrt(cov[1,1])    
    corr_coef[iz]    = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
print("# z sigma_log_da sigma_log_dh correlation")
for i in range(zz.size) :
    print("{},{},{},{}".format(zz[i],sigma_log_da[i],sigma_log_dh[i],corr_coef[i]))

print("\n Combined: sigma_log_da={} sigma_log_dh={}".format(1./np.sqrt(np.sum(1./sigma_log_da**2)),1./np.sqrt(np.sum(1./sigma_log_dh**2))))
