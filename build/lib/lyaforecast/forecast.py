"""Control module for lyaforecast"""
import configparser
import numpy as np
import time 

from lyaforecast.utils import get_file
from lyaforecast.cosmoCAMB import CosmoCamb
from lyaforecast.spectrograph import Spectrograph
from lyaforecast.survey import Survey
from lyaforecast.covariance import Covariance
from lyaforecast.power_spectrum import PowerSpectrum




class Forecast:
    """Main LyaForecast class.
    ...outline...
    """
    # _survey_properties = None
    # _spectro_properties = None

    def __init__(self, cfg_path):
        """
        Parameters
        ----------
        cfg_path : string
            Path to main.ini config file
        """
        init_start_time = time.time()
        # Read config files
        self.config = configparser.ConfigParser()
        self.config.optionxform = lambda option: option
        self.config.read(get_file(cfg_path))

        print('Initialise forecast')
        #initialise cosmology
        self.cosmo = CosmoCamb(self.config['cosmo'].get('filename'),
                                self.config['cosmo'].getfloat('z_ref', None))

        #load survey instance
        self.survey = Survey(self.config)

        #load spectrograph instance
        self.spectrograph = Spectrograph(self.config, self.survey)

        #load power spectrum instance
        self.power_spec = PowerSpectrum(self.cosmo)

        #initialise covariance class (McDonald & Eisenstein (2007)),
        #  that stores info and methods to compute p3d and its variance
        self.covariance = Covariance(self.config, self.cosmo, self.survey, 
                                     self.spectrograph, self.power_spec)

        init_end_time = time.time()
        print(f"Forecast initialized in {init_end_time - init_start_time:.4f} seconds.")

    def run_bao_forecast(self):
        print('Running BAO forecast')

        z_bin_centers = "1.96,2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55"
        zz = np.array(z_bin_centers.split(",")).astype(float)
        #dz = np.diff(zz)
        dz = np.zeros(zz.size)
        dz[1:-1] = (zz[2:]-zz[:-2])/2.
        dz[0]   = zz[1]-zz[0]
        dz[-1]  = zz[-1]-zz[-2]
        
        #arrays to store bao info
        sigma_log_da = np.zeros(zz.size)
        sigma_log_dh = np.zeros(zz.size)
        corr_coef = np.zeros(zz.size)

        for iz in range(zz.size) :
            #limits of individual redshift bins
            z1=zz[iz]-dz[iz]/2
            z2=zz[iz]+dz[iz]/2
            print("z bin = [{}-{}]".format(z1,z2))

            # observed wavelength range from redshift limits, used to calculate mean 
            # redshifts and evolve biases.
            # sadly it means we can't pre-compute eff noise and density.
            self.covariance.lmin = self.cosmo.LYA_REST*(1+z1)
            self.covariance.lmax = self.cosmo.LYA_REST*(1+z2)

            # this uses Luminosity, density, noise model
            # Calum: now computed in the Covariance class initialisation.
            #np_eff,Pw2D,PN_eff = forecast.EffectiveDensityAndNoise()
            self.covariance.compute_eff_density_and_noise()
                    
            # need linear
            #Calum: this info now stored in Covariance instance.
            # k      = np.linspace(0.01,1,100) # h/Mpc
            # dk     = (k[1]-k[0]) # h/Mpc
            # dlk    = dk/k
            # lk     = np.log(k)
            # mu_bin_edges = np.linspace(0,1.,10)
            
            # There is no need to add a marginalization on additive polynomial coefficients
            # because I subtract a high degree polynomial on P(k) to keep only the BAO wiggles
            # (such that the derivatives of the model wrt the alphas are by construction orthogonal to
            # the polynomial coefficients so no marginalization is needed)
            
            #vectorise power_spec computations.
            
            #empty array for fisher information
            fisher_matrix = np.zeros((2,2))            
            
            for i, mu in enumerate(self.covariance.mu):

                p3d = np.array([self.covariance.compute_p3d_hmpc(k,mu) 
                                        for k in self.covariance.k]) # (Mpc/h)**3

                p3d_variance = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.covariance.k]) # (Mpc/h)**6

                # compute a smooth version of p3d
                # not sure how to do much better than a polynomial fit
                x = self.covariance.logk
                y = np.log(p3d)
                x -= np.mean(x)
                x /= (np.max(x)-np.min(x))
                w = np.ones(x.size)
                w[:3] *= 1.e8 
                coef = np.polyfit(x,y,8,w=w)
                poly = np.poly1d(coef)
                smooth_p3d = np.exp(poly(x))

                # get wiggles (peak) only part
                model = p3d - smooth_p3d
                
                # add gaussian damping
                kp = mu * self.covariance.k
                kt = np.sqrt(1-mu**2) * self.covariance.k
                
                # Eisenstein, Seo, White, 2007, Eq. 12
                sig_nl_perp = 3.26 # Mpc/h
                f = self.cosmo.growth_rate # lograthmic growth (at z_ref)
                sig_nl_par = (1 + f) * sig_nl_perp # Mpc/h
                
                model *= np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))

                if False and zz[iz]>2.1 :
                    import matplotlib.pyplot as plt
                    print("z=",zz[iz],"mu=",mu)
                    rebin=2
                    scale=1./np.sqrt(rebin)
                    plt.fill_between(k,model/p3d-1-np.sqrt(varp3d)/p3d*scale,model/p3d-1+np.sqrt(varp3d)/p3d*scale,alpha=0.4)
                    plt.plot(k,model/p3d-1,c="k")
                    kb=k[:(k.size//rebin)*rebin].reshape(k.size//rebin,rebin).mean(-1)
                    plt.plot(kb,np.interp(kb,k,model/p3d-1),"o",c="k")
                    plt.xlim([0.,0.4])
                    plt.show()
                    
                
                # derivative of model wrt to log(k)
                dmodel = np.zeros(self.covariance.k.size)
                dmodel[1:] = model[1:]-model[:-1]
                dmodel_dlk  = dmodel/self.covariance.dlogk
                
                # k = sqrt( kp**2 + kt**2)
                # k'  = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
                # k' = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
                # dk/dap         = mu2 * k
                # dlog(k)/dap    = mu2
                # dlog(k)/dat    = (1-mu2)
                # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
                # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
                
                h = [mu**2,(1 - mu**2)]
                fisher_matrix += np.outer(h,h) * np.sum(dmodel_dlk**2 / p3d_variance)

            cov = np.linalg.inv(fisher_matrix)
            sigma_log_dh[iz] = np.sqrt(cov[0,0])
            sigma_log_da[iz] = np.sqrt(cov[1,1])    
            corr_coef[iz] = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
            
        print("# z sigma_log_da sigma_log_dh correlation")
        for i in range(zz.size) :
            print("{},{},{},{}".format(zz[i],sigma_log_da[i],sigma_log_dh[i],corr_coef[i]))

        print("\n Combined: sigma_log_da={} sigma_log_dh={}".format(1./np.sqrt(np.sum(1./sigma_log_da**2)),
                                                                    1./np.sqrt(np.sum(1./sigma_log_dh**2))))


                
            


