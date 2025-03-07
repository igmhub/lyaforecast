"""Control module for lyaforecast"""
import configparser
import numpy as np
import time 
from pathlib import Path

from lyaforecast.utils import get_file
from lyaforecast.cosmoCAMB import CosmoCamb
from lyaforecast.spectrograph import Spectrograph
from lyaforecast.survey import Survey
from lyaforecast.covariance import Covariance
from lyaforecast.power_spectrum import PowerSpectrum
from lyaforecast.plots import Plots


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

        self.out_file = Path(self.config['output']['filename'])
        self.out_folder = self.out_file.parent

        print('Initialise forecast')
        #initialise cosmology
        self.cosmo = CosmoCamb(self.config['cosmo'].get('filename'),
                                self.config['cosmo'].getfloat('z_ref', None))

        #load survey instance
        self.survey = Survey(self.config)

        #load spectrograph instance
        self.spectrograph = Spectrograph(self.config, self.survey)

        #load power spectrum instance
        self.power_spec = PowerSpectrum(self.config, self.cosmo, self.spectrograph)

        #initialise covariance class (McDonald & Eisenstein (2007)),
        #  that stores info and methods to compute p3d and its variance
        self.covariance = Covariance(self.config, self.cosmo, self.survey, 
                                     self.spectrograph, self.power_spec)
        
        init_end_time = time.time()
        print(f"Forecast initialized in {init_end_time - init_start_time:.4f} seconds.")

    def compute_weights(self):
        print('Computing weights')
        weights = np.zeros((self.survey.num_z_bins,self.survey.num_mag_bins))
        use_z_bin_list = False
        if self.config['survey'].get('z bin centres',None) is not None:
            z_bin_centres = self.config['survey'].get('z bin centres')
            z_bin_centres = np.array(z_bin_centres.split(",")).astype(float)
            dz = np.zeros(z_bin_centres.size)
            dz[1:-1] = (z_bin_centres[2:]-z_bin_centres[:-2])/2.
            dz[0]   = z_bin_centres[1]-z_bin_centres[0]
            dz[-1]  = z_bin_centres[-1]-z_bin_centres[-2]
            len_zz = len(z_bin_centres)
            use_z_bin_list = True
        else:
            zz = np.linspace(self.survey.zmin, self.survey.zmax,
                          self.survey.num_z_bins+1)
            z_bin_centres = np.zeros(self.survey.num_z_bins)
            len_zz = len(zz) - 1

        for iz in range(len_zz):
            #limits of individual redshift bins
            if use_z_bin_list:
                z1=z_bin_centres[iz]-dz[iz]/2
                z2=z_bin_centres[iz]+dz[iz]/2
                z_bin_centre = z_bin_centres[iz]
            else:
                z1=zz[iz]
                z2=zz[iz+1]
                z_bin_centre = z1 + (z2-z1)/2
                z_bin_centres[iz] = z_bin_centre

            print(f"z bin = [{z1}-{z2}], bin centre = {z_bin_centre}")
            
            lmin = self.cosmo.LYA_REST*(1+z1)
            lmax = self.cosmo.LYA_REST*(1+z2)

            self.covariance(lmin,lmax)
            self.covariance.compute_eff_density_and_noise()

            weights[iz] = self.covariance.w

        return weights, z_bin_centres

    def compute_neff(self):
        """Compute effective number density of pixels, 
        defined in McDonald & Eisenstein (2007),
          as function of magnitude and mean redshift"""
        neff = np.zeros((self.survey.num_z_bins,self.survey.num_mag_bins))
        weights = self.compute_weights()

        print('Computing Neff(m,z)')
        for i,wz in enumerate(weights):
            neff[i] = self.covariance.get_np_eff(wz)

        return neff
    

    def compute_survey_volume(self):
        """Compute survey volume (mpc/h) as a function 
          of redshift."""
        
        print('Computing survey volume...')
        survey_vol = np.zeros(self.survey.num_z_bins)
        use_z_bin_list = False
        if self.config['survey'].get('z bin centres',None) is not None:
            z_bin_centres = self.config['survey'].get('z bin centres')
            z_bin_centres = np.array(z_bin_centres.split(",")).astype(float)
            dz = np.zeros(z_bin_centres.size)
            dz[1:-1] = (z_bin_centres[2:]-z_bin_centres[:-2])/2.
            dz[0]   = z_bin_centres[1]-z_bin_centres[0]
            dz[-1]  = z_bin_centres[-1]-z_bin_centres[-2]
            len_zz = len(z_bin_centres)
            use_z_bin_list = True
        else:
            zz = np.linspace(self.survey.zmin, self.survey.zmax,
                          self.survey.num_z_bins+1)
            z_bin_centres = np.zeros(self.survey.num_z_bins)
            len_zz = len(zz) - 1

        for iz in range(len_zz):
            #limits of individual redshift bins
            if use_z_bin_list:
                z1=z_bin_centres[iz]-dz[iz]/2
                z2=z_bin_centres[iz]+dz[iz]/2
                z_bin_centre = z_bin_centres[iz]
            else:
                z1=zz[iz]
                z2=zz[iz+1]
                z_bin_centre = z1 + (z2-z1)/2
                z_bin_centres[iz] = z_bin_centre

            print(f"z bin = [{z1}-{z2}], bin centre = {z_bin_centre}")
            
            lmin = self.cosmo.LYA_REST*(1+z1)
            lmax = self.cosmo.LYA_REST*(1+z2)

            self.covariance(lmin,lmax)

            survey_vol[iz] = self.covariance.get_survey_volume()

        return survey_vol,z_bin_centres
    
    def compute_pk(self):
        """Compute snr per mode for k=0.14h/Mpc, mu=0.6"""

        #temporary
        assert not self.covariance.per_mag, "Cannot plot Pk with magnitude"
        
        p3d_z_k_mu = np.zeros((self.survey.num_z_bins,
                             self.covariance._num_k_bins,
                             self.covariance._num_mu_bins-1))
        p3d_var_z_k_mu = np.zeros_like(p3d_z_k_mu)

        n_pk_z_lya = np.zeros(self.survey.num_z_bins)
        n_pk_z_qso = np.zeros(self.survey.num_z_bins)
        use_z_bin_list = False

        if self.config['survey'].get('z bin centres',None) is not None:
            z_bin_centres = self.config['survey'].get('z bin centres')
            z_bin_centres = np.array(z_bin_centres.split(",")).astype(float)
            dz = np.zeros(z_bin_centres.size)
            dz[1:-1] = (z_bin_centres[2:]-z_bin_centres[:-2])/2.
            dz[0] = z_bin_centres[1]-z_bin_centres[0]
            dz[-1] = z_bin_centres[-1]-z_bin_centres[-2]
            len_zz = len(z_bin_centres)
            use_z_bin_list = True
        else:
            zz = np.linspace(self.survey.zmin, self.survey.zmax,
                          self.survey.num_z_bins+1)
            z_bin_centres = np.zeros(self.survey.num_z_bins)
            len_zz = len(zz) - 1

        for iz in range(len_zz):
            #limits of individual redshift bins
            if use_z_bin_list:
                z1=z_bin_centres[iz]-dz[iz]/2
                z2=z_bin_centres[iz]+dz[iz]/2
                z_bin_centre = z_bin_centres[iz]
            else:
                z1=zz[iz]
                z2=zz[iz+1]
                z_bin_centre = z1 + (z2-z1)/2
                z_bin_centres[iz] = z_bin_centre

            print(f"z bin = [{z1}-{z2}], bin centre = {z_bin_centre}")
            
            lmin = self.cosmo.LYA_REST*(1+z1)
            lmax = self.cosmo.LYA_REST*(1+z2)

            self.covariance(lmin,lmax)

            # some survey settings
            pix_width = self.covariance._pix_kms
            resolution = self.covariance._res_kms

            #weighting
            self.covariance.compute_eff_density_and_noise()

            n_pk_z_lya[iz],n_pk_z_qso[iz] = self.covariance.compute_n_pk(0.14,0.6)

            for i, mu in enumerate(self.power_spec.mu):

                p3d_mu = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3

                p3d_var_mu = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6


                p3d_z_k_mu[iz,:,i] = p3d_mu
                p3d_var_z_k_mu[iz,:,i] = p3d_var_mu


                # # #not making a great approximation here.
                # if i==0:
                #     p3d_z[str(z_bin_centre)] = p3d * self.covariance.dmu
                #     p3d_var_z[str(z_bin_centre)] = (1/self.covariance.mu.size)**2 * p3d_variance
                # else:
                #     p3d_z[str(z_bin_centre)] += p3d * self.covariance.dmu
                #     p3d_var_z[str(z_bin_centre)] += (1/self.covariance.mu.size)**2  * p3d_variance  

        return z_bin_centres,p3d_z_k_mu,p3d_var_z_k_mu,n_pk_z_lya,n_pk_z_qso

    def compute_bao(self):
        pass

    def run_bao_forecast(self,forecast=None):
        print('Running BAO forecast')

        areas = np.array(self.survey.area_deg2)
        resolutions = np.array(self.survey.res_kms)
        qso_densities = np.array(self.survey.qso_density)

        if self.covariance.per_mag:
            sigma_log_dh = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_log_da = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            corr_coef = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
        else:
            sigma_log_da = np.zeros(self.survey.num_z_bins)
            sigma_log_dh = np.zeros(self.survey.num_z_bins)
            corr_coef = np.zeros(self.survey.num_z_bins)

        use_z_bin_list = False
        if self.config['survey'].get('z bin centres',None) is not None:
            z_bin_centres = self.config['survey'].get('z bin centres')
            z_bin_centres = np.array(z_bin_centres.split(",")).astype(float)
            dz = np.zeros(z_bin_centres.size)
            dz[1:-1] = (z_bin_centres[2:]-z_bin_centres[:-2])/2.
            dz[0]   = z_bin_centres[1]-z_bin_centres[0]
            dz[-1]  = z_bin_centres[-1]-z_bin_centres[-2]
            len_zz = len(z_bin_centres)
            use_z_bin_list = True
            #z1=zz[iz]-dz[iz]/2
            #z2=zz[iz]+dz[iz]/2
        else:
            zz = np.linspace(self.survey.zmin, self.survey.zmax,
                          self.survey.num_z_bins+1)
            z_bin_centres = np.zeros(self.survey.num_z_bins)
            len_zz = len(zz) - 1

        #either this or using high dimensional matrices
        #for area in areas:
        #    for resolution in resolutions:
        #        for density in densities:
        #    

        p3d_z = {}
        p3d_var_z = {}

        for iz in range(len_zz):
            #limits of individual redshift bins
            if use_z_bin_list:
                z1=z_bin_centres[iz]-dz[iz]/2
                z2=z_bin_centres[iz]+dz[iz]/2
                z_bin_centre = z_bin_centres[iz]
            else:
                z1=zz[iz]
                z2=zz[iz+1]
                z_bin_centre = z1 + (z2-z1)/2
                z_bin_centres[iz] = z_bin_centre
            print(f"z bin = [{z1}-{z2}], bin centre = {z_bin_centre}")
            #store info in plots
            #self.plots.z_bin_centres[str()]

            # observed wavelength range from redshift limits
            lmin = self.cosmo.LYA_REST*(1+z1)
            lmax = self.cosmo.LYA_REST*(1+z2)

            #call function, setting bin width
            self.covariance(lmin,lmax)

            # some survey settings
            pix_width = self.covariance._pix_kms
            resolution = self.covariance._res_kms

            # this uses Luminosity, density, noise model
            self.covariance.compute_eff_density_and_noise()
                    
            # There is no need to add a marginalization on additive polynomial coefficients
            # because I subtract a high degree polynomial on P(k) to keep only the BAO wiggles
            # (such that the derivatives of the model wrt the alphas are by construction orthogonal to
            # the polynomial coefficients so no marginalization is needed)
            
            #vectorise power_spec computations.
            #empty array for fisher information
            #if per magnitude add extra dimension
            #will expand for SNR too.
            if self.covariance.per_mag:
                fisher_matrix = np.zeros((2,2,self.survey.num_mag_bins))
            else:
                fisher_matrix = np.zeros((2,2))            
            
            for i, mu in enumerate(self.power_spec.mu):

                p3d = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3

                p3d_variance = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6

                # #not making a great approximation here.
                if i==0:
                    p3d_z[str(z_bin_centre)] = p3d * self.power_spec.dmu
                    p3d_var_z[str(z_bin_centre)] = (1/self.power_spec.mu.size)**2 * p3d_variance
                else:
                    p3d_z[str(z_bin_centre)] += p3d * self.power_spec.dmu
                    p3d_var_z[str(z_bin_centre)] += (1/self.power_spec.mu.size)**2  * p3d_variance             

                # compute a smooth version of p3d
                # not sure how to do much better than a polynomial fit
                x = self.power_spec.logk
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
                kp = mu * self.power_spec.k
                kt = np.sqrt(1-mu**2) * self.power_spec.k
                
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
                dmodel = np.zeros(self.power_spec.k.size)
                dmodel[1:] = model[1:]-model[:-1]
                dmodel_dlk  = dmodel/self.power_spec.dlogk
                
                # k = sqrt( kp**2 + kt**2)
                # k'  = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
                # k' = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
                # dk/dap         = mu2 * k
                # dlog(k)/dap    = mu2
                # dlog(k)/dat    = (1-mu2)
                # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
                # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
                h = [mu**2,1 - mu**2]
                if self.covariance.per_mag:
                    fisher_matrix += np.outer(h,h)[:,:,None] * np.sum(dmodel_dlk**2 / p3d_variance.T, axis=1).T
                else:
                    fisher_matrix += np.outer(h,h) * np.sum(dmodel_dlk**2 / p3d_variance)
            
            if self.covariance.per_mag:
                cov = np.linalg.inv(fisher_matrix.T)
                cov_diag = np.diagonal(cov.T,axis1=0,axis2=1)
                sigma_log_dh[iz] = np.sqrt(cov_diag.T[0])
                sigma_log_da[iz] = np.sqrt(cov_diag.T[1])
                corr_coef[iz] = cov.T[0,1]/np.sqrt(cov_diag.T[0]*cov_diag.T[1])

                print("ap={},at={},corr={}".format(sigma_log_da[iz][-1],
                                        sigma_log_dh[iz][-1],corr_coef[iz][-1]))
            else:
                cov = np.linalg.inv(fisher_matrix)
                sigma_log_dh[iz] = np.sqrt(cov[0,0])
                sigma_log_da[iz] = np.sqrt(cov[1,1])    
                corr_coef[iz] = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

                print("ap={},at={},corr={}".format(sigma_log_da[iz],
                                        sigma_log_dh[iz],corr_coef[iz]))
        
        
        if self.covariance.per_mag:
            sigma_log_da_combined_m = 1./np.sqrt(np.sum(1./sigma_log_da**2,axis=0))
            sigma_log_dh_combined_m = 1./np.sqrt(np.sum(1./sigma_log_dh**2,axis=0))
            sigma_log_da_combined = sigma_log_da_combined_m[-1]
            sigma_log_dh_combined = sigma_log_dh_combined_m[-1]
        else:
            sigma_log_da_combined_m = None
            sigma_log_dh_combined_m = None
            sigma_log_da_combined = 1./np.sqrt(np.sum(1./sigma_log_da**2))
            sigma_log_dh_combined = 1./np.sqrt(np.sum(1./sigma_log_dh**2))

        #these aren't log-spaced right?

        print(f'\n Combined: sigma_log_da={sigma_log_da_combined}'
                    f', sigma_log_dh={sigma_log_dh_combined}')
       
        data = {}
        data["redshifts"] = z_bin_centres
        data["mean redshift"] = self.cosmo.z_ref
        data["magnitudes"] = {self.survey.band:self.survey.maglist}
        data["at_err"] = sigma_log_da_combined
        data["ap_err"] = sigma_log_dh_combined
        data['ap_err_m'] = sigma_log_dh_combined_m
        data['at_err_m'] = sigma_log_da_combined_m
        data['ap_err_z'] = sigma_log_dh
        data['at_err_z'] = sigma_log_da
        data['p3d_z'] = p3d_z
        data['p3d_var_z'] = p3d_var_z

        if forecast is not None:
            #load data to plots instance
            self.plots = Plots(forecast,data=data)

            if self.config['control'].getboolean('plot bao'):
                if self.covariance.per_mag:
                    self.plots.plot_da_h_m()
                    self.plots.fig.savefig(self.out_folder.joinpath('dap_dat_dm.png'))
                else:
                    self.plots.plot_da_h_z()
                    self.plots.fig.savefig(self.out_folder.joinpath('dap_dat_z.png'))
        #     if self.config['control'].getboolean('plot p3d'):
        #         if not self.covariance.per_mag:
        #             self.plots.plot_p3d_z()
        #             self.plots.fig.savefig(self.out_folder.joinpath('pk_z.png'))
        #         else:
        #             print('plot p3d requires per_mag = False')
        #     if self.config['control'].getboolean('plot p3d var'):
        #         if not self.covariance.per_mag:
        #             self.plots.plot_var_p3d_z()
        #             self.plots.fig.savefig(self.out_folder.joinpath('var_pk_z.png'))
        #         else:
        #             self.plots.plot_var_p3d_m()
        #             self.plots.fig.savefig(self.out_folder.joinpath('var_pk_m.png'))

        
    def get_cosmo_params(self):
        pass



                
            


