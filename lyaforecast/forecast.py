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

        # Use only BAO part of Pl, true to run BAO forecast.
        self._bao_only = self.config['control'].getboolean('bao only')

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
        eff_vol = np.zeros((self.survey.num_z_bins,self.survey.num_mag_bins))
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

            weights[iz] = self.covariance._w_lya
            eff_vol[iz] = self.covariance._compute_qso_eff_vol(0.14,0.6)


        return z_bin_centres, weights, eff_vol

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
                             self.power_spec._num_k_bins,
                             self.power_spec._num_mu_bins-1))
        p3d_var_z_k_mu = np.zeros_like(p3d_z_k_mu)

        p3d_qso_z_k_mu = np.zeros_like(p3d_z_k_mu)
        p3d_qso_var_z_k_mu = np.zeros_like(p3d_z_k_mu)

        p3d_info = {}


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

                p3d = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                
                p3d_qso = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                            mu, pix_width, resolution, 'qso') 
                                        for k in self.power_spec.k]) 

                p3d_var = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                
                p3d_qso_var = np.array([self.covariance.compute_qso_power_variance(k,mu) 
                                    for k in self.power_spec.k])


                p3d_z_k_mu[iz,:,i] = p3d
                p3d_qso_z_k_mu[iz,:,i] = p3d_qso

                p3d_var_z_k_mu[iz,:,i] = p3d_var
                p3d_qso_var_z_k_mu[iz,:,i] = p3d_qso_var

            p3d_info['p_lya'] = p3d_z_k_mu
            p3d_info['p_qso'] = p3d_qso_z_k_mu
            p3d_info['var_lya'] = p3d_var_z_k_mu
            p3d_info['var_qso'] = p3d_qso_var_z_k_mu

        return z_bin_centres,p3d_info,n_pk_z_lya,n_pk_z_qso

    def compute_bao(self):
        return

    def run_bao_forecast(self,forecast=None):
        print('Running BAO forecast')

        if not self._bao_only:
            raise AssertionError('"control: bao only" must be True to run BAO forecast.')

        areas = np.array(self.survey.area_deg2)
        resolutions = np.array(self.survey.res_kms)
        qso_densities = np.array(self.survey.qso_density)

        if self.covariance.per_mag:
            sigma_dh = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            corr_coef = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
        else:
            sigma_da_lya = np.zeros(self.survey.num_z_bins)
            sigma_dh_lya = np.zeros(self.survey.num_z_bins)
            sigma_da_qso = np.zeros(self.survey.num_z_bins)
            sigma_dh_qso = np.zeros(self.survey.num_z_bins)
            sigma_da_cross = np.zeros(self.survey.num_z_bins)
            sigma_dh_cross = np.zeros(self.survey.num_z_bins)
            sigma_da_lya_lya_lya_qso = np.zeros(self.survey.num_z_bins)
            sigma_dh_lya_lya_lya_qso = np.zeros(self.survey.num_z_bins)
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
            len_zz = len(z_bin_centres)

        #either this or using high dimensional matrices
        #for area in areas:
        #    for resolution in resolutions:
        #        for density in densities:
        #    
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
                fisher_matrix_lya = np.zeros((2,2,self.survey.num_mag_bins))
                fisher_matrix_qso = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)
            else:
                fisher_matrix_lya = np.zeros((2,2))      
                fisher_matrix_qso = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)      

            
            for i, mu in enumerate(self.power_spec.mu):

                p3d_lya = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                p3d_qso = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                            mu, pix_width, resolution, 'qso') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                p3d_cross = np.array([self.power_spec.compute_p3d_hmpc_smooth(z_bin_centre, k,
                                            mu, pix_width, resolution, 'lyaqso') 
                                            for k in self.power_spec.k]) # (Mpc/h)**3

                # In reality var^2/P^2
                p3d_lya_var = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                p3d_qso_var = np.array([self.covariance.compute_qso_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                p3d_cross_var = np.array([self.covariance.compute_cross_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
           

                dp3d_lya_dlogk = self.get_dp_dlogk(p3d_lya,mu)
                dp3d_qso_dlogk = self.get_dp_dlogk(p3d_qso,mu)
                dp3d_cross_dlogk = self.get_dp_dlogk(p3d_cross,mu)
                
                # k = sqrt( kp**2 + kt**2)
                # k'  = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
                # k' = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
                # dk/dap         = mu2 * k
                # dlog(k)/dap    = mu2
                # dlog(k)/dat    = (1-mu2)
                # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
                # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
                fisher_matrix_lya += self.get_fisher(mu,dp3d_lya_dlogk,p3d_lya_var)
                fisher_matrix_qso += self.get_fisher(mu,dp3d_qso_dlogk,p3d_qso_var)
                fisher_matrix_cross += self.get_fisher(mu,dp3d_cross_dlogk,p3d_cross_var)
            

            sigma_dh_lya_z, sigma_da_lya_z, corr_coef_lya_z = self.print_bao(fisher_matrix_lya)
            sigma_dh_qso_z, sigma_da_qso_z, corr_coef_qso_z = self.print_bao(fisher_matrix_qso)
            sigma_dh_cross_z, sigma_da_cross_z, corr_coef_cross_z = self.print_bao(fisher_matrix_cross)

            sigma_dh_lya[iz] = sigma_dh_lya_z
            sigma_da_lya[iz] = sigma_da_lya_z
            corr_coef[iz] = corr_coef_lya_z

            sigma_dh_qso[iz] = sigma_dh_qso_z
            sigma_da_qso[iz] = sigma_da_qso_z

            sigma_dh_cross[iz] = sigma_dh_cross_z
            sigma_da_cross[iz] = sigma_da_cross_z
            

            sigma_da_lya_lya_lya_qso[iz] = 1 / (1 / sigma_da_lya_z + 1 / sigma_da_cross_z)
            sigma_dh_lya_lya_lya_qso[iz] = 1 / (1 / sigma_dh_lya_z + 1 / sigma_dh_cross_z)
        
        
        if self.covariance.per_mag:
            sigma_da_combined_m = 1./np.sqrt(np.sum(1./sigma_da_lya**2,axis=0))
            sigma_dh_combined_m = 1./np.sqrt(np.sum(1./sigma_dh_lya**2,axis=0))
            sigma_da_combined = sigma_da_combined_m[-1]
            sigma_dh_combined = sigma_dh_combined_m[-1]
        else:
            sigma_da_combined_m = None
            sigma_dh_combined_m = None
            sigma_da_combined = 1./np.sqrt(np.sum(1./sigma_da_lya**2))
            sigma_dh_combined = 1./np.sqrt(np.sum(1./sigma_dh_lya**2))
            sigma_da_combined_qso = 1./np.sqrt(np.sum(1./sigma_da_qso**2))
            sigma_dh_combined_qso = 1./np.sqrt(np.sum(1./sigma_dh_qso**2))
            sigma_da_combined_cross = 1./np.sqrt(np.sum(1./sigma_da_cross**2))
            sigma_dh_combined_cross = 1./np.sqrt(np.sum(1./sigma_dh_cross**2))
            sigma_da_combined_comb = 1./np.sqrt(np.sum(1./sigma_da_lya_lya_lya_qso**2))
            sigma_dh_combined_comb = 1./np.sqrt(np.sum(1./sigma_dh_lya_lya_lya_qso**2))

        #these aren't log-spaced right?

        print(f'\n Combined: sigma_da={sigma_da_combined}'
                    f', sigma_dh={sigma_dh_combined}')
        print(f'\n Combined: sigma_da_qso={sigma_da_combined_qso}'
                    f', sigma_dh_qso={sigma_dh_combined_qso}')
        print(f'\n Combined: sigma_da_cross={sigma_da_combined_cross}'
                    f', sigma_dh_cross={sigma_dh_combined_cross}')
        print(f'\n Combined: sigma_da_comb={sigma_da_combined_comb}'
                    f', sigma_dh_comb={sigma_dh_combined_comb}')
       
        data = {}
        data["redshifts"] = z_bin_centres
        data["mean redshift"] = self.cosmo.z_ref
        data["magnitudes"] = {self.survey.band:self.survey.maglist}
        data["at_err_lya"] = sigma_da_combined
        data["ap_err_lya"] = sigma_dh_combined
        data["at_err_cross"] = sigma_da_combined
        data["ap_err_cross"] = sigma_dh_combined
        #data["at_err"] = sigma_da_combined
        #data["ap_err"] = sigma_dh_combined
        data['ap_err_m'] = sigma_dh_combined_m
        data['at_err_m'] = sigma_da_combined_m
        data['ap_err_lya_z'] = sigma_dh_lya
        data['at_err_lya_z'] = sigma_da_lya
        data['ap_err_qso_z'] = sigma_dh_qso
        data['at_err_qso_z'] = sigma_da_qso
        data['ap_err_cross_z'] = sigma_dh_cross
        data['at_err_cross_z'] = sigma_da_cross
        data['ap_err_comb_z'] = sigma_dh_lya_lya_lya_qso
        data['at_err_comb_z'] = sigma_da_lya_lya_lya_qso

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
        
    def get_dp_dlogk(self,model,mu):
        """Return the differential of the a peak power spectrum component, 
            with respect to log k"""
        # I suspect it's not appropriate to do this across mu, or after the model is applied.
        # compute a smooth version of p3d
        # not sure how to do much better than a polynomial fit
        # x = self.power_spec.logk
        # y = np.log(p3d)
        # x -= np.mean(x)
        # x /= (np.max(x)-np.min(x))
        # w = np.ones(x.size)
        # w[:3] *= 1.e8 
        # coef = np.polyfit(x,y,8,w=w)
        # poly = np.poly1d(coef)
        # smooth_p3d = np.exp(poly(x))

        # # get wiggles (peak) only part
        # model = p3d - smooth_p3d
        # add gaussian damping
        kp = mu * self.power_spec.k
        kt = np.sqrt(1-mu**2) * self.power_spec.k
        
        # Eisenstein, Seo, White, 2007, Eq. 12
        sig_nl_perp = 3.26 # Mpc/h
        f = self.cosmo.growth_rate # lograthmic growth (at z_ref)
        sig_nl_par = (1 + f) * sig_nl_perp # Mpc/h
        
        model *= np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))

        # derivative of model wrt to log(k)
        dmodel = np.zeros(self.power_spec.k.size)
        dmodel[1:] = model[1:]-model[:-1]
        dmodel_dlk  = dmodel/self.power_spec.dlogk
                    
        return dmodel_dlk
    
    def get_fisher(self,mu,dp_dlogk,var):
        """Compute fisher matrix for ap, at"""
        h = [mu**2,1 - mu**2]
        if self.covariance.per_mag:
            return np.outer(h,h)[:,:,None] * np.sum(dp_dlogk**2 / var.T, axis=1).T
        else:
            return np.outer(h,h) * np.sum(dp_dlogk**2 / var)
        
    def print_bao(self,fisher_matrix):
        """Print BAO results from Fisher matrix"""
        if self.covariance.per_mag:
            cov = np.linalg.inv(fisher_matrix.T)
            cov_diag = np.diagonal(cov.T,axis1=0,axis2=1)
            sigma_dh = np.sqrt(cov_diag.T[0])
            sigma_da = np.sqrt(cov_diag.T[1])
            corr_coef = cov.T[0,1]/np.sqrt(cov_diag.T[0]*cov_diag.T[1])

            print("ap={},at={},corr={}".format(sigma_da[-1],
                                    sigma_dh[-1],corr_coef[-1]))
        else:
            cov = np.linalg.inv(fisher_matrix)
            sigma_dh = np.sqrt(cov[0,0])
            sigma_da = np.sqrt(cov[1,1])    
            corr_coef = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

            print("ap={},at={},corr={}".format(sigma_da,
                                    sigma_dh,corr_coef))
            
        return sigma_dh, sigma_da, corr_coef



                
            


