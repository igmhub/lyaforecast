"""Control module for lyaforecast. Should be structured as follows: 
    -   we use Covariance class for each config, and store observed powers and INDIVIDUAL covariances in ?dictionaries?
    - Then, using the Fisher class, we compute the parameter measurements 

"""
import configparser
import time 
from pathlib import Path
import logging

import numpy as np

from lyaforecast import (
    CosmoCamb, Covariance, Spectrograph,
    Survey, PowerSpectrum, get_file, 
    setup_logger
)

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

        print('Initialise forecast')

        # Read config files
        self.config = configparser.ConfigParser()
        self.config.optionxform = lambda option: option
        self.config.read(get_file(cfg_path))

        self.out_file = Path(self.config['output']['filename'])
        self.out_folder = self.out_file.parent

        # tracer types
        self._lya_tracer = self.config['lya forest'].get('tracer')
        self._tracer = self.config['tracer'].get('tracer')
        self._cross_tracer = 'lya_' + self._tracer

        # which power spectra to forecast
        self._auto_flag = self.config['control'].getboolean('lya auto')
        self._cross_flag = self.config['control'].getboolean('cross')
        self._tracer_auto_flag = self.config['control'].getboolean('tracer auto')

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
        
        #init_end_time = time.time()
        #print(f"Forecast initialized in {init_end_time - init_start_time:.4f} seconds.")

    def run_forecast(self):
        #setup logger
        logger = setup_logger(self.out_folder)
        logger.info('Running BAO forecast')

        if self.covariance.per_mag:
            sigma_dh_lya = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_lya = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_dh_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_dh_cross = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_cross = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_dh_lya_lya_lya_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_lya_lya_lya_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            corr_coef = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
        else:
            sigma_da_lya = np.zeros(self.survey.num_z_bins)
            sigma_dh_lya = np.zeros(self.survey.num_z_bins)
            sigma_da_tracer = np.zeros(self.survey.num_z_bins)
            sigma_dh_tracer = np.zeros(self.survey.num_z_bins)
            sigma_da_cross = np.zeros(self.survey.num_z_bins)
            sigma_dh_cross = np.zeros(self.survey.num_z_bins)
            sigma_da_lya_lya_lya_tracer = np.zeros(self.survey.num_z_bins)
            sigma_dh_lya_lya_lya_tracer = np.zeros(self.survey.num_z_bins)
            corr_coef = np.zeros(self.survey.num_z_bins)

        for iz, zc in enumerate(self.survey.z_bin_centres):
            logger.info(f"z bin = [{self.survey.z_bin_edges[iz,0]}-"
                        f"{self.survey.z_bin_edges[iz,1]}], bin centre = {zc}")

            # observed wavelength range from redshift limits
            lmin = self.cosmo.LYA_REST * (1 + self.survey.z_bin_edges[iz,0])
            lmax = self.cosmo.LYA_REST * (1 + self.survey.z_bin_edges[iz,1])

            #call function, setting bin width
            self.covariance(lmin,lmax)

            # some survey settings
            pix_width = self.covariance._pix_kms
            resolution = self.covariance._res_kms

            # this uses Luminosity, density, noise model
            self.covariance.compute_eff_density_and_noise()
                    
            #empty array for fisher information
            #if per magnitude add extra dimension
            if self.covariance.per_mag:
                fisher_matrix_lya = np.zeros((2,2,self.survey.num_mag_bins))
                fisher_matrix_tracer = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)
            else:
                fisher_matrix_lya = np.zeros((2,2))      
                fisher_matrix_tracer = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)      

            
            for i, mu in enumerate(self.power_spec.mu):
                p3d_lya = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                p3d_tracer = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                            mu, pix_width, resolution, self._tracer) 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                p3d_cross = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                            mu, pix_width, resolution, self._cross_tracer) 
                                            for k in self.power_spec.k]) # (Mpc/h)**3

                #var^2/P^2
                p3d_lya_var = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                p3d_tracer_var = np.array([self.covariance.compute_tracer_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                p3d_cross_var = np.array([self.covariance.compute_cross_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6


                #get derivatives of p3d
                dp3d_lya_dlogk = self.get_dp_dlogk(p3d_lya,mu)
                dp3d_tracer_dlogk = self.get_dp_dlogk(p3d_tracer,mu)
                dp3d_cross_dlogk = self.get_dp_dlogk(-p3d_cross,mu)
                
                # k = sqrt( kp**2 + kt**2)
                # k'  = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
                # k' = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
                # dk/dap         = mu2 * k
                # dlog(k)/dap    = mu2
                # dlog(k)/dat    = (1-mu2)
                # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
                # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
                fisher_matrix_lya += self.get_fisher(mu,dp3d_lya_dlogk,p3d_lya_var)
                fisher_matrix_tracer += self.get_fisher(mu,dp3d_tracer_dlogk,p3d_tracer_var)
                fisher_matrix_cross += self.get_fisher(mu,dp3d_cross_dlogk,p3d_cross_var)
            

            sigma_dh_lya_z, sigma_da_lya_z, corr_coef_lya_z = self.print_bao(fisher_matrix_lya,'lya')
            sigma_dh_tracer_z, sigma_da_tracer_z, corr_coef_tracer_z = self.print_bao(fisher_matrix_tracer,self._tracer)
            sigma_dh_cross_z, sigma_da_cross_z, corr_coef_cross_z = self.print_bao(fisher_matrix_cross,'cross')

            sigma_dh_lya[iz] = sigma_dh_lya_z
            sigma_da_lya[iz] = sigma_da_lya_z
            corr_coef[iz] = corr_coef_lya_z

            sigma_dh_tracer[iz] = sigma_dh_tracer_z
            sigma_da_tracer[iz] = sigma_da_tracer_z

            sigma_dh_cross[iz] = sigma_dh_cross_z
            sigma_da_cross[iz] = sigma_da_cross_z

            bao_corr_coef = 0
            
            sigma_da_lya_lya_lya_tracer[iz] = self.combine_BAO(sigma_da_lya_z,sigma_da_cross_z,bao_corr_coef)
            sigma_dh_lya_lya_lya_tracer[iz] = self.combine_BAO(sigma_dh_lya_z,sigma_dh_cross_z,bao_corr_coef)

            print(f'at (lyaxlya + lyax{self._tracer}): {sigma_da_lya_lya_lya_tracer[iz]}, ap (lyaxlya + lyax{self._tracer}): {sigma_dh_lya_lya_lya_tracer[iz]}')
        
        
        if self.covariance.per_mag:
            sigma_da_combined_lya_m = 1./np.sqrt(np.sum(1./sigma_da_lya**2,axis=0))
            sigma_dh_combined_lya_m = 1./np.sqrt(np.sum(1./sigma_dh_lya**2,axis=0))
            sigma_da_combined_qso_m = 1./np.sqrt(np.sum(1./sigma_da_tracer**2,axis=0))
            sigma_dh_combined_qso_m = 1./np.sqrt(np.sum(1./sigma_dh_tracer**2,axis=0))
            sigma_da_combined_cross_m = 1./np.sqrt(np.sum(1./sigma_da_cross**2,axis=0))
            sigma_dh_combined_cross_m = 1./np.sqrt(np.sum(1./sigma_dh_cross**2,axis=0))
            sigma_da_combined_comb_m = 1./np.sqrt(np.sum(1./sigma_da_lya_lya_lya_tracer**2,axis=0))
            sigma_dh_combined_comb_m = 1./np.sqrt(np.sum(1./sigma_dh_lya_lya_lya_tracer**2,axis=0))

            sigma_da_combined = sigma_da_combined_lya_m[-1]
            sigma_dh_combined = sigma_dh_combined_lya_m[-1]
            sigma_da_combined_tracer = sigma_da_combined_qso_m[-1]
            sigma_dh_combined_tracer = sigma_dh_combined_qso_m[-1]
            sigma_da_combined_cross = sigma_da_combined_cross_m[-1]
            sigma_dh_combined_cross = sigma_dh_combined_cross_m[-1]
            sigma_da_combined_comb = sigma_da_combined_comb_m[-1]
            sigma_dh_combined_comb = sigma_dh_combined_comb_m[-1]

        else:
            sigma_da_combined = 1./np.sqrt(np.sum(1./sigma_da_lya**2))
            sigma_dh_combined = 1./np.sqrt(np.sum(1./sigma_dh_lya**2))
            sigma_da_combined_tracer = 1./np.sqrt(np.sum(1./sigma_da_tracer**2))
            sigma_dh_combined_tracer = 1./np.sqrt(np.sum(1./sigma_dh_tracer**2))
            sigma_da_combined_cross = 1./np.sqrt(np.sum(1./sigma_da_cross**2))
            sigma_dh_combined_cross = 1./np.sqrt(np.sum(1./sigma_dh_cross**2))
            sigma_da_combined_comb = 1./np.sqrt(np.sum(1./sigma_da_lya_lya_lya_tracer**2))
            sigma_dh_combined_comb = 1./np.sqrt(np.sum(1./sigma_dh_lya_lya_lya_tracer**2))



        print(f'\n Combined: at (lyaxlya)={sigma_da_combined}'
                    fr', ap (lyaxlya)={sigma_dh_combined}')
        print(f'\n Combined: at ({self._tracer})={sigma_da_combined_tracer}'
                    fr', ap ({self._tracer})={sigma_dh_combined_tracer}')
        print(f'\n Combined: at (lyaxtr)={sigma_da_combined_cross}'
                    fr', ap (lyaxtr)={sigma_dh_combined_cross}')
        print(f'\n Combined: at (lyaxlya + lyaxtr)={sigma_da_combined_comb}'
                    fr', ap (lyaxlya + lyaxtr)={sigma_dh_combined_comb}')
       
        data = {}
        data["redshifts"] = self.survey.z_bin_centres
        data["mean redshift"] = self.cosmo.z_ref
        data["magnitudes"] = {self.survey.band:self.survey.maglist}
        data['ap_err_lya_z'] = sigma_dh_lya
        data['at_err_lya_z'] = sigma_da_lya
        data['ap_err_tracer_z'] = sigma_dh_tracer
        data['at_err_tracer_z'] = sigma_da_tracer
        data['ap_err_cross_z'] = sigma_dh_cross
        data['at_err_cross_z'] = sigma_da_cross
        data['ap_err_comb_z'] = sigma_dh_lya_lya_lya_tracer
        data['at_err_comb_z'] = sigma_da_lya_lya_lya_tracer
        
        if self.covariance.per_mag:
            data['ap_err_lya_m'] = sigma_dh_combined_lya_m
            data['at_err_lya_m'] = sigma_da_combined_lya_m
            data['ap_err_qso_m'] = sigma_dh_combined_qso_m
            data['at_err_qso_m'] = sigma_da_combined_qso_m
            data['ap_err_cross_m'] = sigma_dh_combined_cross_m
            data['at_err_cross_m'] = sigma_da_combined_cross_m

        return data