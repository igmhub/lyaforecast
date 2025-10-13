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
    Survey, PowerSpectrum, Fisher, get_file, 
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
        self._cosmo = CosmoCamb(self.config['cosmo'].get('filename'),
                                self.config['cosmo'].getfloat('z_ref', None))

        #load survey instance
        self._survey = Survey(self.config)

        #load spectrograph instance
        self._spectrograph = Spectrograph(self.config, self._survey)

        #load power spectrum instance
        self._power_spec = PowerSpectrum(self.config, self._cosmo, self._spectrograph)

        #initialise covariance class (McDonald & Eisenstein (2007)),
        #  that stores info and methods to compute p3d and its variance
        self._covariance = Covariance(self.config, self._cosmo, self._survey, 
                                     self._spectrograph, self._power_spec)
        
        #init_end_time = time.time()
        #print(f"Forecast initialized in {init_end_time - init_start_time:.4f} seconds.")
    
    @property
    def cosmo(self):
        """Cosmological model instance."""
        return self._cosmo

    @property
    def survey(self):
        """Survey description instance."""
        return self._survey

    @property
    def spectrograph(self):
        """Spectrograph model instance."""
        return self._spectrograph

    @property
    def power_spectrum(self):
        """Power spectrum model instance."""
        return self._power_spec

    def run_forecast(self):
        #setup logger
        logger = setup_logger(self.out_folder)
        logger.info('Running BAO forecast')

        if self._covariance.per_mag:
            sigma_dh_lya = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_da_lya = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_dh_tracer = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_da_tracer = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_dh_cross = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_da_cross = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_dh_lya_lya_lya_tracer = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            sigma_da_lya_lya_lya_tracer = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
            corr_coef = np.zeros((self._survey.num_z_bins, self._survey.num_mag_bins))
        else:
            sigma_da_lya = np.zeros(self._survey.num_z_bins)
            sigma_dh_lya = np.zeros(self._survey.num_z_bins)
            sigma_da_tracer = np.zeros(self._survey.num_z_bins)
            sigma_dh_tracer = np.zeros(self._survey.num_z_bins)
            sigma_da_cross = np.zeros(self._survey.num_z_bins)
            sigma_dh_cross = np.zeros(self._survey.num_z_bins)
            sigma_da_lya_lya_lya_tracer = np.zeros(self._survey.num_z_bins)
            sigma_dh_lya_lya_lya_tracer = np.zeros(self._survey.num_z_bins)
            corr_coef = np.zeros(self._survey.num_z_bins)

        for iz, zc in enumerate(self._survey.z_bin_centres):
            logger.info(f"z bin = [{self._survey.z_bin_edges[iz,0]}-"
                        f"{self._survey.z_bin_edges[iz,1]}], bin centre = {zc}")

            # observed wavelength range from redshift limits
            lmin = self._cosmo.LYA_REST * (1 + self._survey.z_bin_edges[iz,0])
            lmax = self._cosmo.LYA_REST * (1 + self._survey.z_bin_edges[iz,1])

            #call function, setting bin width
            self._covariance(lmin,lmax)

            # this uses Luminosity, density, noise model
            self._covariance.compute_eff_density_and_noise()

            # number of modes as a function of k, in z bin
            num_modes_k = self._covariance.num_modes

            #initialise Fisher matrix computation class
            fisher = Fisher(self._power_spec,self._cosmo,num_modes_k)

                    
            #if per magnitude add extra dimension
            if self._covariance.per_mag:
                fisher_matrix_lya = np.zeros((2,2,self._survey.num_mag_bins))
                fisher_matrix_tracer = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)
            else:
                fisher_matrix_lya = np.zeros((2,2))      
                fisher_matrix_tracer = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)      


            # Loop over tracers
            p3d_cache = {}
            for tracer in ['lya', self._tracer, self._cross_tracer]:
                # Compute P(k, mu) for all mu at once
                # Resulting shape will be (len(mu), len(k))
                p3d_cache[tracer] = abs(np.array([
                    self._power_spec.compute_p3d_hmpc_smooth(
                        zc, self._power_spec.k, mu, 
                        self._covariance.pix_width_kms,
                        self._covariance.pix_res_kms,
                        tracer
                    )
                    for mu in self._power_spec.mu
                ]))

            #compute measured power spectra (e.g. including noise)
            p3d_obs_cache = {}
            for tracer in ['lya', self._tracer, self._cross_tracer]:
                p3d_obs_cache[tracer] = np.array([
                    self._covariance.compute_total_power(self._power_spec.k,
                        mu,
                        tracer)
                    for mu in self._power_spec.mu])

            fisher.compute_fisher(p3d_cache,p3d_obs_cache,self._lya_tracer)


            fisher.compute_derivatives(mu,p3d_lya,f'lya({self._lya_tracer})')
            fisher.compute_derivatives(mu,p3d_lya,self._tracer)
            fisher.compute_derivatives(mu,p3d_lya,f'{self._tracer}_lya({self._lya_tracer})')

            #get derivatives of p3d
            fisher_matrix_lya += fisher.compute_fisher(mu,p3d_lya,p3d_lya_var)
            fisher_matrix_tracer += fisher.compute_fisher(mu,p3d_tracer,p3d_tracer_var)
            fisher_matrix_cross += fisher.compute_fisher(mu,abs(p3d_cross),p3d_cross_var)
                

            sigma_dh_lya_z, sigma_da_lya_z, corr_coef_lya_z = fisher.print_bao(fisher_matrix_lya,f'lya')
            sigma_dh_tracer_z, sigma_da_tracer_z, corr_coef_tracer_z = fisher.print_bao(fisher_matrix_tracer,self._tracer)
            sigma_dh_cross_z, sigma_da_cross_z, corr_coef_cross_z = fisher.print_bao(fisher_matrix_cross,'cross')

            sigma_dh_lya[iz] = sigma_dh_lya_z
            sigma_da_lya[iz] = sigma_da_lya_z
            corr_coef[iz] = corr_coef_lya_z

            sigma_dh_tracer[iz] = sigma_dh_tracer_z
            sigma_da_tracer[iz] = sigma_da_tracer_z

            sigma_dh_cross[iz] = sigma_dh_cross_z
            sigma_da_cross[iz] = sigma_da_cross_z
 
        
        if self._covariance.per_mag:
            sigma_da_combined_lya_m = 1./np.sqrt(np.sum(1./sigma_da_lya**2,axis=0))
            sigma_dh_combined_lya_m = 1./np.sqrt(np.sum(1./sigma_dh_lya**2,axis=0))
            sigma_da_combined_qso_m = 1./np.sqrt(np.sum(1./sigma_da_tracer**2,axis=0))
            sigma_dh_combined_qso_m = 1./np.sqrt(np.sum(1./sigma_dh_tracer**2,axis=0))
            sigma_da_combined_cross_m = 1./np.sqrt(np.sum(1./sigma_da_cross**2,axis=0))
            sigma_dh_combined_cross_m = 1./np.sqrt(np.sum(1./sigma_dh_cross**2,axis=0))

            sigma_da_combined = sigma_da_combined_lya_m[-1]
            sigma_dh_combined = sigma_dh_combined_lya_m[-1]
            sigma_da_combined_tracer = sigma_da_combined_qso_m[-1]
            sigma_dh_combined_tracer = sigma_dh_combined_qso_m[-1]
            sigma_da_combined_cross = sigma_da_combined_cross_m[-1]
            sigma_dh_combined_cross = sigma_dh_combined_cross_m[-1]

        else:
            sigma_da_combined = 1./np.sqrt(np.sum(1./sigma_da_lya**2))
            sigma_dh_combined = 1./np.sqrt(np.sum(1./sigma_dh_lya**2))
            sigma_da_combined_tracer = 1./np.sqrt(np.sum(1./sigma_da_tracer**2))
            sigma_dh_combined_tracer = 1./np.sqrt(np.sum(1./sigma_dh_tracer**2))
            sigma_da_combined_cross = 1./np.sqrt(np.sum(1./sigma_da_cross**2))
            sigma_dh_combined_cross = 1./np.sqrt(np.sum(1./sigma_dh_cross**2))



        print(f'\n Combined: at (lyaxlya)={sigma_da_combined}'
                    fr', ap (lyaxlya)={sigma_dh_combined}')
        print(f'\n Combined: at ({self._tracer})={sigma_da_combined_tracer}'
                    fr', ap ({self._tracer})={sigma_dh_combined_tracer}')
        print(f'\n Combined: at (lyaxtr)={sigma_da_combined_cross}'
                    fr', ap (lyaxtr)={sigma_dh_combined_cross}')
        
        print(fr'\n Combined: at (lyaxlya + lyaxtr)={self.combine_BAO(sigma_da_combined,sigma_da_combined_cross)}'
                fr', ap (lyaxlya + lyaxtr)={self.combine_BAO(sigma_dh_combined,sigma_dh_combined_cross)}')
       
        data = {}
        data["redshifts"] = self._survey.z_bin_centres
        data["mean redshift"] = self._cosmo.z_ref
        data["magnitudes"] = {self._survey.band:self._survey.maglist}
        data['ap_err_lya_z'] = sigma_dh_lya
        data['at_err_lya_z'] = sigma_da_lya
        data['ap_err_tracer_z'] = sigma_dh_tracer
        data['at_err_tracer_z'] = sigma_da_tracer
        data['ap_err_cross_z'] = sigma_dh_cross
        data['at_err_cross_z'] = sigma_da_cross
        data['ap_err_comb_z'] = sigma_dh_lya_lya_lya_tracer
        data['at_err_comb_z'] = sigma_da_lya_lya_lya_tracer
        
        if self._covariance.per_mag:
            data['ap_err_lya_m'] = sigma_dh_combined_lya_m
            data['at_err_lya_m'] = sigma_da_combined_lya_m
            data['ap_err_qso_m'] = sigma_dh_combined_qso_m
            data['at_err_qso_m'] = sigma_da_combined_qso_m
            data['ap_err_cross_m'] = sigma_dh_combined_cross_m
            data['at_err_cross_m'] = sigma_da_combined_cross_m

        return data
    
    def combine_BAO(self,dx1, dx2):

        dx_comb = 1 / (1 / dx1 + 1 / dx2)

        return dx_comb
