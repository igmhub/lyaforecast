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


        sigma_at = np.zeros(self._survey.num_z_bins)
        sigma_ap = np.zeros(self._survey.num_z_bins)
        corr_coef = np.zeros(self._survey.num_z_bins)

        for iz, zc in enumerate(self._survey.z_bin_centres):
            logger.info(f"z bin = [{self._survey.z_bin_edges[0,iz]}-"
                        f"{self._survey.z_bin_edges[1,iz]}], bin centre = {zc}")
            
            # observed wavelength range from redshift limits
            lmin = self._cosmo.LYA_REST * (1 + self._survey.z_bin_edges[0,iz])
            lmax = self._cosmo.LYA_REST * (1 + self._survey.z_bin_edges[1,iz])

            #call function, setting bin width
            self._covariance(lmin,lmax)

            # this uses Luminosity, density, noise model
            self._covariance.compute_eff_density_and_noise()

            # number of modes as a function of k, in z bin
            num_modes_k = self._covariance.num_modes

            #initialise Fisher matrix computation class
            fisher = Fisher(self._power_spec,self._cosmo,num_modes_k)
  

            # Compute P(k, mu) for all mu at once
            # Resulting shape will be (len(mu), len(k))
            p3d_cache = {}
            for tracer in ['lya', self._tracer, self._cross_tracer]:
                #temporary, until I update all dependent functions
                if not tracer == self._cross_tracer:
                    tracer_name = tracer +  '_' +tracer
                else:
                    tracer_name = self._cross_tracer

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
                #temporary, until I update all dependent functions
                if not tracer == self._cross_tracer:
                    tracer_name = tracer +  '_' +tracer
                else:
                    tracer_name = self._cross_tracer

                p3d_obs_cache[tracer_name] = np.array([
                    self._covariance.compute_total_power(self._power_spec.k,
                        mu,
                        tracer)
                    for mu in self._power_spec.mu])

            fisher_mat = fisher.compute_fisher(p3d_cache,p3d_obs_cache,self._lya_tracer)

            sigma_ap_z, sigma_at_z, corr_coef_z = fisher.print_bao(fisher_mat,f'3x2pt')

            sigma_ap[iz] = sigma_ap_z
            sigma_at[iz] = sigma_at_z
            corr_coef[iz] = corr_coef_z


        sigma_at_full = 1./np.sqrt(np.sum(1./sigma_at**2))
        sigma_ap_full = 1./np.sqrt(np.sum(1./sigma_ap**2))


        print(fr'\n  at (3x2pt)={sigma_at_full}'
                fr', ap (3x2pt)={sigma_ap_full}')
       
        data = {}
        data["redshifts"] = self._survey.z_bin_centres
        data["mean redshift"] = self._cosmo.z_ref
        data["magnitudes"] = {self._survey.band:self._survey.maglist}
 