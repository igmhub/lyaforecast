"""Control module for lyaforecast. Should be structured as follows: 
    -   we use Covariance class for each config, and store observed powers and INDIVIDUAL covariances in ?dictionaries?
    - Then, using the Fisher class, we compute the parameter measurements 

"""
import configparser
import time 
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import sys

import numpy as np

from lyaforecast import (
    CosmoCamb, Covariance, Spectrograph,
    Survey, PowerSpectrum, Fisher, get_file, 
    setup_logger
)

@dataclass
class FlagStore:
    lya_auto: bool
    cross: bool
    tracer_auto: bool

    @property
    def include_tracer(self) -> bool:
        """return array of bool for included tracers"""
        return list(asdict(self).values())
    @property
    def is_3x2pt(self) -> bool:
        """Return True if all flags are True."""
        return all(asdict(self).values())
    
# Redirect print to logging
class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        message = message.strip()
        if message:
            self.level(message)
    def flush(self):
        pass

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

        #setup logger
        self.logger = setup_logger(self.out_folder)
        self.logger.info('Running BAO forecast')

        #hold spectra names (for including mulitple configs)
        self.spectrum_names = {}

        # which power spectra to forecast
        self.flags = FlagStore(
            lya_auto=self.config['control'].getboolean('lya auto'),
            cross=self.config['control'].getboolean('cross'),
            tracer_auto=self.config['control'].getboolean('tracer auto')
        )

        # tracer types
        self._lya_tracer = self.config['lya forest'].get('tracer')
        self._tracer = self.config['tracer'].get('tracer')
        self._cross_tracer = 'lya_' + self._tracer

        #not used currently - still unsure what to do.
        self._add_spectum_names()
  
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
        
        sigma_at = np.zeros(self._survey.num_z_bins)
        sigma_ap = np.zeros(self._survey.num_z_bins)
        corr_coef = np.zeros(self._survey.num_z_bins)

        for iz, zc in enumerate(self._survey.z_bin_centres):
            self.logger.info(f"z bin = [{self._survey.z_bin_edges[0,iz]}-"
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
            #TEMPORARY
            corr_names_temp = ['lya', self._cross_tracer, self._tracer]
            corr_names_cut = [c for j, c in enumerate(corr_names_temp)
                   if self.flags.include_tracer[j]]
                
            for j,corr in enumerate(corr_names_cut):
                # #temporary, until I update all dependent functions
                if not corr == self._cross_tracer:
                    corr_name = corr +  '_' + corr
                    corr_names_cut[j] = corr_name
                else:
                    corr_name = self._cross_tracer
                p3d_cache[corr] = (np.array([
                    self._power_spec.compute_p3d_hmpc_smooth(
                        zc, self._power_spec.k, mu, 
                        self._covariance.pix_width_kms,
                        self._covariance.pix_res_kms,
                        corr
                    )
                    for mu in self._power_spec.mu
                ]))
            
            #compute measured power spectra (e.g. including noise)
            p3d_obs_cache = {}
            for j,corr in enumerate(corr_names_temp):
                # #temporary, until I update all dependent functions
                if not corr == self._cross_tracer:
                    corr_name = corr +  '_' + corr
                    corr_names_temp[j] = corr_name
                else:
                    corr_name = self._cross_tracer

                p3d_obs_cache[corr_name] = np.array([
                    self._covariance.compute_total_power(self._power_spec.k,
                        mu,
                        corr)
                    for mu in self._power_spec.mu])
                
            fisher_mat = fisher.compute_fisher(p3d_cache,p3d_obs_cache,corr_names_cut)

            if self.flags.is_3x2pt:
                self.results_name = '3x2pt'
            else:
                self.results_name = ' + '.join(self.spectrum_names.values())

            sigma_ap_z, sigma_at_z, corr_coef_z = fisher.print_bao(fisher_mat,self.results_name)

            sigma_ap[iz] = sigma_ap_z
            sigma_at[iz] = sigma_at_z
            corr_coef[iz] = corr_coef_z


        sigma_at_full = 1./np.sqrt(np.sum(1./sigma_at**2))
        sigma_ap_full = 1./np.sqrt(np.sum(1./sigma_ap**2))


        self.logger.info(fr'Full: at ({self.results_name})={sigma_at_full}'
                fr', ap ({self.results_name})={sigma_ap_full}')
       
        data = {}
        data["redshifts"] = self._survey.z_bin_centres
        data["mean redshift"] = self._cosmo.z_ref
        data["magnitudes"] = {self._survey.band:self._survey.maglist}

    def _add_spectum_names(self):
        #needs to be edited for more than one config
        if self.flags.lya_auto:
            lya_auto_name = f'lya({self._lya_tracer})_lya({self._lya_tracer})'
            self.spectrum_names['lya'] = lya_auto_name
        if self.flags.cross:
            cross_name = f'lya({self._lya_tracer})_{self._tracer}'
            self.spectrum_names['cross'] = cross_name
        if self.flags.tracer_auto:
            tracer_auto_name = f'{self._tracer}_{self._tracer}'
            self.spectrum_names['tracer auto'] = tracer_auto_name