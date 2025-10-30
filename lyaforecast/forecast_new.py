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
    setup_logger, Tracer
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


class NewForecast:
    """Main LyaForecast class.
    ...outline...
    """
    # _survey_properties = None
    # _spectro_properties = None
    _tracer_biases = None

    def __init__(self, cfg_path, tracer_biases=None):
        """
        Parameters
        ----------
        cfg_path : string
            Path to main.ini config file
        """

        # init_start_time = time.time()

        print('Initialise forecast')

        # Read config files
        self.config = configparser.ConfigParser()
        self.config.optionxform = lambda option: option
        self.config.read(get_file(cfg_path))

        self.out_file = Path(self.config['output']['filename'])
        self.out_folder = self.out_file.parent

        # setup logger
        self.logger = setup_logger(self.out_folder)
        self.logger.info('Running BAO forecast')

        # # hold spectra names (for including multiple configs)
        # self.spectrum_names = {}

        # which power spectra to forecast
        # self.flags = FlagStore(
        #     lya_auto=self.config['control'].getboolean('lya auto'),
        #     cross=self.config['control'].getboolean('cross'),
        #     tracer_auto=self.config['control'].getboolean('tracer auto')
        # )

        # tracer types
        # self._lya_tracer = self.config['lya forest'].get('tracer')
        # self._tracer = self.config['tracer'].get('tracer')
        # self._cross_tracer = 'lya_' + self._tracer
        # # not used currently - still unsure what to do.
        # self._add_spectum_names()

        tracer_configs = {}
        for key in self.config.keys():
            if key == 'tracer':
                raise ValueError('tracer section must have an id, e.g. [tracer 1]')
            if 'tracer' in key:
                tracer_id = key.split(' ')[1]
                tracer_configs[tracer_id] = self.config[key]

        self.tracers = {}
        for tracer_id, tracer_config in tracer_configs.items():
            tracer = Tracer(tracer_config)
            self.tracers[tracer.name] = tracer

        self.correlations = {}
        for i, t1 in enumerate(self.tracers):
            for j, t2 in enumerate(self.tracers):
                if i > j:
                    continue
                # if 'lya' in t1 and 'lya' in t2 and t1 != t2:
                #     continue  # skip lya_lya cross

                self.correlations[f"{t1}_{t2}"] = (self.tracers[t1], self.tracers[t2])

        correlation_names = self.config['control'].get('correlations', 'all').split(' ')
        self.correlations_to_compute = []
        for key in self.correlations.keys():
            if 'all' in correlation_names or key in correlation_names:
                self.correlations_to_compute.append(key)

        self.logger.info(f"Tracers: {list(self.tracers.keys())}")
        self.logger.info(f"Correlations: {self.correlations}")

        # load survey instance
        self._survey = Survey(self.config)

        # # initialise cosmology
        self._cosmo = CosmoCamb(
            self.config['cosmo'].get('filename'),
            self.config['cosmo'].getfloat('z_ref', None),
            z_centres=self._survey.z_bin_centres
        )

        # load spectrograph instance
        self._spectrograph = {}
        for corr, (tracer1, tracer2) in self.correlations.items():
            if tracer1.type == 'continuous' and tracer2.type == 'continuous':
                if tracer1.name == tracer2.name:
                    self._spectrograph[corr] = Spectrograph(self.config, self._survey, tracer1)
                else:
                    self._spectrograph[corr] = Spectrograph(
                        self.config, self._survey, tracer1, tracer2)

            elif tracer1.type == 'discrete' and tracer2.type == 'discrete':
                self._spectrograph[corr] = None

            else:
                lya_tracer = (
                    tracer1 if tracer1.type == 'continuous' else tracer2
                )
                self._spectrograph[corr] = Spectrograph(self.config, self._survey, lya_tracer)

        # load power spectrum instance
        self._power_spec = PowerSpectrum(self.config, self._cosmo, self._spectrograph)

        # initialise covariance class (McDonald & Eisenstein (2007)),
        # that stores info and methods to compute p3d and its variance
        self._covariance = {}
        for corr, (tracer1, tracer2) in self.correlations.items():
            self._covariance[corr] = Covariance(
                self.config, self._cosmo, self._survey,
                self._spectrograph[corr], self._power_spec, corr, tracer1, tracer2
            )

        self.reconstruction_factor = self.config['survey'].getfloat('reconstruction factor', 1.0)

        if tracer_biases is not None:
            self.add_tracer_biases(tracer_biases)

        self.num_correlations = len(self.correlations)

        # init_end_time = time.time()
        # print(f"Forecast initialized in {init_end_time - init_start_time:.4f} seconds.")

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

    def new_run_forecast(self):
        sigma_at = {
            corr: np.zeros(self._survey.num_z_bins)
            for corr in self.correlations.keys()
        }
        sigma_ap = {
            corr: np.zeros(self._survey.num_z_bins)
            for corr in self.correlations.keys()
        }
        corr_coef = {
            corr: np.zeros(self._survey.num_z_bins)
            for corr in self.correlations.keys()
        }

        sigma_at['total'] = np.zeros(self._survey.num_z_bins)
        sigma_ap['total'] = np.zeros(self._survey.num_z_bins)
        corr_coef['total'] = np.zeros(self._survey.num_z_bins)

        for iz, zc in enumerate(self._survey.z_bin_centres):
            fisher_input = {
                'p3d_cache': {},
                'p3d_obs_cache': {}
            }
            print(
                f"z bin = [{self._survey.z_bin_edges[0,iz]}-"
                f"{self._survey.z_bin_edges[1,iz]}], bin centre = {zc}"
            )

            for ic, (corr_name, (tracer1, tracer2)) in enumerate(self.correlations.items()):
                print(f"Computing correlation: {corr_name}")

                # Update tracer bias if provided
                if self._tracer_biases is not None:
                    self._power_spec.bias.set_tracer_bias(self._tracer_biases[iz], zbin_index=iz)

                # call function, setting bin width
                self._covariance[corr_name](*self._survey.z_bin_edges[:, iz])
                # this uses Luminosity, density, noise model
                self._covariance[corr_name].compute_eff_density_and_noise(corr_name)

                fisher_input['p3d_cache'][corr_name] = np.array([
                    self._power_spec.compute_p3d_hmpc_smooth(
                        zc, self._power_spec.k, mu,
                        self._covariance[corr_name].pix_width_kms,
                        self._covariance[corr_name].pix_res_kms,
                        corr_name
                    )
                    for mu in self._power_spec.mu
                ])

                fisher_input['p3d_obs_cache'][corr_name] = np.array([
                    self._covariance[corr_name].compute_total_power(
                        self._power_spec.k, mu, corr_name)
                    for mu in self._power_spec.mu
                ])

                # print(f"Computed P(k, mu) for correlation: {corr_name}")
                # print(f"P(k, mu) sum: {fisher_input['p3d_cache'][corr_name].sum()}")
                # print(f"P_obs(k, mu) sum: {fisher_input['p3d_obs_cache'][corr_name].sum()}")

            # number of modes as a function of k, in z bin
            # This should be the same for all correlations in a given z bin
            num_modes_k = self._covariance[corr_name].num_modes

            # Compute fisher for each correlation separately
            # For cross_correlations need to pass the two autos as well (no idea why)
            for ic, corr in enumerate(self.correlations.keys()):
                if corr not in self.correlations_to_compute:
                    continue

                print(f"Computing Fisher for correlation: {corr}")
                # initialise Fisher matrix computation class
                fisher = Fisher(
                    self._power_spec, self._cosmo, num_modes_k, zbin_index=iz,
                    reconstruction_factor=self.reconstruction_factor
                )

                split_corr = corr.split('_')
                if split_corr[0] == split_corr[1]:
                    p3d_cache = {corr: fisher_input['p3d_cache'][corr]}
                    p3d_obs_cache = {corr: fisher_input['p3d_obs_cache'][corr]}
                else:
                    tracer1_auto = f"{split_corr[0]}_{split_corr[0]}"
                    tracer2_auto = f"{split_corr[1]}_{split_corr[1]}"
                    p3d_cache = {
                        corr: fisher_input['p3d_cache'][corr],
                    }
                    p3d_obs_cache = {
                        corr: fisher_input['p3d_obs_cache'][corr],
                        tracer1_auto: fisher_input['p3d_obs_cache'][tracer1_auto],
                        tracer2_auto: fisher_input['p3d_obs_cache'][tracer2_auto],
                    }

                fisher_mat = fisher.compute_fisher(p3d_cache, p3d_obs_cache, [corr])
                sigma_ap_z, sigma_at_z, corr_coef_z = fisher.print_bao(fisher_mat, corr)

                sigma_ap[corr][iz] = sigma_ap_z
                sigma_at[corr][iz] = sigma_at_z
                corr_coef[corr][iz] = corr_coef_z

            # initialise Fisher matrix computation class
            fisher = Fisher(
                self._power_spec, self._cosmo, num_modes_k, zbin_index=iz,
                reconstruction_factor=self.reconstruction_factor
            )

            # Compute fisher for all correlations combined
            p3d_subset = {
                key: fisher_input['p3d_cache'][key]
                for key in self.correlations_to_compute
            }
            fisher_mat = fisher.compute_fisher(
                p3d_subset, fisher_input['p3d_obs_cache'],
                self.correlations_to_compute
            )

            name = f'{self.num_correlations}x2pt'
            sigma_ap_z, sigma_at_z, corr_coef_z = fisher.print_bao(fisher_mat, name)

            sigma_ap['total'][iz] = sigma_ap_z
            sigma_at['total'][iz] = sigma_at_z
            corr_coef['total'][iz] = corr_coef_z

        data = {}
        data["redshifts"] = self._survey.z_bin_centres
        data["fiducial redshift"] = self._cosmo.z_ref

        for corr in self.correlations.keys():
            data[corr] = {
                "sigma_at": sigma_at[corr],
                "sigma_ap": sigma_ap[corr],
                "corr_coef": corr_coef[corr],
            }

        data["sigma_at"] = sigma_at['total']
        data["sigma_ap"] = sigma_ap['total']
        data["corr_coef"] = corr_coef['total']

        return data

    def run_forecast(self):
        sigma_at = np.zeros(self._survey.num_z_bins)
        sigma_ap = np.zeros(self._survey.num_z_bins)
        corr_coef = np.zeros(self._survey.num_z_bins)

        fisher_input = {}
        for iz, zc in enumerate(self._survey.z_bin_centres):
            self.logger.info(
                f"z bin = [{self._survey.z_bin_edges[0,iz]}-"
                f"{self._survey.z_bin_edges[1,iz]}], bin centre = {zc}"
            )

            # observed wavelength range from redshift limits
            lmin = self._cosmo.LYA_REST * (1 + self._survey.z_bin_edges[0, iz])
            lmax = self._cosmo.LYA_REST * (1 + self._survey.z_bin_edges[1, iz])

            # call function, setting bin width
            self._covariance(lmin, lmax)

            # this uses Luminosity, density, noise model
            self._covariance.compute_eff_density_and_noise()

            # number of modes as a function of k, in z bin
            num_modes_k = self._covariance.num_modes

            # initialise Fisher matrix computation class
            fisher = Fisher(
                self._power_spec, self._cosmo, num_modes_k, zbin_index=iz,
                reconstruction_factor=self.reconstruction_factor
            )

            # Update tracer bias if provided
            if self._tracer_biases is not None:
                self._power_spec.bias.set_tracer_bias(self._tracer_biases[iz], zbin_index=iz)

            # Compute P(k, mu) for all mu at once
            # Resulting shape will be (len(mu), len(k))
            p3d_cache = {}
            # TEMPORARY
            corr_names_temp = ['lya', self._cross_tracer, self._tracer]
            corr_names_cut = [
                c for j, c in enumerate(corr_names_temp)
                if self.flags.include_tracer[j]
            ]

            for j, corr in enumerate(corr_names_cut):
                # #temporary, until I update all dependent functions
                if not corr == self._cross_tracer:
                    corr_name = corr + '_' + corr
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

            # compute measured power spectra (e.g. including noise)
            p3d_obs_cache = {}
            for j, corr in enumerate(corr_names_temp):
                # #temporary, until I update all dependent functions
                if not corr == self._cross_tracer:
                    corr_name = corr + '_' + corr
                    corr_names_temp[j] = corr_name
                else:
                    corr_name = self._cross_tracer

                p3d_obs_cache[corr_name] = np.array([
                    self._covariance.compute_total_power(
                        self._power_spec.k, mu, corr)
                    for mu in self._power_spec.mu])

            fisher_mat = fisher.compute_fisher(
                p3d_cache, p3d_obs_cache, corr_names_cut)

            fisher_input[iz] = {
                'fisher_matrix': fisher_mat,
                'p3d_cache': p3d_cache,
                'num_modes_k': num_modes_k,
                'p3d_obs_cache': p3d_obs_cache,
                'corr_names_cut': corr_names_cut,
            }

            if self.flags.is_3x2pt:
                self.results_name = '3x2pt'
            else:
                self.results_name = ' + '.join(self.spectrum_names.values())

            sigma_ap_z, sigma_at_z, corr_coef_z = fisher.print_bao(fisher_mat, self.results_name)

            sigma_ap[iz] = sigma_ap_z
            sigma_at[iz] = sigma_at_z
            corr_coef[iz] = corr_coef_z

        sigma_at_full = 1./np.sqrt(np.sum(1./sigma_at**2))
        sigma_ap_full = 1./np.sqrt(np.sum(1./sigma_ap**2))

        self.logger.info(
            fr'Full: at ({self.results_name})={sigma_at_full}'
            fr', ap ({self.results_name})={sigma_ap_full}'
        )

        data = {}
        data["redshifts"] = self._survey.z_bin_centres
        data["mean redshift"] = self._cosmo.z_ref
        # data["magnitudes"] = {self._survey.band: self._survey.maglist}
        data["sigma_at"] = sigma_at
        data["sigma_ap"] = sigma_ap
        data["corr_coef"] = corr_coef
        data["sigma_at_full"] = sigma_at_full
        data["sigma_ap_full"] = sigma_ap_full
        # data
        self.data = data

        return data, fisher_input

    def _add_spectum_names(self):
        # needs to be edited for more than one config
        if self.flags.lya_auto:
            lya_auto_name = f'lya({self._lya_tracer})_lya({self._lya_tracer})'
            self.spectrum_names['lya'] = lya_auto_name
        if self.flags.cross:
            cross_name = f'lya({self._lya_tracer})_{self._tracer}'
            self.spectrum_names['cross'] = cross_name
        if self.flags.tracer_auto:
            tracer_auto_name = f'{self._tracer}_{self._tracer}'
            self.spectrum_names['tracer auto'] = tracer_auto_name

    def add_tracer_biases(self, tracer_biases):
        """Add tracer biases instance to forecast.

        Parameters
        ----------
        tracer_biases : TracerBiases
            Instance of TracerBiases class.
        """
        assert len(tracer_biases) == self._survey.num_z_bins, (
            "Length of tracer_biases must match number of survey z bins."
        )
        self._tracer_biases = tracer_biases
