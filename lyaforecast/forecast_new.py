"""Run multi-tracer BAO forecasts from an INI configuration."""

import configparser
from pathlib import Path

import numpy as np

from lyaforecast.cosmoCAMB import CosmoCamb
from lyaforecast.covariance import Covariance
from lyaforecast.fisher import Fisher
from lyaforecast.power_spectrum import PowerSpectrum
from lyaforecast.spectrograph import Spectrograph
from lyaforecast.survey import Survey
from lyaforecast.tracer import Tracer
from lyaforecast.utils import get_file, setup_logger


class NewForecast:
    """Updated entry point for running a multi-tracer BAO forecast.

    Supports an arbitrary number of tracers specified as ``[tracer N]`` sections
    in the ini file, and computes individual and combined Fisher matrices for all
    enabled tracer-pair correlations.
    """

    def __init__(self, cfg_path):
        """
        Parameters
        ----------
        cfg_path : str
            Path to the main .ini configuration file.
        """
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
                self.correlations[f"{t1}_{t2}"] = (self.tracers[t1], self.tracers[t2])

        correlation_names = self.config['control'].get('correlations', 'all').split(' ')
        # add the other ordering
        flipped_correlation_names = list()
        for corr in correlation_names :
            if corr == "all":
                continue
            tt=corr.split("_")
            if len(tt)!=2 :
                raise ValueError(f"correlation name '{corr}' not valid")
            if tt[0] == tt[1]:
                continue
            flipped_cor  = tt[1]+"_"+tt[0]
            flipped_correlation_names.append(flipped_cor)
        correlation_names += flipped_correlation_names

        self.correlations_to_compute = []
        for key in self.correlations.keys():
            if 'all' in correlation_names or key in correlation_names:
                self.correlations_to_compute.append(key)

        self.logger.info(f"Tracers: {list(self.tracers.keys())}")
        self.logger.info(f"Correlations: {list(self.correlations.keys())}")

        # load survey instance
        self._survey = Survey(self.config)

        # initialise cosmology
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

        # register the tracer biases
        for tracer_name , tracer in self.tracers.items() :
            if tracer.bias_func is not None :
                short_name = tracer_name+""
                if tracer_name.find("lya(")==0 :
                    short_name = "lya"
                print(f"Setting bias function for power spectrum for {short_name}")
                self._power_spec.bias.set_density_bias_func(short_name,tracer.bias_func)
            else :
                print(f"No bias function provided for {tracer_name}")

        # initialise covariance class (McDonald & Eisenstein (2007)),
        # that stores info and methods to compute p3d and its variance
        self._covariance = {}
        for corr, (tracer1, tracer2) in self.correlations.items():
            self._covariance[corr] = Covariance(
                self.config, self._cosmo, self._survey,
                self._spectrograph[corr], self._power_spec, corr, tracer1, tracer2
            )

        self.reconstruction_factor = self.config['survey'].getfloat('reconstruction factor', 1.0)

        self.num_correlations = len(self.correlations)

    @property
    def cosmo(self):
        """Cosmological model instance.

        Returns
        -------
        CosmoCamb
        """
        return self._cosmo

    @property
    def survey(self):
        """Survey description instance.

        Returns
        -------
        Survey
        """
        return self._survey

    @property
    def spectrograph(self):
        """Spectrograph model instance(s) keyed by correlation name.

        Returns
        -------
        dict
        """
        return self._spectrograph

    @property
    def power_spectrum(self):
        """Power spectrum model instance.

        Returns
        -------
        PowerSpectrum
        """
        return self._power_spec

    def new_run_forecast(self):
        """Run per-correlation and combined BAO forecasts over all redshift bins.

        Returns
        -------
        data : dict
            Redshifts, contiguous zedges, and fiducial redshift; a dictionary of
            sigma_at, sigma_ap, and corr_coef arrays for each correlation; and
            combined sigma_at, sigma_ap, and corr_coef arrays at the top level.
            Unselected correlations retain zero-filled arrays. No results file
            is written; [output] filename determines the logging directory.
        """
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

            # number of modes as a function of k, in z bin
            # This should be the same for all correlations in a given z bin
            num_modes_k = self._covariance[corr_name].num_modes

            # Compute fisher for each correlation separately
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

            # Compute fisher for all correlations combined
            fisher = Fisher(
                self._power_spec, self._cosmo, num_modes_k, zbin_index=iz,
                reconstruction_factor=self.reconstruction_factor
            )

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
        assert(np.all(np.abs(self._survey.z_bin_edges[0][1:]-self._survey.z_bin_edges[1][:-1])<0.0001))
        data["zedges"]    = np.append(self._survey.z_bin_edges[0],self._survey.z_bin_edges[1][-1])
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
