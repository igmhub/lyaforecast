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
    """Flags controlling which power spectra are included in the forecast.

    Parameters
    ----------
    lya_auto : bool
        Include the Lya forest auto-correlation.
    cross : bool
        Include the Lya x discrete tracer cross-correlation.
    tracer_auto : bool
        Include the discrete tracer auto-correlation.
    """

    lya_auto: bool
    cross: bool
    tracer_auto: bool

    @property
    def include_tracer(self):
        """Return ordered list of booleans for [lya_auto, cross, tracer_auto].

        Returns
        -------
        list of bool
        """
        return list(asdict(self).values())

    @property
    def is_3x2pt(self):
        """Return True if all three correlation flags are enabled.

        Returns
        -------
        bool
        """
        return all(asdict(self).values())


class LoggerWriter:
    """Redirect print statements to a Python logger."""

    def __init__(self, level):
        """
        Parameters
        ----------
        level : callable
            Logger method to call for each non-empty message.
        """
        self.level = level

    def write(self, message):
        """Write a stripped message to the logger.

        Parameters
        ----------
        message : str
            Text to log; empty strings are silently ignored.
        """
        message = message.strip()
        if message:
            self.level(message)

    def flush(self):
        """No-op flush for compatibility with the file-like interface."""
        pass


class Forecast:
    """Main entry point for running a BAO forecast from an ini configuration file."""

    _tracer_biases = None

    def __init__(self, cfg_path):
        """
        Parameters
        ----------
        cfg_path : str
            Path to the main .ini configuration file.
        """
        init_start_time = time.time()

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

        # hold spectra names (for including mulitple configs)
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

        # not used currently - still unsure what to do.
        self._add_spectum_names()

        # load survey instance
        self._survey = Survey(self.config)

        # initialise cosmology
        self._cosmo = CosmoCamb(
            self.config['cosmo'].get('filename'),
            self.config['cosmo'].getfloat('z_ref', None),
            z_centres=self._survey.z_bin_centres
        )

        # load spectrograph instance
        self._spectrograph = Spectrograph(self.config, self._survey)

        # load power spectrum instance
        self._power_spec = PowerSpectrum(self.config, self._cosmo, self._spectrograph)

        # initialise covariance class (McDonald & Eisenstein (2007)),
        #  that stores info and methods to compute p3d and its variance
        self._covariance = Covariance(
            self.config, self._cosmo, self._survey,
            self._spectrograph, self._power_spec
        )

        self.reconstruction_factor = self.config['tracer'].getfloat('reconstruction factor', 1.0)

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
        """Spectrograph model instance.

        Returns
        -------
        Spectrograph
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

    def run_forecast(self):
        """Run the BAO forecast over all redshift bins and return per-bin results.

        Returns
        -------
        data : dict
            Dictionary containing redshifts, sigma_at, sigma_ap, corr_coef arrays,
            and combined sigma_at_full / sigma_ap_full.
        fisher_input : dict
            Per-bin diagnostics including Fisher matrices and cached power spectra.
        """
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
                print(f"Computed P3D for {corr_name} at z={zc}")
                print(f"sum P3D: {np.sum(p3d_cache[corr])}")

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
                print(f"Computed P3D_obs for {corr_name} at z={zc}")
                print(f"sum P3D_obs: {np.sum(p3d_obs_cache[corr_name])}")

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
        data["sigma_at"] = sigma_at
        data["sigma_ap"] = sigma_ap
        data["corr_coef"] = corr_coef
        data["sigma_at_full"] = sigma_at_full
        data["sigma_ap_full"] = sigma_ap_full
        self.data = data

        return data, fisher_input

    def _add_spectum_names(self):
        """Build human-readable spectrum labels from the active correlation flags."""
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
        """Register per-redshift-bin tracer biases to override the analytic defaults.

        Parameters
        ----------
        tracer_biases : array_like
            Sequence of bias values with length equal to ``survey.num_z_bins``.
        """
        assert len(tracer_biases) == self._survey.num_z_bins, (
            "Length of tracer_biases must match number of survey z bins."
        )
        self._tracer_biases = tracer_biases
