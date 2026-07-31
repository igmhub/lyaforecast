"""Class to store and perform operations with Fisher matrices."""

import numpy as np


class Fisher:
    """Fisher matrix computation for BAO parameters alpha_parallel and alpha_transverse."""

    def __init__(self, power_spec, cosmo, number_modes, zbin_index=None, reconstruction_factor=1.0):
        """
        Parameters
        ----------
        power_spec : PowerSpectrum
            Power spectrum model instance.
        cosmo : CosmoCamb
            Cosmological model instance.
        number_modes : ndarray
            Number of modes as a function of k for the current redshift bin.
        zbin_index : int, optional
            Index into the redshift bin arrays for growth factor retrieval.
        reconstruction_factor : float, optional
            BAO reconstruction factor applied to non-linear damping.
        """
        self._power_spec = power_spec
        self._cosmo = cosmo
        self._num_modes = number_modes
        self._reconstruction_factor = reconstruction_factor

        # currently hard coded for BAO only
        npars = 2
        self.ncorr = 3
        self.fisher_matrix = np.zeros((npars, npars))
        self.zbin_index = zbin_index

        self._derivatives = {}

    def compute_fisher(self, models, measurements, spectra_list):
        """Compute the 2x2 Fisher matrix for alpha_parallel and alpha_transverse.

        Parameters
        ----------
        models : dict
            Signal power spectra {corr_name: ndarray of shape (n_mu, n_k)}.
        measurements : dict
            Observed (signal + noise) power spectra {corr_name: ndarray of shape (n_mu, n_k)}.
        spectra_list : list of str
            Ordered list of correlation names included in the Fisher calculation.

        Returns
        -------
        ndarray
            Fisher matrix of shape (2, 2) for (alpha_parallel, alpha_transverse).
        """
        fisher = np.zeros((2, 2))
        for i, mu in enumerate(self._power_spec.mu):
            model_mu = np.stack([models[k][i, :] for k in models.keys()], axis=0)
            dmodel_dlk = self.compute_derivatives(model_mu, mu, spectra_list)
            pre_factor_mu = np.outer([mu**2, 1-mu**2], [mu**2, 1-mu**2])

            # measured power spectra
            p_measured_dict = {label: measurements[label][i, :] for label in measurements.keys()}

            p_measured_matrix, label_to_idx = self.gaussian_covariance_array_func(
                p_measured_dict, spectra_list)
            M = np.moveaxis(p_measured_matrix, -1, 0)
            M_inv = np.linalg.inv(M)
            p_measured_matrix_inv = np.moveaxis(M_inv, 0, -1)

            fisher_mu_k = np.einsum('ik,jik,jk->k', dmodel_dlk, p_measured_matrix_inv, dmodel_dlk)
            fisher_mu = np.sum(fisher_mu_k)
            fisher += pre_factor_mu * fisher_mu

        return fisher

    def p_entry_from_label(self, spectra_dict, x, y):
        """Return the observed power array for tracer pair (x, y), handling symmetry.

        Parameters
        ----------
        spectra_dict : dict
            Dictionary mapping correlation strings to power arrays.
        x : str
            Name of the first tracer.
        y : str
            Name of the second tracer.

        Returns
        -------
        ndarray
            Power spectrum array for the requested tracer pair.
        """
        key = f'{x}_{y}' if f'{x}_{y}' in spectra_dict else f'{y}_{x}'
        return spectra_dict[key]

    def gaussian_covariance_array_func(self, measured_power_spectra, labels):
        """Compute the Gaussian covariance matrix C_{AB}(k) for all correlation pairs.

        Parameters
        ----------
        measured_power_spectra : dict
            Keys are strings like 'lya_lya', 'qso_qso', 'lya_qso'; values are arrays of shape (n_k,).
        labels : list of str
            Ordered list of correlation names defining the covariance matrix rows/columns.

        Returns
        -------
        C_array : ndarray
            Covariance array of shape (n_spectra, n_spectra, n_k).
        label_to_index : dict
            Mapping from correlation label to its row/column index.
        """
        N_spectra = len(labels)
        N_k = self._power_spec.k.size

        label_to_index = {label: i for i, label in enumerate(labels)}
        label_pairs = {label: tuple(label.split('_')) for label in labels}

        C_array = np.zeros((N_spectra, N_spectra, N_k))

        for x, A in enumerate(labels):
            iA, jA = label_pairs[A]
            for y, B in enumerate(labels):
                iB, jB = label_pairs[B]

                P_im = self.p_entry_from_label(measured_power_spectra, iA, iB)
                P_jn = self.p_entry_from_label(measured_power_spectra, jA, jB)
                P_in = self.p_entry_from_label(measured_power_spectra, iA, jB)
                P_ji = self.p_entry_from_label(measured_power_spectra, jA, iB)

                C_array[x, y, :] = (P_im * P_jn + P_in * P_ji) / self._num_modes

        return C_array, label_to_index

    def compute_derivatives(self, model, mu, spectra_list):
        """Compute dP/d(log k) for the BAO peak component, with non-linear damping applied.

        Parameters
        ----------
        model : ndarray
            Signal power spectra stacked over correlations, shape (n_corr, n_k).
        mu : float
            Cosine of the angle to the line of sight for this mu slice.
        spectra_list : list of str
            Ordered list of correlation names matching the model rows.

        Returns
        -------
        ndarray
            Derivatives dP_peak/d(log k) of shape (n_corr, n_k).
        """
        # Get P(k) for this μ
        pk = self._get_p_pk(model)

        # Apply peak smoothing
        for i, name in enumerate(spectra_list):
            if 'lya' not in name:
                pk[i] *= self._get_peak_smoothing(
                    mu, self.zbin_index, reconstruction_factor=self._reconstruction_factor)
            else:
                pk[i] *= self._get_peak_smoothing(mu, self.zbin_index)

        # Compute derivative along k (with first entry zero padding)
        dmodel_dlk = np.zeros_like(pk)
        dmodel_dlk[:, 1:] = (pk[:, 1:] - pk[:, :-1]) / self._power_spec.dlogk[1:]

        return dmodel_dlk

    def _get_p_pk(self, model):
        """Extract the BAO peak component by subtracting a smooth polynomial fit.

        Parameters
        ----------
        model : ndarray
            Signal power spectra, shape (n_corr, n_k).

        Returns
        -------
        ndarray
            Peak-only component of shape (n_corr, n_k).
        """
        x = self._power_spec.logk
        x = x - np.mean(x)
        x = x / (np.max(x) - np.min(x))
        w = np.ones(x.size)
        w[:3] *= 1.e8

        pk_list = []
        for row in model:
            sign = np.sign(row)
            y = np.log(np.abs(row)+1e-12)
            coef = np.polyfit(x, y, 8, w=w)
            smooth_amp = np.exp(np.polyval(coef, x))
            smooth = sign * smooth_amp
            pk = row - smooth
            pk_list.append(pk)

        return np.vstack(pk_list)

    def _get_peak_smoothing(self, mu, zbin_index, reconstruction_factor=1):
        """Return the non-linear BAO damping envelope (Eisenstein, Seo & White 2007, Eq. 12).

        Parameters
        ----------
        mu : float
            Cosine of the angle to the line of sight.
        zbin_index : int or None
            Index into the redshift bin arrays; if None, uses z_ref growth rate.
        reconstruction_factor : float, optional
            Factor reducing non-linear damping (1 = no reconstruction, >1 = partial).

        Returns
        -------
        ndarray
            Gaussian damping envelope of shape (n_k,).
        """
        kp = mu * self._power_spec.k
        kt = np.sqrt(1-mu**2) * self._power_spec.k

        # Following Eisenstein, Seo, White, 2007, Eq. 12
        sig_nl_perp = 3.26  # Mpc/h

        if zbin_index is not None:
            # scale with growth factor
            sig_nl_perp *= self._cosmo.growth_factor_ratios[zbin_index]
            f = self._cosmo.growth_rate_zbins[zbin_index]
        else:
            # scale with growth rate at z_ref
            f = self._cosmo.growth_rate

        # Apply reconstruction factor to reduce non-linear damping
        sig_nl_perp /= np.sqrt(reconstruction_factor)

        sig_nl_par = (1 + f) * sig_nl_perp  # Mpc/h
        return np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))

    @staticmethod
    def print_bao(fisher_matrix, which='result'):
        """Invert the Fisher matrix and print the BAO parameter uncertainties.

        Parameters
        ----------
        fisher_matrix : ndarray
            2x2 Fisher matrix for (alpha_parallel, alpha_transverse).
        which : str, optional
            Label printed alongside the results.

        Returns
        -------
        sigma_dh : float
            Uncertainty on alpha_parallel (line-of-sight BAO).
        sigma_da : float
            Uncertainty on alpha_transverse (angular BAO).
        corr_coef : float
            Correlation coefficient between sigma_dh and sigma_da.
        """
        cov = np.linalg.inv(fisher_matrix)
        sigma_dh = np.sqrt(cov[0, 0])
        sigma_da = np.sqrt(cov[1, 1])
        corr_coef = cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])

        print(f"ap ({which})={sigma_dh}, at ({which})={sigma_da},corr={corr_coef}")

        return sigma_dh, sigma_da, corr_coef
