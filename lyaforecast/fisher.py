"""Class to store and perform operations with Fisher matrices."""

import numpy as np 

class Fisher:
    def __init__(self,power_spec,cosmo,number_modes):
        self._power_spec = power_spec
        self._cosmo = cosmo
        self._num_modes = number_modes

        #currently hard coded for BAO only
        npars = 2
        self.ncorr = 3
        self.fisher_matrix = np.zeros((npars,npars))

        self._derivatives = {}

    def compute_fisher(self,models,measurements,spectra_list):
        """Compute fisher matrix for BAO componenets alpha_parallel and alpha_transverse"""
        fisher = np.zeros((2,2))
        #for a given mu work out fisher
        #sum over k,mu. 
        #then do vectorised.
        for i, mu in enumerate(self._power_spec.mu):
            model_mu = np.stack([models[k][i, :] for k in models.keys()], axis=0)
            dmodel_dlk = self.compute_derivatives(model_mu,mu) 
            pre_factor_mu = np.outer([mu**2,1-mu**2],[mu**2,1-mu**2])

            #measured power spectra
            # build dictionary
            p_measured_dict = {label: measurements[label][i, :] for label in measurements.keys()}

            #To-do: differentiate between labels used to compute C, and measured spectra that are acutally passed.
            #Because cross-correlation needs both measured auto correlations
            p_measured_matrix, label_to_idx = self.gaussian_covariance_array_func(p_measured_dict,spectra_list)
            M = np.moveaxis(p_measured_matrix, -1, 0)
            M_inv = np.linalg.inv(M)
            p_measured_matrix_inv = np.moveaxis(M_inv, 0, -1)

            fisher_mu_k = np.einsum('ik,jik,jk->k', dmodel_dlk, p_measured_matrix_inv, dmodel_dlk)
            fisher_mu = np.sum(fisher_mu_k)
            fisher += pre_factor_mu * fisher_mu

        return fisher

    def p_entry_from_label(self,spectra_dict, x, y):
        """
        Return P_tot array for tracer pair (x,y), handling symmetry.
        """
        key = f'{x}_{y}' if f'{x}_{y}' in spectra_dict else f'{y}_{x}'
        return spectra_dict[key]   

    def gaussian_covariance_array_func(self,measured_power_spectra,labels):
        """
        Compute Gaussian covariance as a NumPy array using labels,
        with helper function defined outside loops.
        
        Parameters
        ----------
        labels : dict
            Strings like 'lya_lya', 'qso_qso', 'lya_qso'.
        measured_power_spectra: dict
            Keys are strings like above, Values are arrays of shape (N_k,)
        
        Returns
        -------
        C_array : array, shape (N_spectra, N_spectra, N_k)
        label_to_index : dict mapping label -> array index
        """
        #labels = list(spectra_dict.keys())
        N_spectra = len(labels)
        N_k = self._power_spec.k.size #next(iter(spectra_dict.values())).shape[0]
        
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

    
    
    def compute_derivatives(self,model,mu):
        """Return the differential of the a peak power spectrum component, 
            with respect to log k for a single value of mu. Also add BAO peak broadening."""
        
        #i.e. 
        # k = sqrt( kp**2 + kt**2)
        # k'  = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
        # k' = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
        # dk/dap         = mu2 * k
        # dlog(k)/dap    = mu2
        # dlog(k)/dat    = (1-mu2)
        # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
        # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)

        # Get P(k) for this μ
        pk = self._get_p_pk(model)
        pk = self._apply_peak_smoothing(pk,mu)

        # Compute derivative along k (with first entry zero padding)
        dmodel_dlk = np.zeros_like(pk)
        dmodel_dlk[:, 1:] = (pk[:, 1:] - pk[:, :-1]) / self._power_spec.dlogk[1:]

        return dmodel_dlk

    def _get_p_pk(self, model):
        """Get peak-only component of linear matter power for multiple models (N, 500)."""
        x = self._power_spec.logk
        x = x - np.mean(x)
        x = x / (np.max(x) - np.min(x))
        w = np.ones(x.size)
        w[:3] *= 1.e8

        pk_list = []
        for row in model:  # loop over N models (shape (500,))
            sign = np.sign(row)
            y = np.log(np.abs(row)+1e-12)
            coef = np.polyfit(x, y, 8, w=w)
            smooth_amp = np.exp(np.polyval(coef, x))
            smooth = sign * smooth_amp
            pk = row - smooth
            pk_list.append(pk)

        return np.vstack(pk_list)

    def _apply_peak_smoothing(self,pk,mu):
        """Apply non-linear smoothing to BAO peak model"""
        kp = mu * self._power_spec.k
        kt = np.sqrt(1-mu**2) * self._power_spec.k
        
        #Following Eisenstein, Seo, White, 2007, Eq. 12
        sig_nl_perp = 3.26 # Mpc/h
        f = self._cosmo.growth_rate # lograthmic growth (at z_ref).   ===> CHECK that this is consistent with what we are doing.
        sig_nl_par = (1 + f) * sig_nl_perp # Mpc/h
        
        return pk * np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))


    @staticmethod
    def print_bao(fisher_matrix,which='result'):
        """Print BAO results from Fisher matrix"""
        cov = np.linalg.inv(fisher_matrix)
        sigma_dh = np.sqrt(cov[0,0])
        sigma_da = np.sqrt(cov[1,1])    
        corr_coef = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

        print(f"ap ({which})={sigma_dh}, at ({which})={sigma_da},corr={corr_coef}")
        
        return sigma_dh, sigma_da, corr_coef

                
            


