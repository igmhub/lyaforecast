"""Class to store and perform operations with Fisher matrices."""
#If we want to combine tracers, we need to compute our fisher matrices with full covariance matrices. Hence, I will start to move everything related to that to this module.
# We want to combine: 
# - lya auto, tracer auto, and cross within one config. 
# - these from a different config (set of tracers).

#The way this is all coded would make multi-config combinations quite tricky.
#how do we remove redundant data? Need to include the compute-lya-auto, compute-cross, compute-tracer-auto options in configs.
#raise an error if two rows are dependent (matrix is non-invertible) because measurements have been repeated.

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

    def compute_fisher(self,models,measurements,lya_tracer):
        """Compute fisher matrix for BAO componenets alpha_parallel and alpha_transverse"""


        for key in models.keys():
            corr_name = key 
            # if corr_name == 'lya':
            #     corr_name += f'({lya_tracer})'

            self.compute_derivatives(models[key],corr_name)

        breakpoint()
        # models
        # measurements

        # for mu in ...:
            
        # self.compute_derivatives(mu,p3d_lya,self._tracer)
        # self.compute_derivatives(mu,p3d_lya,f'{self._tracer}_lya({lya_tracer})')





        # h = [mu**2, 1 - mu**2]
        # return np.outer(h,h) * np.sum(dp_dlogk**2 / cov)
    
    def compute_derivatives(self,model,corr_name):
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

        # Initialize 2D array
        dmodel_dlk = np.zeros((self._power_spec.mu.size, self._power_spec.k.size))

        # # Loop over mu (or model, if you prefer)
        for i, mod_mu in enumerate(model):
            # Get P(k) for this μ
            pk = self._get_p_pk(mod_mu)
            pk = self._apply_peak_smoothing(pk, self._power_spec.mu[i])

            # Compute derivative along k
            dmodel = np.zeros(self._power_spec.k.size)
            dmodel[1:] = pk[1:] - pk[:-1]  # simple finite difference
            dmodel_dlk[i, :] = dmodel / self._power_spec.dlogk  # store in row i

        # for i, mod_mu in model:

        #     pk = self._get_p_pk(mod_mu)
        #     pk = self._apply_peak_smoothing(pk,self._power_spec.mu[i])

        #     dmodel = np.zeros(self._power_spec.k.size)
        #     dmodel[1:] = pk[1:]-pk[:-1]
        #     dmodel_dlk  = dmodel/self.power_spec.dlogk

        self._derivatives[f'dP_dlogk_{corr_name}'] = dmodel_dlk

                    
        
    def _get_p_pk(self,model):
        """Get peak only component of linear matter power"""
        x = self._power_spec.logk
        y = np.log(model)
        x -= np.mean(x)
        x /= (np.max(x)-np.min(x))
        w=np.ones(x.size)
        w[:3] *= 1.e8 
        coef=np.polyfit(x,y,8,w=w)
        pol=np.poly1d(coef)
        smooth = np.exp(pol(x))
        pk = model-smooth    

        return pk

    def _apply_peak_smoothing(self,pk,mu):
        """Apply non-linear smoothing to BAO peak model"""
        kp = mu * self._power_spec.k
        kt = np.sqrt(1-mu**2) * self._power_spec.k
        
        #Following Eisenstein, Seo, White, 2007, Eq. 12
        sig_nl_perp = 3.26 # Mpc/h
        f = self._cosmo.growth_rate # lograthmic growth (at z_ref).   ===> CHECK that this is consistent with what we are doing.
        sig_nl_par = (1 + f) * sig_nl_perp # Mpc/h
        
        return pk * np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))
    
    def get_full_covariance(self):


        cov_mat = np.zeros((self.ncorr,self.ncorr)) 


        return
    


    @staticmethod
    def print_bao(fisher_matrix,which):
        """Print BAO results from Fisher matrix"""
        # if self.covariance.per_mag:
        #     cov = np.linalg.inv(fisher_matrix.T)
        #     cov_diag = np.diagonal(cov.T,axis1=0,axis2=1)
        #     sigma_dh = np.sqrt(cov_diag.T[0])
        #     sigma_da = np.sqrt(cov_diag.T[1])
        #     corr_coef = cov.T[0,1]/np.sqrt(cov_diag.T[0]*cov_diag.T[1])

        #     print(f"ap ({which})={sigma_dh[-1]}, at ({which})={sigma_da[-1]},corr={corr_coef[-1]}")
        # else:
        cov = np.linalg.inv(fisher_matrix)
        sigma_dh = np.sqrt(cov[0,0])
        sigma_da = np.sqrt(cov[1,1])    
        corr_coef = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

        print(f"ap ({which})={sigma_dh}, at ({which})={sigma_da},corr={corr_coef}")
        
        return sigma_dh, sigma_da, corr_coef

                
            


