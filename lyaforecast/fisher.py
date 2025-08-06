"""Class to store and perform operations with Fisher matrices."""
#If we want to combine tracers, we need to compute our fisher matrices with full covariance matrices. Hence, I will start to move everything related to that to this module.
# We want to combine: 
# - lya auto, tracer auto, and cross within one config. 
# - these from a different config (set of tracers).

#The way this is all coded would make multi-config combinations quite tricky.
#how do we remove redundant data? Need to include the compute-lya-auto, compute-cross, compute-tracer-auto options in configs.
#raise an error if two rows are dependent (matrix is non-invertible) because measurements have been repeated.

class Fisher:
    def __init__(self):
        fisher = dict()









    def compute_fisher(self,mu,model,cov):
        """Compute fisher matrix for BAO componenets alpha_parallel and alpha_transverse"""
        dp_dlogk = self._get_dp_dlogk(mu,model)

        h = [mu**2,1 - mu**2]
        if self.covariance.per_mag:
            return np.outer(h,h)[:,:,None] * np.sum(dp_dlogk**2 / cov.T, axis=1).T
        else:
            return np.outer(h,h) * np.sum(dp_dlogk**2 / cov)
        
    def print_bao(self,fisher_matrix,which):
        """Print BAO results from Fisher matrix"""
        if self.covariance.per_mag:
            cov = np.linalg.inv(fisher_matrix.T)
            cov_diag = np.diagonal(cov.T,axis1=0,axis2=1)
            sigma_dh = np.sqrt(cov_diag.T[0])
            sigma_da = np.sqrt(cov_diag.T[1])
            corr_coef = cov.T[0,1]/np.sqrt(cov_diag.T[0]*cov_diag.T[1])

            print(f"ap ({which})={sigma_dh[-1]}, at ({which})={sigma_da[-1]},corr={corr_coef[-1]}")
        else:
            cov = np.linalg.inv(fisher_matrix)
            sigma_dh = np.sqrt(cov[0,0])
            sigma_da = np.sqrt(cov[1,1])    
            corr_coef = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

            print(f"ap ({which})={sigma_dh}, at ({which})={sigma_da},corr={corr_coef}")
            
        return sigma_dh, sigma_da, corr_coef
    

    def _get_dp_dlogk(self,mu,model):
            """Return the differential of the a peak power spectrum component, 
                with respect to log k, add peak broadening."""
            
            pk = self._get_p_pk(model)
            
            pk = self._apply_peak_smoothing(pk,mu)

            # derivative of model wrt to log(k)
            dmodel = np.zeros(self.power_spec.k.size)
            dmodel[1:] = pk[1:]-pk[:-1]
            dmodel_dlk  = dmodel/self.power_spec.dlogk
                        
            return dmodel_dlk
        
    def _get_p_pk(self,model):
        """Get peak only component of linear matter power"""
        x = self.power_spec.logk
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
        kp = mu * self.power_spec.k
        kt = np.sqrt(1-mu**2) * self.power_spec.k
        
        #Following Eisenstein, Seo, White, 2007, Eq. 12
        sig_nl_perp = 3.26 # Mpc/h
        f = self.cosmo.growth_rate # lograthmic growth (at z_ref).   ===> CHECK that this is consistent with what we are doing.
        sig_nl_par = (1 + f) * sig_nl_perp # Mpc/h
        
        return pk * np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))

                
            


