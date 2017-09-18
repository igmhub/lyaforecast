import numpy as np
import analytic_bias_McD2003 as bM03
import analytic_p1d_PD2013 as p1D
import cosmoCAMB as cCAMB

class TheoryLyaP3D(object):
    """Make predictions for Lyman alpha 3D P(z,k,mu).
        Should only be used at the level of Fisher forecasts.
        Uses CAMB to generate linear power, and McDonald (2003) for Lya stuff.
        All units internally are in h/Mpc."""

    def __init__(self, cosmo=None):
        """cosmo is an optional cosmoCAMB.Cosmology object"""
        if cosmo:
            self.cosmo=cosmo
            self.zref=cosmo.pk_zref
        else:
            self.zref=2.25
            self.cosmo=cCAMB.Cosmology(pk_zref=self.zref)

        # get linear power spectrum 
        self.kmin=1.e-4
        self.kmax=1.e1
        self.linPk = self.cosmo.LinPk_hMpc(self.kmin,self.kmax,1000)

    def FluxP3D_hMpc(self,z,k_hMpc,mu,linear=False):
        """3D power spectrum P_F(z,k,mu). 

            If linear=True, it will ignore small scale correction."""
        # get linear power at zref
        k = np.fmax(k_hMpc,self.kmin)
        k = np.fmin(k,self.kmax)
        P = self.linPk(k)
        # get flux scale-dependent biasing (or only linear term)
        b = bM03.bias_hMpc_McD2003(k,mu,linear)
        # get (approximated) redshift evolution
        zevol = pow( (1+z)/(1+self.zref), 3.8)
        return P * b * zevol

    def FluxP1D_hMpc(self,z,k_hMpc,res_hMpc=None,pix_hMpc=None):
        """Analytical P1D, in units of h/Mpc instead of km/s."""
        # transform to km/s
        dkms_dhMpc = self.cosmo.dkms_dhMpc(z)
        k_kms = k_hMpc / dkms_dhMpc
        # get analytical P1D from Palanque-Delabrouille (2013)
        P_kms = p1D.P1D_z_kms_PD2013(z,k_kms)
        P_hMpc = P_kms / dkms_dhMpc
        if res_hMpc:
            # smooth with Gaussian
            P_hMpc *= np.exp(-pow(k_hMpc*res_hMpc,2))
        if pix_hMpc:
            # smooth with Top Hat
            kpix = np.fmax(k_hMpc*pix_hMpc,1.e-5)
            P_hMpc *= pow(np.sin(kpix/2)/(kpix/2),2)
        return P_hMpc

