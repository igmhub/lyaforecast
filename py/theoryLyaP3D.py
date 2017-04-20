import numpy as np
import scipy.interpolate
import analytic_bias_McD2003 as bM03
import analytic_p1d_PD2013 as p1D
import camb
from camb import model, initialpower

class TheoryLyaP3D(object):
    """Make predictions for Lyman alpha 3D P(z,k,mu).
        Should only be used at the level of Fisher forecasts.
        Uses CAMB to generate linear power, and McDonald (2003) for Lya stuff.
        All units internally are in h/Mpc."""

    def __init__(self):
        # get linear power spectrum at z=2.25 from CAMB, in h/Mpc
        self.zref=2.25

        # setup cosmological model
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=67.5,ombh2=0.022,omch2=0.122)
        self.pars.InitPower.set_params(As=2e-9,ns=0.965, r=0)
        self.pars.set_matter_power(redshifts=[self.zref], kmax=10.0)

        # compute and store linear power spectrum (at zref)
        self.pars.NonLinear = model.NonLinear_none
        self.results = camb.get_results(self.pars)
        self.kmin=1.e-4
        self.kmax=1.e1
        kh,_,pk = self.results.get_matter_power_spectrum(minkh=self.kmin,
                                                maxkh=self.kmax,npoints=1000)
        self.linPk = scipy.interpolate.interp1d(kh,pk[0,:])

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
        dkms_dhMpc = self.results.hubble_parameter(z)/self.pars.H0/(1+z)*100
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

