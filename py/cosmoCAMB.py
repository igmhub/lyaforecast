import numpy as np
import scipy.interpolate
import camb
from camb import model, initialpower

class Cosmology(object):
    """Compute cosmological functions using CAMB.

       Plenty of room for improvement, for now just something to get started."""

    def __init__(self, h=0.719, om=0.1414, ob=0.02214, ns=0.961, pk_zref=None, sigma8=0.83906665):
        """Setup cosmological model.

            Default values of the parametesr are the same parameters as in c++ code
            
            Parameters
            ----------
            h : float
            Reduced Hubble constant at z = 0
            * Default: 71.9km/s/Mpc
            
            om : float, optional
            Omega matter: density of non-relativistic matter in units of the
            critical density at z=0.
            * Default: 0.1414
            
            ob : float, optional
            Omega baryons: density of baryonic matter in units of the critical
            density at z=0.
            * Default: 0.02214
            
            ns : float or None, optional
            Spectral index of the power spectrum. If this is set to None (the default),
            any computation that requires its value will raise an exception.
            * Default: 0.961
            
            pk_ref : float or None, optional
            If pk_zref is set, it will compute linear power at z=pk_zref.
            * Default: None
            
            sigma8 : float, optional
            Amplitude of the (linear) power spectrum on the scale of 8 h^{-1}Mpc. If
            this is set to None (the default), any computation that requires its value 
            will raise an exception.
            * Default: 0.83906665 (to get As=2.2e-9)
            """
        
        self.pars = camb.CAMBparams()
        oc = om-ob
        def findAs(sigma8=sigma8):
            As_tmp = 2.2e-9
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=100.0*h,ombh2=ob,omch2=oc)
            pars.InitPower.set_params(As=As_tmp,ns=ns, r=0)
            #Not non-linear corrections couples to smaller scales than you want
            pars.set_matter_power(redshifts=[0.0], kmax=10.0)
            
            #Linear spectra
            pars.NonLinear = model.NonLinear_none
            results = camb.get_results(pars)
            s8 = np.array(results.get_sigma8())
            
            return  As_tmp*(sigma8/s8[0])**2
        As = findAs()
        
        
        self.pars.set_cosmology(H0=100.0*h,ombh2=ob,omch2=oc)
        self.pars.InitPower.set_params(As=As,ns=ns, r=0)
        self.pk_zref=pk_zref
        if self.pk_zref != None:
            self.pars.set_matter_power(redshifts=[self.pk_zref], kmax=10.0)
            # compute and store linear power spectrum (at zref)
            self.pars.NonLinear = model.NonLinear_none
            self.results = camb.get_results(self.pars)
        else:
            # compute only background expansion
            self.results = camb.get_results(self.pars)
    
        # not sure where to put this
        self.c_kms = 2.998e5
        self.lya_A = 1215.67

    def LinPk_hMpc(self,kmin=1.e-4,kmax=1.e1,npoints=1000):
        """Return linear power interpolator in units of h/Mpc, at zref"""
        if self.pk_zref:
            kh,_,pk = self.results.get_matter_power_spectrum(minkh=kmin,
                                                    maxkh=kmax,npoints=npoints)
            return scipy.interpolate.interp1d(kh,pk[0,:])
        else:
            print('if you want LinPk_hMpc, initialize Cosmology with pk_zref')
            raise SystemExit

    def dkms_dhMpc(self,z):
        """Convertion factor from Mpc/h to km/s, at redshift z."""
        return self.results.hubble_parameter(z)/self.pars.H0/(1+z)*100.0

    def dkms_dlobs(self,z):
        """Convertion factor from lambda_obs to km/s, at redshift z."""
        return self.c_kms / self.lya_A / (1+z) 

    def dhMpc_dlobs(self,z):
        """Convertion factor from lambda_obs to Mpc/h, at redshift z."""
        return self.dkms_dlobs(z) / self.dkms_dhMpc(z)

    def dhMpc_ddeg(self,z):
        """Convertion factor from degrees to Mpc/h, at redshift z."""
        dMpc_drad=self.results.angular_diameter_distance(z)*(1+z)
        #print('dMpc_drad',dMpc_drad)
        return dMpc_drad*np.pi/180.0*self.pars.H0/100.0

