import numpy as np
import cosmoCAMB as cCAMB
import theoryLyaP3D as P3D
import qso_LF as qLF
import spectrograph as sp
import analytic_p1d_PD2013 as p1D

class FisherForecast(object):
    """Compute error-bars for Lyman alpha P(z,k,mu) for a given survey."""

    def __init__(self):
        # Lya P3D defined at this redshift
        self.zref=2.25
        # Cosmological model
        self.cosmo = cCAMB.Cosmology(self.zref)
        # Lya P3D theory
        self.LyaP3D = P3D.TheoryLyaP3D(self.cosmo)
        # quasar luminosity function
        self.QLF = qLF.QuasarLF()
        # spectrograph
        self.spec = sp.Spectrograph(band='g')
        # survey (will make object)
        self.area_deg2 = 14000
        # redshift range
        self.z_min = 2.0
        self.z_max = 2.2
        # magnitude range 
        self.mag_min = 19.0
        self.mag_max = 23.0
        # pixel width and resolution (in Angstroms)
        self.pix_A = 1.0
        self.res_A = 1.0
        # verbosity level
        self.verbose = 0


    def Nmodes(self,kt_deg,kp_kms,dkt_deg,dkp_kms):
        """Compute number of Fourier modes given Volume and bin width."""
        return 1.0

    def FluxP3D_degkms(self,z,kt_deg,kp_kms,linear=False):
        """3D Lya power spectrum in observed coordinates. 
            If linear=True, it will ignore small scale correction."""
        # transform Mpc/h to km/s
        dkms_dhMpc = self.cosmo.dkms_dhMpc(z)
        kp_hMpc = kp_kms * dkms_dhMpc
        # transform Mpc/h to degrees
        dhMpc_ddeg = self.cosmo.dhMpc_ddeg(z)
        kt_hMpc = kt_deg / dhMpc_ddeg
        # compute polar decomposition
        k_hMpc = np.sqrt(kp_hMpc*kp_hMpc+kt_hMpc*kt_hMpc)
        mu = kp_hMpc / (k_hMpc+1.e-10)
        # compute power in Mpc/h
        P_hMpc = self.LyaP3D.FluxP3D_hMpc(z,k_hMpc,mu,linear)
        # convert power to observed units
        P_degkms = P_hMpc * dkms_dhMpc / dhMpc_ddeg / dhMpc_ddeg
        if self.verbose > 1:
            print('z = ',z)
            print('kp_kms = ',kp_kms)
            print('kt_deg = ',kt_deg)
            print('dkms_dhMpc = ',dkms_dhMpc)
            print('kp_hMpc = ',kp_hMpc)
            print('dhMpc_ddeg = ',dhMpc_ddeg)
            print('kt_hMpc = ',kt_hMpc)
            print('k_hMpc = ',k_hMpc)
            print('mu = ',mu)
            print('P_hMpc =',P_hMpc)
            print('P_degkms =',P_degkms)
        return P_degkms

    def VarFluxP3D_degkms(self,z,kt_deg,kp_kms,linear=False):
        """3D Lya power variance per mode"""
        # signal
        P3D_degkms = self.FluxP3D_degkms(z,kt_deg,kp_kms,linear)
        # P1D for aliasing
        P1D_kms = p1D.P1D_z_kms_PD2013(z,kp_kms)
        # meanwhile...
        n_eff_deg2 = 10.0
        P_total_degkms = P3D_degkms + P1D_kms / n_eff_deg2
        var_P_degkms = 2*P_total_degkms*P_total_degkms
        if self.verbose > 1:
            print('z',z)
            print('kt_deg',kt_deg)
            print('kp_kms',kp_kms)
            print('P3D_degkms',P3D_degkms)
            print('P1D_kms',P1D_kms)
            print('P_total_degkms',P_total_degkms)
            print('var_P_degkms',var_P_degkms)
        return var_P_degkms

