import numpy as np
import cosmoCAMB as cCAMB
import theoryLyaP3D as P3D
import qso_LF as qLF
import spectrograph as sp
import analytic_p1d_PD2013 as p1D

class FisherForecast(object):
    """Compute error-bars for Lyman alpha P(z,k,mu) for a given survey.
        Different redshif bins are treated as independent, and right
        now this object only deals with one redshift bin at a time.
        """

    def __init__(self, zref=2.25, cosmo=None):
        """
            Initialize class instance
            
            Parameters
            ----------
            zref : float
            Redshift at which the Lya P3D is defined at
            * Default: 2.25
            
            cosmo : cCAMB.Cosmology
            Cosmological model at the redshift of reference
            * Default: cCAMB.Cosmology(pk_zref=zref)
            
            RETURNS:
            An instance of Spectrum
            
            RAISES:
            * SpectrumFormatError when the given format is not found in FORMATS
            * SpectrumTypeError when the arguments are of incorrect type
            
            EXAMPLES:
            * spec = Spectrum('test_image.fits')
            
            """
        # Lya P3D defined at this redshift
        self.zref=2.25
        # Cosmological model
        if cosmo == None:
            self.cosmo = cCAMB.Cosmology(pk_zref=self.zref)
        else:
            self.cosmo = cosmo
        # Lya P3D theory
        self.LyaP3D = P3D.TheoryLyaP3D(self.cosmo)
        # quasar luminosity function
        self.QLF = qLF.QuasarLF()
        # spectrograph
        self.spec = sp.Spectrograph(band='g')
        # survey (should probably be its own object at some point...)
        self.area_deg2 = 14000.0
        # each mini-survey will cover only (lmin,lmax)
        self.lmin = 3501.0
        self.lmax = 3701.0
        # quasar redshift range
        self.zq_min = 2.0
        self.zq_max = 4.0
        # magnitude range 
        self.mag_min = 16.5
        self.mag_max = 23.0
        # pixel width and resolution in km/s (for now)
        self.pix_kms = 50.0
        self.res_kms = 70.0
        # definition of forest (lrmin,lrmax)
        self.lrmin = 985.0
        self.lrmax = 1200.0
        # S/N weights evaluated at this Fourier mode
        self.kt_w_deg=7.0 # ~ 0.1 h/Mpc
        self.kp_w_kms=0.001 # ~ 0.1 h/Mpc
        # verbosity level
        self.verbose = 1

    def mean_z(self):
        """ given wavelength range covered in bin, compute central redshift"""
        l = np.sqrt(self.lmin*self.lmax)
        z = l/self.cosmo.lya_A-1.0
        return z

    def L_kms(self):
        """Depth of redshift bin, in km/s"""
        c_kms = self.cosmo.c_kms
        L_kms = c_kms*np.log(self.lmax/self.lmin)
        return L_kms

    def Lq_kms(self):
        """Length of Lya forest, in km/s"""
        c_kms = self.cosmo.c_kms
        Lq_kms = c_kms*np.log(self.lrmax/self.lrmin)
        return Lq_kms

    def FluxP1D_kms(self,kp_kms):
        """1D Lya power spectrum in observed coordinates,
            smoothed with pixel widht and resolution."""
        z = self.mean_z()
        # get P1D before smoothing
        P1D_kms = p1D.P1D_z_kms_PD2013(z,kp_kms)
        # smoothing (pixelization and resolution)
        Kernel=self.spec.SmoothKernel_kms(z,self.pix_kms,self.res_kms,kp_kms)
        P1D_kms *= (Kernel*Kernel)
        return P1D_kms

    def FluxP3D_degkms(self,kt_deg,kp_kms,linear=False):
        """3D Lya power spectrum in observed coordinates. 
            Power smoothed with pixel width and resolution.
            If linear=True, it will ignore small scale correction."""
        z = self.mean_z()
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
        # smoothing (pixelization and resolution)
        Kernel=self.spec.SmoothKernel_kms(z,self.pix_kms,self.res_kms,kp_kms)
        P_degkms *= (Kernel*Kernel)
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

    def TotalFluxP3D_degkms(self,kt_deg,kp_kms,linear=False,
                            Pw2D=None,PN_eff=None):
        """Sum of 3D Lya power, aliasing and effective noise power. 
            If Pw2D or PN_eff are not passed, it will compute them"""
        # figure out mean redshift of the mini-survey
        z = self.mean_z()
        # signal
        P3D = self.FluxP3D_degkms(kt_deg,kp_kms,linear)
        # P1D for aliasing
        P1D = self.FluxP1D_kms(kp_kms)
        # check if we have everything
        if not Pw2D or not PN_eff:
            # get Pw2D and PNeff in McDonald & Eisenstein (2007)
            np_eff,Pw2D,PN_eff = self.EffectiveDensityAndNoise()
        PT = P3D + Pw2D*P1D + PN_eff
        return PT

    def FluxP3D_hMpc(self,k_hMpc,mu,linear=False):
        """3D Lya power, in units of (Mpc/h)^3, including pixel width and
            resolution smoothing."""
        z = self.mean_z()
        # decompose into line of sight and transverse components
        kp_hMpc = k_hMpc * mu
        kt_hMpc = k_hMpc * np.sqrt(1.0-mu*mu)
        # transform from comoving to observed coordinates
        dkms_dhMpc = self.cosmo.dkms_dhMpc(z)
        kp_kms = kp_hMpc / dkms_dhMpc
        dhMpc_ddeg = self.cosmo.dhMpc_ddeg(z)
        kt_deg = kt_hMpc * dhMpc_ddeg
        # get 3D power in units of observed coordinates
        P_degkms = self.FluxP3D_degkms(kt_deg,kp_kms,linear)
        # convert into units of (Mpc/h)^3
        P_hMpc = P_degkms * dhMpc_ddeg * dhMpc_ddeg / dkms_dhMpc
        return P_hMpc

    def VarFluxP3D_hMpc(self,k_hMpc,mu,dk_hMpc,dmu,linear=False,
                        Pw2D=None,PN_eff=None):
        """Variance of 3D Lya power, in units of (Mpc/h)^3.
            Note that here 0 < mu < 1.
            If Pw2D or PN_eff are not passed, it will compute them"""
        z = self.mean_z()
        # decompose into line of sight and transverse components
        kp_hMpc = k_hMpc * mu
        kt_hMpc = k_hMpc * np.sqrt(1.0-mu*mu)
        # transform from comoving to observed coordinates
        dkms_dhMpc = self.cosmo.dkms_dhMpc(z)
        kp_kms = kp_hMpc / dkms_dhMpc
        dhMpc_ddeg = self.cosmo.dhMpc_ddeg(z)
        kt_deg = kt_hMpc * dhMpc_ddeg
        # get total power in units of observed coordinates
        totP_degkms = self.TotalFluxP3D_degkms(kt_deg,kp_kms,linear,Pw2D,PN_eff)
        # convert into units of (Mpc/h)^3
        totP_hMpc = totP_degkms * dhMpc_ddeg * dhMpc_ddeg / dkms_dhMpc
        # survey volume in units of (Mpc/h)^3
        V_degkms = self.area_deg2 * self.L_kms()
        V_hMpc = V_degkms * dhMpc_ddeg * dhMpc_ddeg / dkms_dhMpc
        # based on Eq 8 in Seo & Eisenstein (2003), but note that here we
        # use 0 < mu < 1 and they used -1 < mu < 1
        Nmodes = V_hMpc * k_hMpc*k_hMpc*dk_hMpc*dmu / (2*np.pi*np.pi)
        varP = 2 * np.power(totP_hMpc,2) / Nmodes
        return varP

    def I1_degkms(self,zq,mags,weights):
        """Integral 1 in McDonald & Eisenstein (2007).
            It represents an effective density of quasars, and it depends
            on the current value of the weights, that in turn depend on I1.
            We solve these iteratively (converges very fast)."""
        # quasar number density
        dkms_dz = self.cosmo.c_kms / (1+zq)
        dndm_degdz = self.QLF.dNdzdmddeg2(zq,mags)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = mags[1]-mags[0]
        # weighted density of quasars
        I1 = np.sum(dndm_degkms*weights)*dm
        if self.verbose > 2:
            print('dkms_dz',dkms_dz)
            print('dndm_degdz',dndm_degdz)
            print('dndm_degkms',dndm_degkms)
            print('mags',mags)
            print('integrant',dndm_degkms*weights)
            print('I1',I1)
        return I1

    def I2_degkms(self,zq,mags,weights):
        """Integral 2 in McDonald & Eisenstein (2007).
            It is used to set the level of aliasing."""
        # quasar number density
        dndm_degdz = self.QLF.dNdzdmddeg2(zq,mags)
        dkms_dz = self.cosmo.c_kms / (1+zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = mags[1]-mags[0]
        I2 = np.sum(dndm_degkms*weights*weights)*dm
        return I2

    def I3_degkms(self,zq,lc,mags,weights):
        """Integral 3 in McDonald & Eisenstein (2007).
            It is used to set the effective noise power."""
        # pixel noise variance (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # quasar number density
        dndm_degdz = self.QLF.dNdzdmddeg2(zq,mags)
        dkms_dz = self.cosmo.c_kms / (1+zq)
        dndm_degkms = dndm_degdz / dkms_dz
        dm = mags[1]-mags[0]
        I3 = np.sum(dndm_degkms*weights*weights*varN)*dm
        return I3

    def np_eff_degkms(self,zq,mags,weights):
        """Effective density of pixels, n_p^eff in McDonald & Eisenstein (2007).
            It is used in constructing the weights as a function of mag."""
        # get effective density of quasars
        I1 = self.I1_degkms(zq,mags,weights)
        # number of pixels in a forest
        Npix = self.Lq_kms() / self.pix_kms
        np_eff = I1 * Npix
        if self.verbose > 2:
            print('I1',I1)
            print('Npix',Npix)
            print('np_eff',np_eff)
        return np_eff

    def VarN_m(self,zq,lc,mags):
        """Noise pixel variance as a function of magnitude (dimensionless)"""
        z = lc / self.cosmo.lya_A - 1.0
        # pixel in Angstroms
        pix_A = self.pix_kms / self.cosmo.dkms_dlobs(z)
        # noise rms per pixel
        noise_rms = np.empty_like(mags)
        for i,m in enumerate(mags):
            noise_rms[i] = self.spec.PixelNoiseRMS(m,zq,lc,pix_A)
        noise_var = noise_rms*noise_rms
        return noise_var

    def PN_m_degkms(self,zq,lc,mags,weights):
        """Effective noise power as a function of magnitude,
            referred to as P_N(m) in McDonald & Eisenstein (2007).
            Note this is a 3D power, not 1D, and it is used in 
            constructing the weights as a function of magnitude."""
        # pixel noise variance (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # 3D effective density of pixels
        neff = self.np_eff_degkms(zq,mags,weights)
        PN = varN / neff
        if self.verbose > 2:
            print('noise variance',varN)
            print('neff',neff)
            print('PN',PN)
        return PN

    def Weights1(self,P3D_degkms,P1D_kms,zq,lc,mags,weights):
        """Compute new weights as a function of magnitude, using P3D.
            This version of computing the weights is closer to the one
            described in McDonald & Eisenstein (2007)."""
        # 3D noise power as a function of magnitude
        PN = self.PN_m_degkms(zq,lc,mags,weights)
        # effective 3D density of quasars
        I1 = self.I1_degkms(zq,mags,weights)
        # 2D density of lines of sight (units of 1/deg^2)
        n2D_los = I1 * self.Lq_kms()
        # weights include aliasing as signal
        PS = P3D_degkms + P1D_kms / n2D_los
        weights = PS/(PS+PN)
        if self.verbose > 2:
            print('P3D',P3D_degkms)
            print('P1D',P1D_kms)
            print('PN',PN)
            print('I1',I1)
            print('n2D_los',n2D_los)
            print('PA',n2D_los*P1D_kms)
            print('PS',PS)
            print('weights',weights)
        return weights

    def Weights2(self,P3D_degkms,P1D_kms,zq,lc,mags,weights):
        """Compute new weights as a function of magnitude, using pixel var.
            This version of computing the weights is closer to the c++ code
            developed by Pat McDonald and used in official forecasts of DESI.
            It gives identical results than Weights1 above."""
        # noise pixel variance as a function of magnitude (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # pixel variance from P1D (dimensionless)
        var1D = P1D_kms / self.pix_kms
        # effective 3D density of pixels
        neff = self.np_eff_degkms(zq,mags,weights)
        # pixel variance from P3D (dimensionless)
        var3D = P3D_degkms * neff
        # signal variance (include P1D and P3D)
        varS = var3D + var1D
        weights = varS/(varS+varN)
        if self.verbose > 2:
            print('P3D',P3D_degkms)
            print('P1D',P1D_kms)
            print('varN',varN)
            print('var1D',var1D)
            print('neff',neff)
            print('var3D',var3D)
            print('varS',varS)
            print('weights',weights)
        return weights

    def InitialWeights(self,P1D_kms,zq,lc,mags):
        """Compute initial weights as a function of magnitude, using only
            P1D and noise variance."""
        # noise pixel variance as a function of magnitude (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # pixel variance from P1D (dimensionless)
        var1D = P1D_kms / self.pix_kms
        weights = var1D/(var1D+varN)
        if self.verbose > 2:
            print('P1D',P1D_kms)
            print('varN',varN)
            print('noise_rms',np.sqrt(varN))
            print('var1D',var1D)
            print('weights',weights)
        return weights

    def ComputeWeights(self,P3D_degkms,P1D_kms,zq,lc,mags,Niter=3):
        """Compute weights as a function of magnitude. 
            We do it iteratively since the weights depend on I1, and 
            I1 depends on the weights."""
        # compute first weights using only 1D and noise variance
        weights = self.InitialWeights(P1D_kms,zq,lc,mags)
        for i in range(Niter):
            if self.verbose > 1:
                print(i,'<w>',np.mean(weights))
            weights = self.Weights1(P3D_degkms,P1D_kms,zq,lc,mags,weights)
            #weights = self.Weights2(P3D_degkms,P1D_kms,zq,lc,mags,weights)
            if self.verbose > 2:
                print('weights',weights)
        return weights

    def EffectiveDensityAndNoise(self):
        """Compute effective density of lines of sight and eff. noise power.
            Terms Pw2D and PN_eff in McDonald & Eisenstein (2007)."""
        # mean wavelenght of bin
        lc = np.sqrt(self.lmin*self.lmax)
        # redshift of quasar for which forest is centered in z
        lrc = np.sqrt(self.lrmin*self.lrmax)
        zq = lc/lrc-1.0
        if self.verbose>0:
            print('lc, lrc, zq =',lc,lrc,zq)
        # evaluate P1D and P3D for weighting
        P3D_w = self.FluxP3D_degkms(self.kt_w_deg,self.kp_w_kms)
        P1D_w = self.FluxP1D_kms(self.kp_w_kms)

        # The code below is closer to the method described in the publication,
        # but the c++ code by Pat is more complicated. 
        # The code below evaluates all the quantities at the central redshift
        # of the bin, and uses a single quasar redshift assuming that all pixels
        # in the bin have restframe wavelength of the center of the forest.
        # Pat's code, instead, computes an average over both redshift of 
        # absorption and over quasar redshift.

        # set range of magnitudes used (same as in c++ code)
        mmin=self.mag_min
        mmax=self.mag_max
        # same binning as in c++ code
        dm=0.025
        N=int((mmax-mmin)/dm)
        mags = np.linspace(mmin,mmax,N)
        # get weights (iteratively)
        weights = self.ComputeWeights(P3D_w,P1D_w,zq,lc,mags)

        # given weights, compute integrals in McDonald & Eisenstein (2007)
        I1 = self.I1_degkms(zq,mags,weights)
        I2 = self.I2_degkms(zq,mags,weights)
        I3 = self.I3_degkms(zq,lc,mags,weights)
        if self.verbose > 0:
            print('I1, I2, I3 =',I1,I2,I3)

        # length of forest in km/s
        Lq = self.Lq_kms()
        # length of pixel in km/s
        lp = self.pix_kms
        # effective 3D density of pixels
        np_eff = I1*Lq/lp
        # Pw2D in McDonald & Eisenstein (2007)
        Pw2D = I2/I1/I1/Lq
        # PNeff in McDonald & Eisenstein (2007)
        PN_eff = I3*lp/I1/I1/Lq
        if self.verbose > 0:
            print('np_eff, Pw2D, PN_eff =',np_eff,Pw2D,PN_eff)
        return np_eff,Pw2D,PN_eff

