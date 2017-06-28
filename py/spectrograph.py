import numpy as np
import pylab
from scipy.interpolate import RegularGridInterpolator

class Spectrograph(object):
    """Class to describe a given spectrograph, and return noise estimates.
        Should be pythonized, right now is really inefficient."""

    def __init__(self, band='g'):
        """Construct object, probably from files"""
        self.band=band
        if not self._setup():
            print("couldn't setup spectrograph")
            raise SystemExit

    def _read_file(self,mag):
        """Read one of the files with SNR as a function of zq and lambda"""
        if self.band is 'g':
            fname='../data/LyaDESISNR/sn-spec-lya-g'+str(mag)+'-t4000.dat'
        elif self.band is 'r':
            fname='../data/LyaDESISNR/sn-spec-lya-r'+str(mag)+'-t4000.dat'
        else:
            print('wrong band type in Spectrograph',self.band)
            raise SystemExit
        data = pylab.loadtxt(fname)
        l_A = data[:,0]
        SN = data[:,1:]
        return l_A,SN

    def _setup(self):
        """Setup objects from file(s). Files were generated using
            desihub/desimodel/bin/desi_quicklya.py"""
        # number of exposures in file with SNR
        self.file_Nexp = 4
        # quasar magnitudes in file
        self.mags = np.arange(19.25,25.0,0.50)
        # quasar redshifts in file
        self.zq = np.arange(2.0,4.9,0.25)
        # pixel wavelengths in file
        self.lobs_A = None
        # signal to noise per pixel in file
        self.SN = None
        Nm=len(self.mags)
        for i in range(Nm): 
            m = self.mags[i]
            l_A,SN = self._read_file(m)
            if i == 0:
                self.lobs_A = l_A
                Nm=len(self.mags)
                Nz=len(self.zq)
                Nl=len(self.lobs_A)
                self.SN = np.empty((Nm,Nz,Nl))
            self.SN[i,:,:] = SN.transpose()
    
        # setup interpolator
        self.SN=RegularGridInterpolator((self.mags,self.zq,self.lobs_A),self.SN)
        return True

    def range_zq(self):
        """Return range of quasar redshifts from file"""
        return self.zq[0],self.zq[-1]
  
    def range_mag(self):
        """Return range of magnitudes from file"""
        return self.mags[0],self.mags[-1]
  
    def range_lobs_A(self):
        """Return range of wavelengths from file"""
        return self.lobs_A[0],self.lobs_A[-1]
  
    def PixelNoiseRMS(self,rmag,zq,lobs_A,pix_A,Nexp=4):
        """Normalized noise RMS as a function of observed magnitude, quasar
          redshift, pixel wavelength (in A), and pixel width (in A).
          Normalized means that this is the noise for delta_flux, not flux, and
          brighter quasars will have less normalized noise.
          In other words, this is inverse of signal to noise.
          If S/N = 0, or not covered, return very large number."""
        large_noise=1e10  
        if rmag > self.mags[-1]: return large_noise
        if zq > self.zq[-1] or zq < self.zq[0]: return large_noise
        if lobs_A > self.lobs_A[-1] or lobs_A < self.lobs_A[0]: 
            return large_noise

        # if brighter than minimum magnitude, use minimum
        # (c++ code does extrapolation, not clear what is better)
        trmag = np.fmax(rmag, self.mags[0])
 
        # DESI file returns S/N per Angstrom, per file_Nexp exposures
        SN = self.SN([trmag,zq,lobs_A])
        # scale with pixel width
        SN *= np.sqrt(pix_A)
        # scale with number of exposures
        SN *= np.sqrt(1.0*Nexp/self.file_Nexp)
        # prevent division by zero
        SN = np.fmax(SN,1.0/large_noise)
        return 1.0/SN

    def SmoothKernel_kms(self,z,pix_kms,res_kms,k_kms):
        """Convolution kernel for the field (square this for power),
            including both pixelization and resolution"""
        # pixelization
        x = np.fmax(0.5*k_kms*pix_kms,1.e-10)
        kernelPixel = np.sin(x)/x
        kernelGauss = np.exp(-0.5*k_kms*k_kms*res_kms*res_kms)
        return kernelPixel * kernelGauss
