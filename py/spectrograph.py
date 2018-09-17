import sys,os
import numpy as np
import pylab
from scipy.interpolate import RegularGridInterpolator

class Spectrograph(object):
    """Class to describe a given spectrograph, and return noise estimates.
        Should be pythonized, right now is really inefficient."""

    def __init__(self, filenames):
        """Construct object, probably from files"""
        self.filenames=filenames
        if not self._setup():
            print("couldn't setup spectrograph")
            raise SystemExit

    def _read_file(self,fname):
        """Read one of the files with SNR as a function of zq and lambda"""

        data = pylab.loadtxt(fname)
        l_A = data[:,0]
        SN = data[:,1:]
        return l_A,SN

    def _setup(self):
        """Setup objects from file(s). Files were generated using
            desihub/desimodel/bin/desi_quicklya.py"""

        # expect files with following header
        """
# INFILE= /home/guy/software/DESI/desimodel/data/spectra/spec-lya.dat
# BAND= r
# MAG= 19.25
# EXPTIME= 4000.0
# NEXP= 4
#
# Wave SN(z=2.0) SN(z=2.25) SN(z=2.5) SN(z=2.75) SN(z=3.0) SN(z=3.25) SN(z=3.5) SN(z=3.75) SN(z=4.0) SN(z=4.25) SN(z=4.5) SN(z=4.75)

        """
        # find range of magnitudes
        mags=[]

        # pass-band for magnitudes
        self.band = None
        # number of exposures in file with SNR
        self.file_Nexp = None
        # quasar redshifts in file
        self.zq = None
        
        keys=["INFILE","BAND","MAG","EXPTIME","NEXP"]
        for filename in self.filenames :
            head=dict()
            file=open(filename)
            for line in file.readlines() :
                if line[0] != "#" : continue
                line=line.replace("#","").strip()
                for k in keys :
                    if line.find(k)==0 :
                        head[k]=line.split()[-1]
                if line.find("Wave")>=0 :
                    # will bravely read the redshifts here to make sure we get it right
                    vals=line.split()
                    tmpz=[]
                    for tmp in vals[1:] :
                        if tmp.find("SN(z=")<0 :
                            print("error in reading line",line)
                            print("I expect something like:")
                            print("# Wave SN(z=2.0) SN(z=2.25) SN(z=2.5) SN(z=2.75) SN(z=3.0) SN(z=3.25) SN(z=3.5) SN(z=3.75) SN(z=4.0) SN(z=4.25) SN(z=4.5) SN(z=4.75)")
                            sys.exit(12)
                        tmpz.append(float(tmp.replace("SN(z=","").replace(")","")))
                    head["Z"]=np.array(tmpz)
                        

                    
            file.close()
            #print(os.path.basename(filename),head)
            if self.band is None :
                self.band=head["BAND"]
            else :
                assert(self.band==head["BAND"])
            if self.file_Nexp is None :
                self.file_Nexp=int(head["NEXP"])
            else :
                assert(self.file_Nexp==int(head["NEXP"]))
            if self.zq is None :
                self.zq=head["Z"]
            else :
                assert(np.all(self.zq==head["Z"]))
            mags.append(float(head["MAG"]))

        # quasar magnitudes in file
        self.mags = np.array(mags)

        print("In S/N files:")
        print("pass-band for magnitudes =",self.band)
        print("Nexp in file             =",self.file_Nexp)
        print("Redshifts                =",self.zq)
        print("Magnitudes               =",self.mags)
        
       
        
        
        # pixel wavelengths in file
        self.lobs_A = None
        # signal to noise per pixel in file
        self.SN = None
        Nm=len(self.mags)
        for i in range(Nm):
            print("reading mag={} in {}".format(self.mags[i],self.filenames[i]))
            m = self.mags[i]
            l_A,SN = self._read_file(self.filenames[i])
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
  
    def PixelNoiseRMS(self,rmag,zq,lobs_A,pix_A):
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

        # do not scale with number of exposures here
        # we do it in the computation of the S/N files
        # scale with number of exposures
        # SN *= np.sqrt(1.0*Nexp/self.file_Nexp)
        
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
