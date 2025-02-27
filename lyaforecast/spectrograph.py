import numpy as np
from scipy.interpolate import RegularGridInterpolator
from lyaforecast.utils import get_dir, get_file, check_file
import sys

class Spectrograph:
    """Class to describe a given spectrograph, and return noise estimates.
        Should be pythonized, right now it's really inefficient."""

    def __init__(self, config, survey):
        """Construct object, probably from files"""
        #this is to be read from file currently.
        self._zq = None
        #magnitude info from survey class (g or r)
        self._band = survey.band
        #exposure number/time to read from file, fixed for now.
        self._file_num_exp = 4
        self._file_exp_time = 4000
        #get directory containing snr/mag files
        self._snr_file_dir = get_dir(config['survey'].get('snr-file-dir'))
        #list of filenames
        self._filenames = list(self._snr_file_dir.glob('*'))
        assert len(self._filenames) > 0, 'SNR files not found'

        #load snr per pixel interpolater
        self._setup_desi_spectro()

    def _read_file(self,mag):
        """Read one of the files with SNR as a function of zq and lambda, given magnitude, band, exptime"""
        #set exposure time, currently fixed at 4000.
        #this is not particularly flexible
        fname = self._snr_file_dir.joinpath(f'sn-spec-lya-20180907-{self._band}{mag}-t{str(self._file_exp_time)}-nexp{self._file_num_exp}.dat')

        print("reading magnitude {} in file {}".format(mag,fname))
        fname = check_file(fname)

        data = np.loadtxt(fname)
        lambda_obs = data[:,0]
        pixel_snr = data[:,1:]

        return lambda_obs,pixel_snr

    def _setup_old(self):
        """Setup objects from file(s). Files were generated using
            desihub/desimodel/bin/desi_quicklya.py"""
        # number of exposures in file with SNR
        self._file_num_exp = 4
        # quasar magnitudes in file
        self.mags = np.arange(19.25,25.0,0.5)
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
        self.SN = RegularGridInterpolator((self.mags,self.zq,self.lobs_A),self.SN)
        return True
    
    def _setup_desi_spectro(self):
        """Setup objects from file(s). Files were generated using
                desihub/desimodel/bin/desi_quicklya.py"""
        
        print('Setting up DESI spectrograph')

        self._read_desi_file_header()

        num_magnitudes=len(self._magnitudes)
        for i in range(num_magnitudes):
            m = self._magnitudes[i]
            lambda_obs_m,pixel_snr_mag = self._read_file(m)
            if i == 0:
                self._lambda_obs_m = lambda_obs_m
                num_m  = num_magnitudes
                num_z = len(self._zq)
                num_wave = len(lambda_obs_m)

                self._snr_mat = np.zeros((num_m,num_z,num_wave))
            self._snr_mat[i,:,:] = pixel_snr_mag.T

        # setup interpolator
        self._snr_interp = RegularGridInterpolator((self._magnitudes,self._zq,self._lambda_obs_m),self._snr_mat,bounds_error=False, fill_value=None)

    def _read_desi_file_header(self):
        # expect files with following header
        # This is very awkward, but don't currently have a better way to make it more flexible.

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
        magnitudes = []
        # quasar redshifts in file
        self.file_z_qso = None
        keys=["INFILE","BAND","MAG","EXPTIME","NEXP"]
        for filename in self._filenames:
            head=dict()
            file=open(filename)
            
            for line in file.readlines():
                if line[0] != "#" : continue
                line=line.replace("#","").strip()
                for k in keys:
                    if line.find(k)==0:
                        head[k]=line.split()[-1]
                if line.find("Wave")>=0:
                    # will bravely read the redshifts here to make sure we get it right
                    vals=line.split()
                    tmpz=[]
                    for tmp in vals[1:]:
                        if tmp.find("SN(z=")<0:
                            print("error in reading line",line)
                            print("I expect something like:")
                            print("# Wave SN(z=2.0) SN(z=2.25) SN(z=2.5) SN(z=2.75) SN(z=3.0) SN(z=3.25) SN(z=3.5) SN(z=3.75) SN(z=4.0) SN(z=4.25) SN(z=4.5) SN(z=4.75)")
                            sys.exit(12)
                        tmpz.append(float(tmp.replace("SN(z=","").replace(")","")))
                    head["Z"]=np.array(tmpz)
            file.close()

            if self._band is None:
                self._band=head["BAND"]
            else:
                assert(self._band==head["BAND"])
            if self._file_num_exp is None:
                self._file_num_exp=int(head["NEXP"])
            else:
                assert(self._file_num_exp==int(head["NEXP"]))
            if self._zq is None:
                self._zq=head["Z"]
            else:
                assert(np.all(self._zq==head["Z"]))
            magnitudes.append(float(head["MAG"]))

        # quasar magnitudes in file
        self._magnitudes = np.array(magnitudes)
        self._magnitudes.sort()
        print(f'WARNING: maximum brightness capped at magnitude {self._magnitudes[0]}')

        print("In S/N files:")
        print("pass-band for magnitudes =", self._band)
        print("Nexp in file             =", self._file_num_exp)
        print("Redshifts                =", self._zq)
        print("Magnitudes               =", self._magnitudes)

    def range_zq(self):
        """Return range of quasar redshifts from file"""
        return self.zq[0],self.zq[-1]
  
    def range_mag(self):
        """Return range of magnitudes from file"""
        return self.mags[0],self.mags[-1]
  
    def range_lobs_A(self):
        """Return range of wavelengths from file"""
        return self.lobs_A[0],self.lobs_A[-1]
  
    def get_pixel_rms_noise(self,rmag,zq,lam_obs,pix_width,num_exp=4):
        """Normalized noise RMS as a function of observed magnitude, quasar
          redshift, pixel wavelength (in A), and pixel width (in A).
          Normalized means that this is the noise for delta_flux, not flux, and
          brighter quasars will have less normalized noise.
          In other words, this is inverse of signal to noise.
          If S/N = 0, or not covered, return very large number."""
        large_noise=1e10  
        if rmag > self._magnitudes[-1]: 
            print('WARNING: extrapolating beyond stored magnitude information')
            #print(f'mag {rmag} too faint, returning large noise')
            return large_noise        
        if zq > self._zq[-1] or zq < self._zq[0]: 
            print(f'zqso {zq} out of range, returning large noise')
            return large_noise
        
        if (self._lambda_obs_m[-1] < lam_obs) or (lam_obs < self._lambda_obs_m[0]):
            print('Forest wavelength out of bounds, returning large noise')
            return large_noise


        # if brighter than minimum magnitude, use minimum
        # (c++ code does extrapolation, not clear what is better)
        trmag = np.fmax(rmag, self._magnitudes[0])

        # DESI file returns S/N per Angstrom, per file_num_exp exposures
        snr_per_ang = self._snr_interp([trmag,zq,lam_obs])
        # scale with pixel width
        snr_per_exp = snr_per_ang * np.sqrt(pix_width)
        # scale with number of exposures
        snr = snr_per_exp * np.sqrt(num_exp / self._file_num_exp)
        # prevent division by zero
        snr = np.fmax(snr,1 / large_noise)

        return 1 / snr

    def smooth_kernel_kms(self,pix_kms,res_kms,k_kms):
        """Convolution kernel for the field (square this for power),
            including both pixelization and resolution"""
        # pixelization
        x = np.fmax(0.5 * k_kms * pix_kms, 1.e-10)
        x = 0.5 * k_kms * pix_kms
        pixel_kernel = np.sin(x) / x
        gauss_kernel = np.exp(-0.5 * k_kms**2 * res_kms**2)
        return pixel_kernel * gauss_kernel