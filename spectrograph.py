import numpy as np

class Spectrograph(object):
  """Class to describe a given spectrograph, and return noise estimates"""
  def __init__(self):
    """Construct object, probably from files"""
    if not self._setup():
      print "couldn't setup spectrograph"
      exit()

  def _setup(self):
    """Setup objects from file(s)"""
    # exposure time (seconds), or whatever is stored in files
    self.texp_sec = 4000.0
    return True

  def PixelNoiseRMS(self,rmag,zq,lobs_A,pix_A):
    """Noise RMS as a function of observed magnitude, quasar redshift, 
      pixel wavelength (in A), and pixel width (in A)"""

    # DESI file probably returns noise per Angstrom and unit time
    sigma_N_A_t = 1.0
    # we need to multiply this by sqrt(pix_A)
    sigma_N_t = sigma_N_A_t * np.sqrt(pix_A)
    # and also multiply by exposure time, or number...
    sigma_N = sigma_N_t * np.sqrt(self.texp_sec)
    return sigma_N

