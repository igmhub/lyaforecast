import numpy as np

class QuasarLF(object):
  """Class to describe a quasar luminosity function"""
  def __init__(self):
    """Construct object, probably with files describing QL"""
    self._setup()

  def _setup(self):
    """Setup objects from file(s)"""
    self.zmin=2.0
    self.zmax=5.0
    self.rmin=15.0
    self.rmax=23.5

  def QL_range_z(self):
    """Return range of quasar redshifts covered by QL"""
    return self.zmin,self,zmax

  def QL_range_rmag(self):
    """Return range of r magnitudes covered by QL"""
    return self.rmin,self.rmax

  def QL_dN_dV_drmag_hMpc3(self,zq,rmag):
    # comoving density per apparent magnitude, in units of (Mpc/h)^3 
    return 1.e-6

