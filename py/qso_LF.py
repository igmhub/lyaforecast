import numpy as np
import pylab
import scipy.interpolate

class QuasarLF(object):
    """Class to describe a quasar luminosity function"""

    def __init__(self,fname=None):
        """Construct object, probably with files describing QL"""
        self._setup(fname)

    def _setup(self,fname):
        """Setup objects from file(s)"""
        if not fname: 
            # use file from Christophe / Nathalie
            fname = '../data/dNdzdg_QSO.dat'
        # read table       
        z,m,dNdmdzddeg2 = pylab.loadtxt(fname,unpack=True)
        z = np.unique(z)
        m = np.unique(m)
        # create interpolation object
        self.dNdmdzddeg2 = scipy.interpolate.interp2d(z,m,dNdmdzddeg2)
        # figure out allowed redshift range (will check out of bounds)
        dz = z[1]-z[0]
        self.zmin = z[0] - 0.5*dz
        dz = z[-1] - z[-2]
        self.zmax = z[-1] + 0.5*dz
        # figure out allowed magnitude range (will check out of bounds)
        dm = m[1]-m[0]
        self.mmin = m[0] - 0.5*dm
        dm = m[-1] - m[-2]
        self.mmax = m[-1] + 0.5*dm

    def QL_range_z(self):
        """Return range of quasar redshifts covered by QL"""
        return self.zmin,self.zmax

    def QL_range_mag(self):
        """Return range of magnitudes covered by QL"""
        return self.mmin,self.mmax

    def QL_dNdzdmddeg2(self,zq,mag):
        """Quasar luminosity function, observed units.
        
           Per unit redshift, unit (observed) magnitude and square degree."""
        # check redshift in bounds
        if zq > self.zmax or zq < self.zmin:
            print(zq,'out of z bounds (',self.zmin,',',self.zmax,')')
            raise SystemExit

        # check magnitude in bounds
        if np.amax(mag) > self.mmax or np.amin(mag) < self.mmin:
            print(mag,'out of magbounds (',self.mmin,',',self.mmax,')')
            raise SystemExit

        return self.dNdmdzddeg2(zq,mag)
