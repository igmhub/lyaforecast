import numpy as np
import pylab
import scipy.interpolate

class QuasarLF(object):
    """Class to describe a quasar luminosity function"""

    def __init__(self, fname='../data/dNdzdg_QSO.dat'):
        """Construct object, probably with files describing QL
            Optionally specify a filename containing dNdzdg info. If not 
            specified use ../data/dNdzdg_QSO.dat
            """
        self._setup_YecheFile(fname)

    def _setup_YecheFile(self, fname):
        """Setup objects from file"""
        # use file from Christophe / Nathalie
        fname = fname
        # read table       
        z,m,tdNdmdzddeg2 = pylab.loadtxt(fname,unpack=True)
        z = np.unique(z)
        m = np.unique(m)
        # assume dz=0.2, dm=0.5
        dz=0.2
        dm=0.5
        tdNdmdzddeg2 /= (dz*dm)
        tdNdmdzddeg2 = np.reshape(tdNdmdzddeg2,[len(z),len(m)])
        # figure out allowed redshift range (will check out of bounds)
        self.zmin = z[0] - 0.5*dz
        self.zmax = z[-1] + 0.5*dz
        # figure out allowed magnitude range (will check out of bounds)
        self.mmin = m[0] - 0.5*dm
        self.mmax = m[-1] + 0.5*dm
        # create interpolation object
        #self._dNdmdzddeg2 = scipy.interpolate.interp2d(z,m,tdNdmdzddeg2)
        self._dNdmdzddeg2 = scipy.interpolate.RectBivariateSpline(z,m,
                  tdNdmdzddeg2,bbox=[self.zmin,self.zmax,self.mmin,self.mmax],
                  kx=2,ky=2)

    def range_z(self):
        """Return range of quasar redshifts covered by QL"""
        return self.zmin,self.zmax

    def range_mag(self):
        """Return range of magnitudes covered by QL"""
        return self.mmin,self.mmax

    def dNdzdmddeg2(self,zq,mag):
        """Quasar luminosity function, observed units.
        
           Per unit redshift, unit (observed) magnitude and square degree."""
        return self._dNdmdzddeg2(zq,mag)
