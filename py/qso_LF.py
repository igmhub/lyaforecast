import numpy as np
import pylab
import scipy.interpolate

class QuasarLF(object):
    """Class to describe a quasar luminosity function"""

    def __init__(self,filename,total_density=None):
        """Construct object, probably with files describing QL"""
        self._setup_YecheFile(filename,total_density)
        
    def _setup_YecheFile(self,filename,total_density=None):
        """Setup objects from file"""
        print("reading",filename,"in QuasarLF:_setup_YecheFile")
        # read table       
        tz,tm,tdNdmdzddeg2 = pylab.loadtxt(filename,unpack=True)
        z = np.unique(tz)
        m = np.unique(tm)
        if total_density is not None :
            # the requirement is 50 quasars per square degree above 2.15
            z_min_lya = 2.15
            current_total_density = np.sum(tdNdmdzddeg2[tz>z_min_lya])
            print("Scaling dndzdmag from a total density (z>={}) of {} to {}/deg2".format(z_min_lya,current_total_density,total_density))
            tdNdmdzddeg2 *= (total_density/current_total_density)
        
        print("WARNING!!! assume dz=0.2, dm=0.5")
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
