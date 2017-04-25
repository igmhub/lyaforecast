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
        self.z_min = 2.1
        self.z_max = 3.0
        # magnitude range 
        self.mag_min = 19.0
        self.mag_max = 23.0
        # pixel width and resolution (in Angstroms)
        self.pix_A = 1.0
        self.res_A = 1.0

