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
        # survey (will make object)
        self.area_deg2 = 14000
        # quasar luminosity function
        self.QLF = qLF.QuasarLF()
        # spectrograph
        self.spec = sp.Spectrograph(band='g')

    def compute(self):
        print('compute Fisher forecast and print results')
