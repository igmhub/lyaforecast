import numpy as np
from constants import LYA_WL, LIGHT_SPEED

class Survey(object):
    """Class to describe a given ssurvey"""
    def __init__(self, area_deg2=14000.0, mini_survey_coverage=(3501.0, 3701.0), zq_range=(2.0, 4.0), mag_range=(16.5, 23.0), pixel_width=50.0, pixel_res=70.0, forest=(985.0, 1200.0), SN_weights=(7.0, 0.001)):
        """
            Construct object
            
            Parameters
            ----------
            area_deg2 : float, optional
            Survey area in [deg^{2}]
            * Default: 14000.0
            
            mini_survey_coverage : tuple, optional
            Each mini-survey will cover only (lmin, lmax)
            * Default: (3501.0, 3701.0)
            
            zq_range : tuple, optional
            Quasar redshift range (zq_min, zq_max)
            * Default: (2.0, 4.0)
            
            mag_range : tuple, optional
            Magnitude range (mag_min, mag_max)
            * Default: (2.0, 4.0)
            
            pixel_width : float, optional
            Pixel's width in [km/s]
            * Default : 50.0
            
            pixel_res : float, optional
            Pixel's resolution in [km/s]
            * Default : 70.0
            
            forest : tuple, optional
            Definition of forest (lrmin,lrmax)
            * Default : (985.0, 1200.0)
            
            SN_weights : tuple, optional
            S/N weights evaluated at this Fourier mode
            * Default : (7.0, 0.001) # at (~0.1h/Mpc)
            
        """
        # survey area
        self.area_deg2 = area_deg2
        
        # each mini-survey will cover only (lmin,lmax)
        self.lmin, self.lmax = mini_survey_coverage
        
        # quasar redshift range
        self.zq_min, self.zq_max = zq_range

        # magnitude range
        self.mag_min, self.mag_max = mag_range
        
        # pixel width and resolution in km/s (for now)
        self.pix_width = pixel_width
        self.res_kms = pixel_res
        
        # definition of forest (lrmin,lrmax)
        self.lrmin, self.lrmax = forest
        
        # S/N weights evaluated at this Fourier mode
        self.kt_w_deg, self.kp_w_kms = SN_weights


    def mean_z(self):
        """ given wavelength range covered in bin, compute central redshift"""
        return np.sqrt(self.lmin*self.lmax)/LYA_WL-1.0

    def L_kms(self):
        """Depth of redshift bin, in km/s"""
        return LIGHT_SPEED*np.log(self.lmax/self.lmin)

    def Lq_kms(self):
        """Length of Lya forest, in km/s"""
        return LIGHT_SPEED*np.log(self.lrmax/self.lrmin)

    def volume(self):
        """survey volume in units of deg^2*kms"""
        return self.area_deg2 * self.L_kms()

    def mean_wavelength(self):
        """mean wavelenght of bin"""
        return np.sqrt(self.lmin*self.lmax)

    def mean_forest_wavelength(self):
        """mean wavelength of the forest"""
        return np.sqrt(self.lrmin*self.lrmax)

