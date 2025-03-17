# lyaforecast
Tools to forecast cosmological constraints from Lyman alpha surveys 

A collaborative code to predict BAO uncertainties as a function of survey characteristics (area, density of lines of sight, signal to noise...). Eventually, we will be able to forecast constraints from P1D as well.

Some basic ingredients include:
 - Spectrograph defined in py/spectrograph.py, currently reads DESI files from desihub to compute expected noise in pixel as a function of quasar magnitude, redshift, and wavelength. 
 - Survey specifications (area, lmin, lmax...) can be found in lyaforecast/survey.py, the setting will likely change soon. Defaults to DESI numbers.
 - The quasar/LBG dn/dz is read from file in lyaforecast/survey.py, currently only compatible with DESI formatting.
 - Simple analytical approximation to P1D(z,k) to estimate the aliasing noise, from Palanque-Delabrouille et al. (2013)
 - Simple analytical (or CAMB + McDonald 2003 based) code to estimate flux P3D(z,k,mu) to estimate the signal.
 - Covariances of multiple correlations estimated in lyaforecast/covariance.py.
 - Weights to estimate covariances stored in lyaforecast/weights.py.
 - Control module for forecasting lyaforecast/forecast.py 

Future plans:
 - Estimate covariance between cross- and auto-correlation.
 - Improve functionality of code, including information for other surveys.
 - Forecast P1D.

Required libraries:
 - numpy
 - scipy
 - camb (Python module for CAMB)