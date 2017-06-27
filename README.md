# lyaforecast
Tools to forecast cosmological constraints from Lyman alpha surveys 

The idea is to build a collaborative code to predict not only BAO uncertainties as a function of survey characteristics (area, density of lines of sight, signal to noise...) but also to eventually have forecasts for P1D as well.

For now (June 27, 2017) it can only forecast the errorbars on a Lya P3D measurement, but hopefully it will also forecast BAO uncertainties.

Some basic ingredients include:
 - Spectrograph defined in py/spectrograph.py, currently reads DESI files from desihub to compute expected noise in pixel as a function of quasar magnitude, redshift, and wavelength. 
 - Survey specifications (area, lmin, lmax...) can be found in py/forecast.py, the setting will likely change soon. Defaults to DESI numbers.
 - Quasar luminosity function is read from file using py/qsoLF.py, currently reads default DESI numbers.
 - Simple analytical approximation to P1D(z,k) to estimate the aliasing noise, from Palanque-Delabrouille et al. (2013)
 - Simple analytical (or CAMB + McDonald 2003 based) code to estimate flux P3D(z,k,mu) to estimate the signal.

To do: 
 - Standard BAO Fisher forecast code that would take as input the signal to noise per band power, and return alphas as a function of z bin.
 - Forecast quasar-lya cross-correlations and its covariance with auto-correlation.
 - Forecast P1D.

Required libraries:
 - numpy
 - pylab
 - scipy
 - camb (Python module for CAMB)
 
 Examples:
  - plot_noise: plot DESI noise rms per pixel, as a function of quasar redshift, magnitude and pixel wavelength
  - plot_camb_linPk: plot LCDM matter power spectrum using CAMB
  - plot_cosmo: same than plot_camb_linPk, but using our own wrapper (cosmoCAMB)
  - plot_P3D: plot flux 3D power using CAMB and biasing from McDonald (2003)
  - plot_P1D: plot flux 1D power using analyticial function from Palanque-Delabrouille et al. (2013)
  - plot_qsoLF: plot QSO luminosity funtion, as a function of quasar redshift and magnitude
  - plot_fisher: plot expected 3D P(z,k,mu) and errorbars 
 
