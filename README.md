# lyaforecast
Tools to forecast cosmological constraints from Lyman alpha surveys 

The idea is to build a collaborative code to predict not only BAO uncertainties as a function of survey characteristics (area, density of lines of sight, signal to noise...) but also to eventually have forecasts for P1D as well.

Most basic ingredients would include:
 - input file with survey area, and density of quasars per unit redshift and magnitude (r or g)
 - input file with expected instrumental noise as a function of wavelength, for a quasar at a given redshift and magnitude, and as a function of exposure time.
 - simple analytical approximation to P1D(z,k) to estimate the aliasing noise.
 - simple analytical (or CAMB + McDonald 2003 based) code to estimate flux P3D(z,k,mu) to estimate the signal
 - standard BAO Fisher forecast code that would take as input the signal to noise per band power, and return alphas as a function of z bin

Future refinements:
 - accurately include cross-correlations, and covariance with auto-correlation.

Required libraries:
 - numpy
 - classy (Python module for CLASS)
 - scipy
 
 Examples:
  - plot_linPk: plot linear power from CLASS
  - plot_LyaP3D: plot flux 3D power using CLASS and biasing from McDonald (2003)
  - plot_LyaP1D: plot flux 1D power using analyticial function from Palanque-Delabrouille et al. (2013)
  
