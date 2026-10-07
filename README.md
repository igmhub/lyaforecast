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

## Integrated forest-pixel mode (optional)

By default, each redshift bin is represented by a single quasar redshift z_q whose forest is centred on the bin, and the weights and the McDonald & Eisenstein (2007) integrals use the S/N at that (z_q, lambda) only. Setting `integrate forest pixels = True` in `[control]` instead performs the double integral over the quasar redshift and the forest pixels, as in the legacy C++ code: every quasar redshift whose forest overlaps the bin [lambda_min, lambda_max] contributes only the pixels lying inside the bin, each with the S/N of its own (magnitude, z_q, lambda_pix) and a pixel width pix_ang = pix_kms * lambda_pix / c. The z_q integral is a midpoint rule uniform in ln(1+z_q); the pixels are midpoints uniform in ln(lambda) over the exact overlap of forest and bin. The weights w(z_q, lambda, m) are iterated as in the central mode, and the aliasing weight N2/N1^2 and effective noise power N3 pix_kms/N1^2 follow from the integrals N_n (the analogues of I_n L_q). The effective redshift of the forest pixels is returned in the forecast output under `forest_zeff` (one array over z bins per correlation involving a Lya tracer). In this mode the weights of a Lya x discrete-tracer correlation are computed with the Lya auto power of the same Lya tracer (the weights are a property of the forest sample, and the cross power is negative for b_Lya < 0), so `forest_zeff` of the cross equals that of the auto. A bin that the quasar redshift range `[min_zq_forest, max_zq_forest]` leaves without forest pixels has `forest_zeff = NaN`, zero aliasing weight and a very large finite effective noise power (1e20 deg^2 km/s, equivalent to a pixel rms of 1e10 as for out-of-range pixels in the central mode), so the forecast completes and the bin carries no Lya information. Defaults reproduce the central-redshift results exactly.

`[control]` keys (all optional):
 - `integrate forest pixels` (bool, default False): enable the mode above.
 - `num zq integration points` (int, default 128): number of z_q midpoints.
 - `num forest wavelength points` (int, default 64): number of pixel midpoints per z_q inside the bin.
 - `weights include aliasing` (bool, default True): add the aliasing term P1D_w * N2/N1^2 to the signal power when computing the weights (McDonald & Eisenstein 2007, Eq. 19). Applies to both the central and the integrated modes. Set it to False to reproduce forecasts made before 2026-10-06.

Lya `[tracer N]` keys (optional, used only by the integrated mode):
 - `min_zq_forest` (float, default 2.15): lowest quasar redshift of the forest sample.
 - `max_zq_forest` (float, default: upper z edge of the tracer dn/dz table): highest quasar redshift.

Density normalisation key (optional, Lya `[tracer N]` and discrete `qso` tracers, both modes):
 - `min_z_density_norm` (float, default 2.15): `target density` is the total density of the dn/dzdm table rows with bin centre z > `min_z_density_norm`. Note that with 0.1-wide bins centred at x.x5 the default excludes the [2.1, 2.2] bin; use 2.1 when the target density is defined for z > 2.1. Other discrete tracers are normalised over the full table.

Future plans:
 - Estimate covariance between cross- and auto-correlation.
 - Improve functionality of code, including information for other surveys.
 - Forecast P1D.

Required libraries:
 - numpy
 - scipy
 - camb (Python module for CAMB)
 