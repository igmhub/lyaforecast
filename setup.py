#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Compute Fisher forecast for Lyman-alpha forest clustering."

setup(name="lya_forecast", 
      version="0.1.0",
      description=description,
      url="https://github.com/igmhub/lyaforecast",
      author="Andreu Font-Ribera",
      py_modules=['analytic_bias_McD2003',
                  'analytic_p1d_PD2013',
                  'cosmoCAMB',
                  'theoryLyaP3D',
                  'forecast',
                  'qso_LF',
                  'spectrograph'],
      package_dir={'': 'py'})

