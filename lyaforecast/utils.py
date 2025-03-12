from pathlib import Path
import os.path
import lyaforecast
import numpy as np
import mcfit
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline


def pk_to_xi(k, Pk, ell=0, extrap=True):
    xi = mcfit.P2xi(k, l=ell, lowring=True)
    rr, CF = xi(Pk, extrap=extrap)
    return InterpolatedUnivariateSpline(rr, CF)


def xi_to_pk(r, xi, ell=0, extrap=False):
    P = mcfit.xi2P(r, l=ell, lowring=True)
    kk, Pk = P(xi, extrap=extrap)
    return InterpolatedUnivariateSpline(kk, Pk)

def f_xiSB(r, am3, am2, am1, a0, a1):
    par = [am3, am2, am1, a0, a1]
    model = np.zeros((len(par), r.size))
    tw = r != 0.
    model[0, tw] = par[0] / r[tw]**3
    model[1, tw] = par[1] / r[tw]**2
    model[2, tw] = par[2] / r[tw]**1
    model[3, tw] = par[3]
    model[4, :] = par[4] * r
    model = np.array(model)
    return model.sum(axis=0)

def get_pk_smooth(cosmo_results,k,pk):
    pars = cosmo_results.Params
    pars2 = cosmo_results.get_derived_params()
    coef_Planck2015 = (pars.H0/ 67.31) * (pars2['rdrag'] / 147.334271564563)
    sb1_rmin = 50. * coef_Planck2015
    sb1_rmax = 82. * coef_Planck2015
    sb2_rmin = 150. * coef_Planck2015
    sb2_rmax = 190. * coef_Planck2015
    xi = pk_to_xi(k, pk)
    r = np.logspace(-7., 3.5, 10000)
    xi = xi(r)

    w = ((r >= sb1_rmin) & (r < sb1_rmax)) | ((r >= sb2_rmin) & (r < sb2_rmax))
    sigma = 0.1 * np.ones(xi.size)
    sigma[(r >= sb1_rmin - 2.) & (r < sb1_rmin + 2.)] = 1.e-6
    sigma[(r >= sb2_rmax - 2.) & (r < sb2_rmax + 2.)] = 1.e-6
    popt, pcov = curve_fit(f_xiSB, r[w], xi[w], sigma=sigma[w])

    model = f_xiSB(r, *popt)
    xi_smooth = xi.copy()
    ww = (r >= sb1_rmin) & (r < sb2_rmax)
    xi_smooth[ww] = model[ww]

    extrap = True
    pk_smooth = xi_to_pk(r, xi_smooth, extrap=extrap)
    pk_smooth = pk_smooth(k)
    pk_smooth *= pk[-1] / pk_smooth[-1]

    return pk_smooth

def check_file(input_path):
    """ Verify a file exists, if not raise error

    Parameters
    ----------
    path : string
        Input path. Only absolute.

    """
    # First check if it's an absolute path
    if input_path.is_file():
        return input_path
    else:
        raise RuntimeError('The path does not exist: ', input_path)

def get_file(path):
    """ Find files on the system.

    Checks if it's an absolute or relative path inside LyaCast

    Parameters
    ----------
    path : string
        Input path. Can be absolute or relative to lyacast
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_file():
        return input_path
    # Get the lyacast path and check inside lyacast (this returns LyaCast/lyacast)
    lyacast_path = Path(os.path.dirname(lyaforecast.__file__))

    # Check if it's a resource
    resource = lyacast_path / 'resources' / input_path
    if resource.is_file():
        return resource
    
    # Check if it's a data source
    data = lyacast_path / 'resources/data' / input_path
    if data.is_file():
        return data
    
    # Check if it's a default config
    default_cfg = lyacast_path / 'resources/default_configs' / input_path
    if default_cfg.is_file():
        return default_cfg
    
    # Check if it's a camb config
    camb_cfg = lyacast_path / 'resources/camb_configs' / input_path
    if camb_cfg.is_file():
        return camb_cfg

    raise RuntimeError('The path does not exist: ', input_path, 'or', resource)

def get_dir(path):
    """ Find directory on the system.

    Checks if it's an absolute or relative path inside LyaCast

    Parameters
    ----------
    path : string
        Input path. Can be absolute or relative to lyacast
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_dir():
        return input_path

    # Get the lyacast path and check inside lyacast (this returns LyaCast/lyacast)
    lyacast_path = Path(os.path.dirname(lyaforecast.__file__))

    # Check if it's a resource
    resource = lyacast_path / 'resources' / input_path
    if resource.is_dir():
        return resource
    
    # Check if it's a data source (folder)
    data = lyacast_path / 'resources/data' / input_path
    if data.is_dir():
        return data

    raise RuntimeError('The directory does not exist: ', input_path)
