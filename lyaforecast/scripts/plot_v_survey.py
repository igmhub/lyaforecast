#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots
import matplotlib.pyplot as plt
import numpy as np 


"""Compute survey volume (Mpc/h) as a function of redshift"""

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Run Forecast.')

    parser.add_argument('--config', '-i',
                        type=str, 
                        default=None, 
                        help='Config file')

    args = parser.parse_args()
    return args

def _make_style(style='seaborn-1'):
    """Apply a Seaborn style with additional customizations."""
    base_styles = {
        "seaborn-1": "seaborn-v0_8-notebook",
        "seaborn-2": "seaborn-darkgrid",
        "ggplot": "ggplot",
        "classic": "classic",
    }

    set1_colours = plt.cm.Set1.colors
    base_style = base_styles.get(style, "seaborn-v0_8-notebook")

    custom_rc = {
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 2,
        "grid.alpha": 0.5,
        "font.family": "serif",
        "legend.frameon": True,
        "axes.prop_cycle": plt.cycler(color=set1_colours),
    }

    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "figure.figsize": (7, 3.5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    })
    return plt.style.context(base_style), plt.rc_context(custom_rc)

if __name__ == "__main__":
    args = get_args()

    with _make_style()[0], _make_style()[1]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
    forecast = Forecast(args.config)
    
    z_bin_edges = forecast.survey.z_bin_edges
    zcentres = forecast.survey.z_bin_centres
    survey_volume = np.zeros(zcentres.size)
    for i, zc in enumerate(zcentres):

        lmin = forecast.cosmo.LYA_REST * (1 + z_bin_edges[i,0])
        lmax = forecast.cosmo.LYA_REST * (1 + z_bin_edges[i,1])

        forecast.covariance(lmin, lmax)
    
        survey_volume[i] = forecast.covariance.get_survey_volume()

    ax.plot(zcentres,survey_volume/1e6,color='blue',alpha=0.5)

    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.grid()
    ax.set_ylabel(r'Volume $[h^{-1}$Gpc]')
    ax.set_xlabel(r'$z$')
    
    fig.savefig(forecast.out_folder.joinpath('survey_volume_dz.png'))

