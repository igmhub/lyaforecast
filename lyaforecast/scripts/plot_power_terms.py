#!/usr/bin/env python
"""Plot terms that contribute to P_T (P3D, aliasing, effective noise) as a function of k."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lyaforecast import Forecast

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Run Forecast.')

    parser.add_argument('--config', '-i',
                        type=str, 
                        default=None,
                        nargs='+', 
                        help='Config file')
    
    return parser.parse_args()

def _make_style(style='seaborn-1'):
    """Apply a Seaborn style with additional customizations."""
    import matplotlib.pyplot as plt
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

# Main execution block
if __name__ == '__main__':
    args = get_args()

    with _make_style()[0], _make_style()[1]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for j,cfg in enumerate(args.config):
        forecast = Forecast(cfg)
                
        z_bin_edges, zc = forecast._get_z_bins()

        lmin = forecast.cosmo.LYA_REST * (1 + z_bin_edges[0,0])
        lmax = forecast.cosmo.LYA_REST * (1 + z_bin_edges[1,0])

        forecast.covariance(lmin, lmax)
        forecast.covariance.compute_eff_density_and_noise()

        p_n_eff = forecast.covariance._effective_noise_power[-1]
        
        k_hmpc = np.linspace(1e-2,0.5,500)
        mus = [0.00001,0.5,1]
        linestyles = ['solid','dashed','dotted']

        for i,mu in enumerate(mus):

            kp_hmpc = k_hmpc * mu
            kt_hmpc = k_hmpc * np.sqrt(1-mu**2)

            kp_skm = kp_hmpc / forecast.cosmo.velocity_from_distance(zc)
            kt_deg = kt_hmpc * forecast.cosmo.distance_from_degrees(zc)

            p3d = forecast.power_spec.compute_p3d_kms(zc,kt_deg,kp_skm,
                                                        120,
                                                        60,'lya')
            
            aliasing = forecast.covariance.compute_aliasing(zc,kt_deg,kp_skm)

            ax.plot(k_hmpc, p3d, color='blue', label=r'$P_F$', alpha=0.5, linestyle=linestyles[i])
            ax.plot(k_hmpc, aliasing, color='green', label=r'$P_w^\perp P_F^\mathrm{1D}$', alpha=0.5,linestyle=linestyles[i])
            ax.plot(k_hmpc, p_n_eff * np.ones(500), color='purple', label=r'$P_N^\mathrm{eff}$', alpha=0.5,linestyle=linestyles[i])

            if i == 0:
                ax.legend(fontsize=20)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'Power $(k,\mu,z=2.75)$')
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    

    fig.savefig(forecast.out_folder.joinpath('power_terms.png'))
