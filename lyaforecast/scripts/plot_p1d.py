#!/usr/bin/env python
"""Plot effective densities of measurements. For Lya it's the 2D effective density defined by McQuinn and White 2011. For tracers it's just the 3D density."""

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

        forecast = Forecast(args.config)

        z = 2.5
        k_min_hmpc = 0
        k_max_hmpc = 0.5
        num_k_bins = 500
        k_hmpc = np.linspace(k_min_hmpc, k_max_hmpc, num_k_bins)

        k_skm = k_hmpc / forecast.cosmo.velocity_from_distance(z)

        breakpoint()
        p1d = forecast.power_spec.compute_p1d_palanque2013(z, k_skm)

        ax.plot(k_skm, p1d, color='blue', alpha=0.5)
        ax.set_ylabel('P1D [km/s]')
        ax.set_xlabel(r'$k$ [s/km]')

        fig.savefig(forecast.out_folder.joinpath('p1d.png'))
