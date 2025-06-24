#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots

def main():
    args = get_args()

    forecast = Forecast(args.config)

    plotter = Plots(forecast=forecast)

    neff = forecast.compute_neff()

    plotter.plot_neff(neff)

    plotter.fig.savefig(forecast.out_folder.joinpath('neff.png'))

def from_script(args):
    import numpy as np 
    import matplotlib.pyplot as plt

    with _make_style()[0], _make_style()[1]: 
        fig,ax = plt.subplots(1,2,figsize=(20,6))
        linestyles=['solid','dashed']

    for j,cfg in enumerate(args.config):
        forecast = Forecast(cfg)
        neff = forecast.compute_neff()


        lim = 22.5
        ulim = 26
        w = forecast.survey.maglist > lim
        for i,zbc in enumerate(forecast.survey.z_bin_centres):
            ne_i = neff['lya'][i][w] #/ forecast.cosmo.velocity_from_distance(zbc)**2 
                           # * neff['bin_lengths'][i])
            n_i_tr = neff['tracer'][i][w] #/ forecast.cosmo.distance_from_degrees(zbc)**2 
                           # * neff['bin_lengths'][i])

            ax[0].plot(forecast.survey.maglist[w],ne_i,label=f'z={zbc}',linestyle=linestyles[j])
            ax[1].plot(forecast.survey.maglist[w],n_i_tr,label=f'z={zbc}',linestyle=linestyles[j])

        ax[0].legend(loc=2,fontsize=15)
        ax[0].set_xlim(lim,ulim)
        ax[0].set_xlabel(r'$r_{\rm max}$',fontsize=15)
        ax[0].set_ylabel(r'$\overline{n}_{\rm eff}[\rm (km/s)^{-3}]$',fontsize=15)
        ax[0].set_yscale('log')

        ax[1].legend(loc=2,fontsize=15)
        ax[1].set_xlim(lim,ulim)
        ax[1].set_xlabel(r'$r_{\rm max}$',fontsize=15)
        ax[1].set_ylabel(r'$\overline{n}_{\rm tracer}[\rm deg^{-2}(km/s)^{-1}]$',fontsize=15)
        ax[1].set_yscale('log')

        ax[0].grid(which='both')
        ax[1].grid(which='both')

    fig.savefig(forecast.out_folder.joinpath('neff.png'))




def _make_style(style='seaborn-1'):
        import matplotlib.pyplot as plt
        """Apply a Seaborn style with additional customizations."""
        # Define built-in styles (Seaborn, ggplot, etc.)
        base_styles = {
            "seaborn-1": "seaborn-v0_8-notebook",  # Seaborn notebook style
            "seaborn-2": "seaborn-darkgrid",  # Seaborn dark grid style
            "ggplot": "ggplot",  # ggplot style
            "classic": "classic",  # Matplotlib's classic style
        }

        set1_colours = plt.cm.Set1.colors
        # Select the base style (default: "style1")
        base_style = base_styles.get(style, "seaborn-v0_8-notebook")

        # Additional font and size customizations
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
    "text.usetex": False,  # Disable LaTeX (since you can't install it)
    "mathtext.fontset": "cm",  # Use Computer Modern for math
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    
    "font.family": "serif",  # Use serif font (JCAP style)
    "font.size": 10,  # General font size
    "axes.labelsize": 12,  # Axis label font size
    "axes.titlesize": 12,  # Title font size
    "xtick.labelsize": 10,  # Tick label size
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    
    "axes.linewidth": 1,  # Thicker axis for clarity
    "xtick.major.size": 5,  # Major tick size
    "ytick.major.size": 5,
    "xtick.minor.size": 3,  # Minor tick size
    "ytick.minor.size": 3,
    "xtick.major.width": 1,  # Major tick width
    "ytick.major.width": 1,
    
    "lines.linewidth": 1.5,  # Thicker lines for better visibility
    "lines.markersize": 6,  # Slightly larger markers
    
    "figure.figsize": (7, 3.5),  # Double-column width (~7 in) with good aspect ratio
    "figure.dpi": 300,  # High resolution
    "savefig.dpi": 300,  # Save high-resolution figures
    "savefig.format": "pdf",  # Save as vector PDF
    "savefig.bbox": "tight",  # Remove extra whitespace
})
        # Apply both the base Seaborn style and customizations
        return plt.style.context(base_style), plt.rc_context(custom_rc)
    
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Run Forecast.')

    parser.add_argument('--config', '-i',
                        type=str, 
                        default=None,
                        nargs='+', 
                        help='Config file')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    from_script(args)
