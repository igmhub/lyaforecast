#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots
import matplotlib.pyplot as plt
import numpy as np

def main():
    args = get_args()

    errors = []
    for cfg in args.config:
        forecast = Forecast(cfg)

        errors_i = forecast.run_bao_forecast()

        errors.append(errors_i)

        
    fig = plot_da_h_z(errors)

    fig.savefig(forecast.out_folder.joinpath('dap_dat_z.png'))

def plot_da_h_z(forecast_data):
    with _make_style()[0], _make_style()[1]: 
        fig,ax = plt.subplots(2,2,figsize=(25,18))

        colours = ['blue','green']
        labels = ['LBG','QSO']
        for i,data in enumerate(forecast_data):

            ap_lya = data['ap_err_lya_z'] * 100
            at_lya = data['at_err_lya_z'] * 100
            ap_tr = data['ap_err_tracer_z'] * 100
            at_tr = data['at_err_tracer_z'] * 100
            ap_cross = data['ap_err_cross_z'] * 100
            at_cross = data['at_err_cross_z'] * 100
            ap_comb = data['ap_err_comb_z'] * 100
            at_comb = data['at_err_comb_z'] * 100
            zs = data['redshifts']

            if i==1:
                # ax[0,0].plot(zs,ap_lya,alpha=0.5,color=colours[i],label=labels[i])
                # ax[0,0].plot(zs,at_lya,linestyle='dashed',alpha=0.5,color=colours[i])
                ax[1,0].plot(zs,ap_cross,alpha=0.5,color='red',label=r'Ly$\alpha$(QSO)xLBG')
                ax[1,0].plot(zs,at_cross,linestyle='dashed',alpha=0.5,color='red')
                continue

            ax[0,0].plot(zs,ap_lya,alpha=0.5,color=colours[i],label=r'$\alpha_\parallel$')
            ax[0,0].plot(zs,at_lya,linestyle='dashed',alpha=0.5,color=colours[i],label=r'$\alpha_\perp$')

            ax[0,1].plot(zs,ap_tr,alpha=0.5,color=colours[i])
            ax[0,1].plot(zs,at_tr,linestyle='dashed',alpha=0.5,color=colours[i])

            ax[1,0].plot(zs,ap_cross,alpha=0.5,color=colours[i])
            ax[1,0].plot(zs,at_cross,linestyle='dashed',alpha=0.5,color=colours[i])

            #combined lya auto and lya-qso cross.

            ax[1,1].plot(zs,ap_comb,alpha=0.5,color=colours[i])
            ax[1,1].plot(zs,at_comb,linestyle='dashed',alpha=0.5,color=colours[i])


            ax[1,0].set_xlabel(fr'$z$')
            ax[1,1].set_xlabel(fr'$z$')
            ax[0,0].set_ylabel(f'% error')
            ax[1,0].set_ylabel(f'% error')

        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend(fontsize=22)
        ax[1,1].legend()

        ax[0,0].grid()
        ax[0,1].grid()
        ax[1,0].grid()
        ax[1,1].grid()

        ax[0,0].set_ylim(1.5,20)
        ax[0,1].set_ylim(1.5,20)
        ax[1,0].set_ylim(1.5,20)
        ax[1,1].set_ylim(1.5,20)

        ax[0,0].set_title(r'Ly$\alpha$ power')
        ax[0,1].set_title(r'Tracer power')
        ax[1,0].set_title(r'Ly$\alpha$-tracer cross power')
        ax[1,1].set_title(r'Ly$\alpha$ auto + Ly$\alpha$-tracer cross')

        return fig


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
            "axes.labelsize": 28,
            "axes.titlesize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 26,
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
    "axes.labelsize": 18,  # Axis label font size
    "axes.titlesize": 12,  # Title font size
    "xtick.labelsize": 10,  # Tick label size
    "ytick.labelsize": 10,
    "legend.fontsize": 18,
    
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
                        nargs = '+',
                        help='Config file')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()