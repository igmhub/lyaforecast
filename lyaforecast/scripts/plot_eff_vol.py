#!/usr/bin/env python
"""Plot effective densities of measurements. For Lya it's the 2D effective density defined by McQuinn and White 2011. For tracers it's just the 3D density."""

import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots

def main():
    raise ValueError

def from_script(args):
    import numpy as np 
    import matplotlib.pyplot as plt

    with _make_style()[0], _make_style()[1]: 
        fig,ax = plt.subplots(1,2,figsize=(20,6))
        linestyles=['solid','dashed']

    for j,cfg in enumerate(args.config):
        forecast = Forecast(cfg)

        if forecast._lya_tracer == 'qso':
                lab_lya = r'Ly$\alpha$(QSO)$^\mathrm{auto}$'
                if forecast.survey.area_deg2 == 14000:
                    lab_lya = r'Ly$\alpha$(QSO)$^\mathrm{auto}$ (DESI)'
        elif forecast._lya_tracer == 'lbg':
                lab_lya = r'Ly$\alpha$(LBG)$^\mathrm{auto}$'

        if forecast._tracer == 'qso':
                lab_tr = r'QSO$^\mathrm{auto}$'
        elif forecast._tracer == 'lbg':
                lab_tr = r'LBG$^\mathrm{auto}$'
        elif forecast._tracer == 'lae':
                lab_tr = r'LAE$^\mathrm{auto}$'


        vol_eff = forecast.compute_eff_vol()

        lim = 18
        ulim = 26
        ylim = 1e-3
        yulim = 1e-1
        w = forecast.survey.maglist > lim
        w &= forecast.survey.maglist < ulim

        if forecast._lya_tracer == 'lbg':
            w = forecast.survey.maglist > 22
            w &= forecast.survey.maglist < ulim

        colours = ['blue','green','red']
        for i,zbc in enumerate(forecast.survey.z_bin_centres):

            print(f'Plotting effective volume for z={zbc}')

            ne_i = (vol_eff['lya'][i][w])
            n_i_tr = (vol_eff['tracer'][i][w])

            ax[0].plot(forecast.survey.maglist[w],ne_i,alpha=0.8,color=colours[j],label=lab_lya,linestyle=linestyles[i])
            ax[1].plot(forecast.survey.maglist[w],n_i_tr,alpha=0.8,color=colours[j],label=lab_tr,linestyle=linestyles[i])

            if i == 0:
                ax[0].legend(loc='upper left',fontsize=18)
                ax[1].legend(loc=2,fontsize=18)

        ax[0].set_xlim(20,)
        ax[0].set_ylim(ylim,)
        ax[0].set_xlabel(r'$r_{\rm max}$',fontsize=18)
        ax[0].set_ylabel(r'${\rm V}_{\rm eff,\alpha}/V_s$',fontsize=18)
        ax[0].set_yscale('log')

        
        ax[1].set_xlim(20,)
        ax[0].set_ylim(ylim,)
        ax[1].set_xlabel(r'$r_{\rm max}$',fontsize=18)
        ax[1].set_ylabel(r'${\rm V}_{\rm eff,tracer}/V_s$',fontsize=18)
        ax[1].set_yscale('log')

        ax[0].grid(which='both')
        ax[1].grid(which='both')

    fig.savefig(forecast.out_folder.joinpath('vol_eff.png'))




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
