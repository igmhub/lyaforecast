#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots
import matplotlib.pyplot as plt

def main():
    args = get_args()

    with _make_style()[0], _make_style()[1]: 
        fig,ax = plt.subplots(1,2,figsize=(20,6))
        linestyles=['solid','dashed']

    for j,cfg in enumerate(args.config):
        forecast = Forecast(cfg)

        z_bin_centres,_,n_pk_z_lya,n_pk_z_tracer = forecast.compute_pk()
            
        ax[0].plot(z_bin_centres,n_pk_z_lya,linestyle=linestyles[j])
        ax[1].plot(z_bin_centres,n_pk_z_tracer,linestyle=linestyles[j])

    ax[0].set_xlabel(fr'z')
    ax[0].set_ylabel(r'$nP^{(\rm Ly\alpha)}(k=0.14,\mu=0.6)$')
    ax[0].grid()
    ax[0].set_yscale('linear')

    ax[1].set_xlabel(fr'z')
    ax[1].set_ylabel(r'$nP^{(\rm Tracer)}(k=0.14,\mu=0.6)$')
    ax[1].grid()
    ax[1].set_yscale('linear')
            
    fig.savefig(forecast.out_folder.joinpath('np_z.png'))

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
    main()


