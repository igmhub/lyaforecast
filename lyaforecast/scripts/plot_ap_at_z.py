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

        colours = ['blue','green','red','cyan']
        labels = ['LBG','QSO','LAE']
        labels_detailed = np.array([[r'Ly$\alpha$(LBG)$^\mathrm{auto}$',r'Ly$\alpha$(QSO)$^\mathrm{auto}$','',''],
                           [r'LBG$^\mathrm{auto}$','',r'LAE$^\mathrm{auto}$',''],
                           [r'Ly$\alpha$(LBG)$^\mathrm{cross}$',r'Ly$\alpha$(QSO)xLBG',r'Ly$\alpha$(LBG)xLAE',r'Ly$\alpha$(QSO)xLAE'],
                           [r'Ly$\alpha$(LBG)$^\mathrm{auto+cross}$','','','']])
        
        alphas = np.array([[0.5,0.5,0,0],
                  [0.5,0,0.5,0],
                  [0.5,0.5,0.5,0.5],
                  [0.5,0,0,0]])
        
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

            ax[0,0].plot(zs,ap_lya,alpha=alphas[0,i],color=colours[i],label=labels_detailed[0,i])#label=r'$\alpha_\parallel$')
            ax[0,0].plot(zs,at_lya,linestyle='dashed',alpha=alphas[0,i],color=colours[i])#label=r'$\alpha_\perp$')

            ax[0,1].plot(zs,ap_tr,alpha=alphas[1,i],color=colours[i],label=labels_detailed[1,i])
            ax[0,1].plot(zs,at_tr,linestyle='dashed',alpha=alphas[1,i],color=colours[i])

            ax[1,0].plot(zs,ap_cross,alpha=alphas[2,i],color=colours[i],label=labels_detailed[2,i])
            ax[1,0].plot(zs,at_cross,linestyle='dashed',alpha=alphas[2,i],color=colours[i])

            ax[1,1].plot(zs,ap_comb,alpha=alphas[3,i],color=colours[i],label=labels_detailed[3,i])
            ax[1,1].plot(zs,at_comb,linestyle='dashed',alpha=alphas[3,i],color=colours[i])


        
   
            ax[1,0].set_xlabel(fr'$z$')
            ax[1,1].set_xlabel(fr'$z$')
            ax[0,0].set_ylabel(fr'% error ($dz=0.1$)')
            ax[1,0].set_ylabel(fr'% error ($dz=0.1$)')

        #cosmic variance limits
        cosmic_variance_desi_ii_lya_at = [0.77351024, 0.77110882, 0.76960042, 0.76885945, 0.76878171,
                                                0.76928012, 0.77028135, 0.77172334, 0.77355316, 0.77572549,
                                                0.77820132, 0.78094688, 0.78393284, 0.78713362, 0.79052678]
        cosmic_variance_desi_ii_lya_ap = [1.75440014, 1.74898995, 1.74560357, 1.7439563 , 1.74381197,
                                            1.74497314, 1.74727364, 1.75057275, 1.75475054, 1.75970431,
                                            1.7653456 , 1.77159792, 1.77839478, 1.78567819, 1.79339734]
        cosmic_variance_desi_lya_ap = [1.04187624, 1.0386703 , 1.03666593, 1.03569407, 1.03561449,
                                            1.03630996, 1.03768182, 1.03964652, 1.04213286, 1.04507985,
                                            1.04843498, 1.05215283, 1.05619393, 1.06052385, 1.06511242]
        cosmic_variance_desi_lya_at = [0.45773395, 0.45631349, 0.45542146, 0.45498354, 0.45493808,
                                            0.45523354, 0.45582653, 0.45668032, 0.45776361, 0.45904956,
                                            0.4605151 , 0.46214024, 0.46390763, 0.46580213, 0.46781047]

        ax[0,0].plot(zs,cosmic_variance_desi_ii_lya_ap,alpha=0.5,label=r'$\sigma^2_{\mathrm{cosmic},\mathrm{DESI}-\mathrm{II}}$',
                        color='black')#,label=r'$\sigma_\mathrm{cosmic,LBG}$')
        ax[0,0].plot(zs,cosmic_variance_desi_ii_lya_at,#label=r'$\sigma_\mathrm{cosmic,LBG}$',
                        linestyle='dashed',alpha=0.5,color='black')
        ax[0,0].plot(zs,cosmic_variance_desi_lya_ap,alpha=0.5,
                        color='darkblue',label=r'$\sigma^2_\mathrm{cosmic,DESI}$')
        ax[0,0].plot(zs,cosmic_variance_desi_lya_at,#label=r'$\sigma_\mathrm{cosmic,QSO}$',
                        linestyle='dashed',alpha=0.5,color='darkblue')

        ax[0,0].legend(loc='upper left')
        ax[0,1].legend()
        ax[1,0].legend(fontsize=22)
        ax[1,1].legend()

        ax[0,0].grid(which='both')
        ax[0,1].grid(which='both')
        ax[1,0].grid(which='both')
        ax[1,1].grid(which='both')

        #ax[0,0].text(2.09,1.95,r'$\sigma^2_\mathrm{cosmic,LBG}$',fontsize=18)
       # ax[0,0].text(2.09,cosmic_variance_desi_ii_lya_LBG_at[0])
        #ax[0,0].text(2.09,1.20,r'$\sigma^2_\mathrm{cosmic,QSO}$',fontsize=18)
       # ax[0,0].text(2.09,0.5,r'$\sigma_\mathrm{cosmic,QSO}$')
        
        ax[0,0].set_ylim(0,20)
        ax[0,1].set_ylim(0,20)
        ax[1,0].set_ylim(0,20)
        ax[1,1].set_ylim(0,20)


        ax[0,0].set_yscale('linear')
        ax[0,1].set_yscale('linear')
        ax[1,0].set_yscale('linear')
        ax[1,1].set_yscale('linear')


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