"""Plot power spectra and parameter error bars as functions of survey properties."""
import matplotlib.pyplot as plt
class Plots:
    def __init__(self,config,survey,data) -> None:
        self.plot_distances = config['control'].getboolean('plot distances')
        #We will require in some cases that n_redshift_bins > 1.
        n_redshift_bins = config['power spectrum'].getint('num z bins')
        #check magnitude band
        self.survey = survey
        self.data = data


    def plot_da_h_m(self):
        ap = self.data['ap_err_m'] * 100
        at = self.data['at_err_m'] * 100
        band = self.survey.band
        mags = self.data['magnitudes'][band]

        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.plot(mags,ap,label=fr'$\alpha_\parallel$')
            ax.plot(mags,at,label=fr'$\alpha_\perp$',linestyle='dashed')
            ax.set_xlabel(fr'${band}_{{max}}$')
            ax.set_ylabel(f'% error')
            ax.legend()
            ax.grid()

            self.fig = fig

    def _make_style(self,style='seaborn-1'):
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

        # Apply both the base Seaborn style and customizations
        return plt.style.context(base_style), plt.rc_context(custom_rc)