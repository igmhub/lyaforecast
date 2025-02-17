"""Plot power spectra and parameter error bars as functions of survey properties."""
import matplotlib.pyplot as plt
import numpy as np
class Plots:
    def __init__(self, config,survey):
        """
        Initialize Plots instance with configuration and survey details.

        Parameters
        ----------
        config : configparser.ConfigParser
            Configuration file for Lyman-alpha forest forecast.
        survey : Survey
            Instance of Survey class containing survey properties.
        """
        self.plot_bao = config['control'].getboolean('plot bao')
        self.plot_p3d = config['control'].getboolean('plot p3d')
        self.plot_p3d_var = config['control'].getboolean('plot p3d var')

        # We will require in some cases that n_redshift_bins > 1.
        self.n_redshift_bins = config['power spectrum'].getint('num z bins')
        # Check magnitude band
        self.survey = survey
        self.z_bin_centres = {}
        self.p3d = {}
        self.var_p3d = {}

    #May not need a call but do it for now.
    def __call__(self,data,covariance):
        """
        Parameters
        ----------
        data : dict
            Contains data used in plotting routines.
        """
        self._data = data
        self._covariance = covariance


    def plot_da_h_m(self):
        ap = self._data['ap_err_m'] * 100
        at = self._data['at_err_m'] * 100
        band = self.survey.band
        mags = self._data['magnitudes'][band]

        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.plot(mags,ap,label=fr'$\alpha_\parallel$')
            ax.plot(mags,at,label=fr'$\alpha_\perp$',linestyle='dashed')
            ax.set_xlabel(fr'${band}_{{max}}$')
            ax.set_ylabel(f'% error')
            ax.legend()
            ax.grid()
            ax.set_yscale('linear')
            self.fig = fig

    def plot_p3d_z(self):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.set_xlabel(fr'k')
            ax.set_ylabel(fr'$kP(k)\pi$')
            ax.set_xlim(0.03,0.3)
            ax.autoscale(axis='y')
            ax.grid()
            ax.set_yscale('log')

            for key in self.p3d:
                ax.errorbar(self._covariance.k,
                        self._covariance.k * self.p3d[key] / np.pi,yerr=0,#self.var_p3d[key],
                        label=f'z = {key}')
            
            ax.legend()
            
            self.fig = fig

    def plot_var_p3d_z(self):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.set_xlabel(fr'k')
            ax.set_ylabel(f'var[P(k)]')
            ax.grid()
            ax.set_yscale('log')

            for key in self.p3d:
                ax.plot(self._covariance.k,
                        self.var_p3d[key],
                        label=f'z = {key}')
                
            ax.legend()
            
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