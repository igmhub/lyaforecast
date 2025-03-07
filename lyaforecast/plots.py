"""Plot power spectra and parameter error bars as functions of survey properties."""
import matplotlib.pyplot as plt
import numpy as np

class Plots:
    def __init__(self,forecast=None,data=None):
        """
        Initialize Plots instance with configuration and survey details.

        Parameters
        ----------
        survey : Survey
            Instance of Survey class containing survey properties.
        """
        
        self._forecast = forecast
        if forecast is not None:
            self._survey = forecast.survey
            self._covariance = forecast.covariance

        if data is not None:
            self._data = data
            self.p3d = self._data['p3d_z']
            self.var_p3d = self._data['p3d_var_z']

    def plot_da_h_m(self):
        ap = self._data['ap_err_m'] * 100
        at = self._data['at_err_m'] * 100
        band = self._survey.band
        mags = self._data['magnitudes'][band]

        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            #ax.plot(mags[1:],ap[1:]/ap[:-1],label=fr'$\alpha_\parallel$')
            ax.plot(mags,ap,label=fr'$\alpha_\parallel$')
            ax.plot(mags,at,label=fr'$\alpha_\perp$',linestyle='dashed')
            ax.set_xlabel(fr'${band}_{{max}}$')
            ax.set_ylabel(f'% error')
            ax.set_xlim(19,23)
            ax.set_ylim(0,5)
            ax.legend()
            ax.grid()
            ax.set_yscale('linear')
            self.fig = fig

    def plot_da_h_z(self):
        ap = self._data['ap_err_z'] * 100
        at = self._data['at_err_z'] * 100
        zs = self._data['redshifts']

        zs_desi_sci = [2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55]
        ap_desi_sci = [1.99,2.11,2.26,2.47,2.76,3.18,3.70,4.57,6.19,8.89]
        at_desi_sci = [1.95,2.18,2.46,2.86,3.40,4.21,5.29,7.10,10.46,15.91]

        zs_desi_sv = np.array([2.15,2.25,2.35,2.45,2.55,2.65,2.75,2.85,2.95,3.05,3.15,3.25,3.35,3.45])
        ap_desi_sv = [2.16,2.24,2.36,2.52,2.77,3.11,3.5,4.05,4.71,5.51,6.78,8.41,11.1,14.8]
        at_desi_sv = [2.02,2.14,2.33,2.56,2.9,3.38,3.95,4.69,5.59,6.73,8.47,10.73,14.48,19.92]

        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            #ax.plot(mags[1:],ap[1:]/ap[:-1],label=fr'$\alpha_\parallel$')
            ax.plot(zs,ap,label=fr'$\alpha_\parallel forecast$')
            ax.plot(zs,at,label=fr'$\alpha_\perp forecast$',linestyle='dashed')
            ax.scatter(zs_desi_sv,ap_desi_sv,
                       label=fr'$\alpha_\parallel$ DESI SV',color='red',
                       marker='.',alpha=0.5)
            ax.scatter(zs_desi_sv,at_desi_sv,
                       label=fr'$\alpha_\perp$ DESI SV',color='blue',
                       marker='x',alpha=0.5)
            ax.set_xlabel(fr'$z$')
            ax.set_ylabel(f'% error')
            #ax.set_xlim(19,23)
            #ax.set_ylim(0,5)
            ax.legend()
            ax.grid()
            ax.set_yscale('linear')
            self.fig = fig

    def plot_pk_z(self,p3d_z_k_mu,var_z_k_mu,z_bins):
        mu_ind = 5
        mu_val = round(self._covariance.mu[mu_ind],2)
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.set_xlabel(fr'k')
            ax.set_ylabel(fr'$kP(k,\mu={mu_val})/\pi$')
            #ax.set_xlim(0.03,0.3)
            ax.autoscale(axis='y')
            ax.grid()
            ax.set_yscale('linear')
            for j,p_k_mu in enumerate(p3d_z_k_mu[:-5,:,:]):
                ax.errorbar(self._covariance.k,
                        self._covariance.k * p_k_mu[:,mu_ind] / np.pi,
                        yerr= self._covariance.k * var_z_k_mu[j,:,mu_ind] / np.pi,
                        label=f'z = {z_bins[j]}',
                        alpha=0.3)
            ax.legend()
            
            self.fig = fig

    def plot_var_pk_z(self,var_z_k_mu,z_bins):
        mu_ind = 5
        mu_val = round(self._covariance.mu[mu_ind],2)
        with self._make_style()[0], self._make_style()[1]:
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.set_xlabel(fr'k [$h$/Mpc]')
            ax.set_ylabel(fr'$\sigma_P(k,\mu = {mu_val})$')
            ax.grid()
            ax.set_yscale('log')
            for j,var_k_mu in enumerate(var_z_k_mu[:-5,:,:]):
                ax.plot(self._covariance.k,
                        var_k_mu[:,mu_ind],
                        label=f'z = {z_bins[j]}',
                        alpha=0.3)
            ax.legend()
            
            self.fig = fig

    def plot_n_pk_z(self,zbs,n_p3d_z_lya,n_p3d_z_qso):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.set_xlabel(fr'z')
            ax.set_ylabel(r'$\overline{n}P(0.14,0.6)$')
            ax.grid()
            ax.set_yscale('linear')
            ax.plot(zbs,n_p3d_z_lya,label='lya')
            ax.plot(zbs,n_p3d_z_lya,label='qso')
            ax.legend()
            
            self.fig = fig

    def plot_var_p3d_m(self):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            ax.set_xlabel(fr'k')
            ax.set_ylabel(f'var[P(k)]')
            ax.set_xlim(0.03,0.3)
            ax.grid()
            ax.set_yscale('log')

            for key in self.p3d:
                ax.plot(self._covariance.k,
                        self.var_p3d[key].T[0], alpha = 0.5,ls='dashed',
                        label=fr'$z = {key}, m_{{\rm max}} = {self._survey.maglist[0]}$')
                ax.plot(self._covariance.k,
                        self.var_p3d[key].T[-1], alpha = 0.5,ls='dashed',
                        label=fr'$z = {key}, m_{{\rm max}} = {self._survey.maglist[-1]}$')

            ax.legend()
            
            self.fig = fig

    def plot_qso_lf(self):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,2,figsize=(15,6))

            # get limits, for now fix this
            m = np.linspace(0,23.5,100)
            qlf = self._survey.get_qso_lum_func
            # plot QL at different z
            ax[0].plot(m,qlf(2.0,m),label='z=2.0')
            ax[0].plot(m,qlf(2.5,m),label='z=2.5')
            ax[0].plot(m,qlf(3.0,m),label='z=3.0')
            ax[0].plot(m,qlf(3.5,m),label='z=3.5')

            ax[0].legend(loc=2,fontsize=15)
            ax[0].set_yscale('log')
            ax[0].set_xlim(16.5,23.5)
            ax[0].set_ylim(1e-3,)
            ax[0].set_xlabel('r mag',fontsize=15)
            ax[0].set_ylabel(r'dN/dzdm$\rm{ddeg}^2$',fontsize=15)
            ax[0].grid()

            #dn_dzddeg vs z

            desi_sv_z = np.array([1.65,1.75,1.85,1.95,2.05,2.15,2.25,2.35,2.45,
                                  2.55,2.65,2.75,2.85,2.95,3.05,3.15,3.25,3.35,3.45])
            desi_sv_points = np.array([12.1,11.8,11.1,10.6,9.5,8.8,8,7.2,
                                       6.2,5.3,4.4,3.6,3.3,2.6,2.2,1.7,1.4,1.1,0.7])
            
            dz = desi_sv_z[1] - desi_sv_z[0]
            qlf_dzddeg = np.zeros(desi_sv_z.size)
            for k,zi in enumerate(desi_sv_z):
                qlf_dzddeg_i = np.sum(qlf(zi,m) * (m[1] - m[0]))
                qlf_dzddeg[k] = qlf_dzddeg_i

            ax[1].plot(desi_sv_z,qlf_dzddeg,label=r'Forecast')
            #ax[1].plot(z,qlf_dzddeg * dz,label=r'Forecast')

            desi_sci_z = np.array([1.96,2.12,2.28,2.43,2.59,2.75,2.91,
                          3.07,3.23,3.39,3.55,3.70,3.86,4.02])
            desi_sci_points = np.array([82,69,53,43,37,31,26,21,16,13,9,7,5,3])



            ax[1].scatter(desi_sci_z,desi_sci_points,color='black',
                       marker='x',alpha=0.5,label=r'DESI sci')

            ax[1].scatter(desi_sv_z,desi_sv_points/(0.1),color='blue',
                       marker='x',alpha=0.7,label=r'DESI SV')
        
            ax[1].set_xlabel('z',fontsize=15)
            #ax[1].set_ylabel(r'$dN/dz$\rm{ddeg}^2$',fontsize=15)
            ax[1].set_ylabel(r'$dn/dz$',fontsize=15)
            ax[1].grid()
            ax[1].legend()

            self.fig = fig

    def plot_neff(self,neff):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,2,figsize=(15,6))
            for i,zbc in enumerate(self._survey.z_bin_centres):
                ax[0].plot(self._survey.maglist,neff[i],label=f'z={zbc}')

            ax[0].legend(loc=2,fontsize=15)
            ax[0].set_xlim(21,23)
            ax[0].set_xlabel(r'$r_{\rm max}$',fontsize=15)
            ax[0].set_ylabel(r'$\overline{n}_{\rm eff}[\rm (km/s)^{-3}]$',fontsize=15)
            ax[0].set_yscale('linear')
            
            ax[1].plot(self._survey.z_bin_centres,neff[:,-1])
            ax[1].set_xlabel(r'$z$',fontsize=15)

            self.fig = fig

    def plot_survey_volume(self,volume,z_bin_centres):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,1,figsize=(8,6))
            ax.plot(z_bin_centres,volume/1e9,label='Forecast')

            z_desi_sv = [1.65,1.75,1.85,1.95,2.05]
            points_desi_sv = [0.17,0.16,0.14,0.13,0.1]

            ax.scatter(z_desi_sv,points_desi_sv,
                       alpha=0.5,color='black',label='DESI SV')

            ax.set_xlabel(r'$z$',fontsize=15)
            ax.set_ylabel(r'$V_{\rm eff} (h^{-1}\mathrm{Gpc})^{3}$',fontsize=15)
            ax.set_yscale('linear')
            ax.legend()

            self.fig = fig

    def plot_weights(self,weights,zbins):
        with self._make_style()[0], self._make_style()[1]: 
            fig,ax = plt.subplots(1,2,figsize=(15,6))
            for i,zbc in enumerate(self._survey.z_bin_centres):
                ax[0].plot(self._survey.maglist,weights[i],label=f'z={zbc}')

            ax[0].legend(loc=2,fontsize=15)
            ax[0].set_xlabel(r'$r$',fontsize=15)
            ax[0].set_ylabel(r'$w(m)$',fontsize=15)
            ax[0].set_yscale('linear')
            
            weights_z = weights[:,-1]
            ax[1].plot(zbins,weights_z)
            ax[1].set_xlabel(r'$z$',fontsize=15)
            ax[1].set_ylabel(r'$w(z)$',fontsize=15)

            self.fig = fig


    def plot_snr_per_ang(self):
        #this is inflexible at the moment - only designed to run with DESIQSO spectro.
        with self._make_style()[0], self._make_style()[1]:
            fig,ax = plt.subplots(1,1,figsize=(10,6))
            from_file = False
            if from_file:
                snr_mat = self._forecast.spectrograph._snr_mat
                mags = [19.25,19.75,20.25,20.75,21.25,21.75,22.25,22.75,23.25]
                lam = self._forecast.spectrograph._lambda_obs_m
                for i,mag in enumerate(mags):
                    # average over all redshift
                    snr = snr_mat[i].mean(axis=0)
                    ax.plot(lam,snr,label=f'r={mag}')
                    ax.set_ylim(1e-1)
                    ax.set_xlim(3500,6000)
            else:
                mags = [21,21.5,22,22.5,23]
                zqs = [2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75]
                lmax = 6000
                lmin = 3600
                nb = 100
                snr = np.zeros(nb)
                snr_z = np.zeros(nb)
                lam = np.linspace(lmin,lmax,nb)
                for mag in mags:
                    snr = 0
                    for zq in zqs:
                        for i,l in enumerate(lam):
                            snr_z[i] = self._forecast.spectrograph.get_snr_per_ang(mag,zq,l)
                        snr += snr_z/len(zqs)

                    ax.plot(lam,snr,label=f'r={mag}')

            ax.legend(loc='lower left',fontsize=15)
            ax.set_xlabel(r'SNR per $\AA$',fontsize=15)
            ax.set_ylabel(r'$\lambda[\AA]$',fontsize=15)
            ax.set_yscale('log')
        
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
