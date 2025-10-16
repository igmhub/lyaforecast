"""Control module for lyaforecast. Should be structured as follows: 
    -   we use Covariance class for each config, and store observed powers and INDIVIDUAL covariances in ?dictionaries?
    - Then, using the Fisher class, we compute the parameter measurements 

"""
import configparser
import numpy as np
import time 
from pathlib import Path

from lyaforecast.utils import get_file, get_pk_smooth
from lyaforecast.cosmoCAMB import CosmoCamb
from lyaforecast.spectrograph import Spectrograph
from lyaforecast.survey import Survey
from lyaforecast.covariance import Covariance
from lyaforecast.power_spectrum import PowerSpectrum
from lyaforecast.plots import Plots


class Forecast:
    """Main LyaForecast class.
    ...outline...
    """
    # _survey_properties = None
    # _spectro_properties = None

    def __init__(self, cfg_path):
        """
        Parameters
        ----------
        cfg_path : string
            Path to main.ini config file
        """
        init_start_time = time.time()
        # Read config files
        self.config = configparser.ConfigParser()
        self.config.optionxform = lambda option: option
        self.config.read(get_file(cfg_path))

        self.out_file = Path(self.config['output']['filename'])
        self.out_folder = self.out_file.parent

        # tracer types
        self._lya_tracer = self.config['lya forest'].get('tracer')
        self._tracer = self.config['tracer'].get('tracer')
        self._cross_tracer = 'lya_' + self._tracer

        # which power spectra to forecast
        self._auto_flag = self.config['control'].getboolean('lya auto')
        self._cross_flag = self.config['control'].getboolean('cross')
        self._tracer_auto_flag = self.config['control'].getboolean('tracer auto')

        print('Initialise forecast')
        #initialise cosmology
        self.cosmo = CosmoCamb(self.config['cosmo'].get('filename'),
                                self.config['cosmo'].getfloat('z_ref', None))

        #load survey instance
        self.survey = Survey(self.config)

        #load spectrograph instance
        self.spectrograph = Spectrograph(self.config, self.survey)

        #load power spectrum instance
        self.power_spec = PowerSpectrum(self.config, self.cosmo, self.spectrograph)

        #initialise covariance class (McDonald & Eisenstein (2007)),
        #  that stores info and methods to compute p3d and its variance
        self.covariance = Covariance(self.config, self.cosmo, self.survey, 
                                     self.spectrograph, self.power_spec)
        
        init_end_time = time.time()
        print(f"Forecast initialized in {init_end_time - init_start_time:.4f} seconds.")

    def compute_weights(self,compute_neff=False):
        """
        Compute weights as a function of magnitude and mean redshift
        """
        weights = {}
        neffs = {}

        weights_lya = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
        weights_tracer = np.zeros_like(weights_lya)
        eff_vol_z = np.zeros_like(weights_lya)

        bin_lengths = np.zeros(self.survey.num_z_bins)

        z_bin_edges,z_bin_centres = self._get_z_bins()

        neff_lya = np.zeros_like(z_bin_centres)
        neff_tracer = np.zeros_like(z_bin_centres)

        for iz, zc in enumerate(z_bin_centres):
            lmin = self.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
            lmax = self.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])

            self.covariance(lmin, lmax)
            self.covariance.compute_eff_density_and_noise()
            # self.covariance.compute_3d_power_variance(0.14, 0.6)
            self.covariance.compute_cross_power_variance(0.14, 0.6)

            weights_lya[iz] = self.covariance._w_lya
            weights_tracer[iz] = self.covariance._w_tracer 

            #taking only the max magnitude neff.
            neff_lya[iz] =  self.covariance.get_neff_2D_lya(0.14, 0.6)[-1]
            neff_tracer[iz] = self.covariance.weights.get_n_tracer()[-1]

            bin_lengths[iz] = self.covariance._get_redshift_depth()

        weights['lya'] = weights_lya
        weights['tracer'] = weights_tracer

        neffs['lya'] = neff_lya[-1]
        neffs['tracer'] = neff_tracer[-1]
        neffs['bin_lengths'] = bin_lengths
        
        if compute_neff:
            return neffs
        else:
            return z_bin_centres, weights, eff_vol_z
        

    def compute_eff_vol(self):
            
        eff_vols = {}
        eff_vol_lya = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
        eff_vol_tr = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))

        z_bin_edges,z_bin_centres = self._get_z_bins()

        for iz, zc in enumerate(z_bin_centres):
            lmin = self.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
            lmax = self.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])
            
            self.covariance(lmin, lmax)
            self.covariance.compute_eff_density_and_noise()

            eff_vol_lya[iz] = self.covariance._compute_lya_eff_vol(0.14, 0.6) / self.covariance.get_survey_volume()
            eff_vol_tr[iz] = self.covariance._compute_tracer_eff_vol(0.14, 0.6) / self.covariance.get_survey_volume()

        eff_vols['lya'] = eff_vol_lya
        eff_vols['tracer'] = eff_vol_tr

        return eff_vols

            

    def _get_z_bins(self):
        if self.config['survey'].get('z bin centres', None) is not None:
            z_bin_centres = np.array(self.config['survey'].get('z bin centres').split(",")).astype(float)
            dz = np.zeros(z_bin_centres.size)
            dz[1:-1] = (z_bin_centres[2:] - z_bin_centres[:-2]) / 2.
            dz[0] = z_bin_centres[1] - z_bin_centres[0]
            dz[-1] = z_bin_centres[-1] - z_bin_centres[-2]
            z_bin_edges = np.array([z_bin_centres - dz / 2, z_bin_centres + dz / 2])
        else:
            z_list = np.linspace(self.survey.zmin, self.survey.zmax, self.survey.num_z_bins + 1)
            z_bin_edges = np.array([[z_list[i], z_list[i + 1]] for i in range(self.survey.num_z_bins)]).T
            z_bin_centres = z_bin_edges.mean(axis=0)

        return z_bin_edges, z_bin_centres

    def compute_neff(self):
        """Compute effective number density of pixels, or tracers,
        defined in McDonald & Eisenstein (2007),
          as function of magnitude and mean redshift"""

        neff = self.compute_weights(compute_neff=True)

        return neff
    
    def compute_zeff(self):
        """Compute effective redshift of measurements"""
        z_bin_edges,z_bin_centres = self._get_z_bins()

        w_lya = np.zeros(z_bin_centres.size)
        w_cross = np.zeros(z_bin_centres.size)
        w_tracer = np.zeros(z_bin_centres.size)

        for iz, zc in enumerate(z_bin_centres):
            lmin = self.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
            lmax = self.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])

            self.covariance(lmin, lmax)
            self.covariance.compute_eff_density_and_noise()
            # self.covariance.compute_3d_power_variance(0.14, 0.6)
            self.covariance.compute_cross_power_variance(0.14, 0.6)

            w_lya[iz] = sum(self.covariance._w_lya)
            w_cross[iz] = sum(self.covariance._w_cross)
            w_tracer[iz] = sum(self.covariance._w_tracer)
        
        z_eff_lya = np.sum(z_bin_centres * w_lya)/sum(w_lya)
        z_eff_cross = np.sum(z_bin_centres * w_cross)/sum(w_cross)
        z_eff_tracer = np.sum(z_bin_centres * w_tracer)/sum(w_tracer)

        z_eff_lya_cross = np.sum(z_bin_centres * w_lya*w_cross)/(sum(w_lya*w_cross))

        print('zeff lya: ',round(z_eff_lya,3))
        print('zeff cross: ',round(z_eff_cross,3))
        print('zeff tracer: ',round(z_eff_tracer,3))
        print('zeff lya+cross: ',round(z_eff_lya_cross,3))



    def compute_survey_volume(self):
        """Compute survey volume (Mpc/h) as a function of redshift"""
        survey_volume = np.zeros(self.survey.num_z_bins)

        z_bin_edges,z_bin_centres = self._get_z_bins()
        for iz, zc in enumerate(z_bin_centres):
            lmin = self.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
            lmax = self.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])

            self.covariance(lmin, lmax)
            survey_volume[iz] = self.covariance.get_survey_volume()

        return survey_volume, z_bin_centres

    def compute_pk(self):
        """Compute snr per mode for k=0.14h/Mpc, mu=0.6"""

        z_bin_centres, p3d_info, n_pk_lya, n_pk_qso = self._compute_pk()

        return z_bin_centres, p3d_info, n_pk_lya, n_pk_qso


    def _compute_pk(self):
        """Compute snr per mode for k=0.14h/Mpc, mu=0.6"""

        p3d_z_k_mu = np.zeros((self.survey.num_z_bins,
                             self.power_spec._num_k_bins,
                             self.power_spec._num_mu_bins-1))
        p3d_var_z_k_mu = np.zeros_like(p3d_z_k_mu)
        p3d_tracer_z_k_mu = np.zeros_like(p3d_z_k_mu)
        p3d_tracer_var_z_k_mu = np.zeros_like(p3d_z_k_mu)

        p3d_info = {}

        n_pk_z_lya = np.zeros(self.survey.num_z_bins)
        n_pk_z_tr = np.zeros(self.survey.num_z_bins)

        z_bin_edges,z_bin_centres = self._get_z_bins()
        for iz, zc in enumerate(z_bin_centres):
            lmin = self.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
            lmax = self.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])

            self.covariance(lmin, lmax)

            # some survey settings
            pix_width = self.covariance._pix_kms
            resolution = self.covariance._res_kms

            #weighting
            self.covariance.compute_eff_density_and_noise()

            n_pk_z_lya[iz], n_pk_z_tr[iz] = self.covariance.compute_n_pk(0.14, 0.6)

            for i, mu in enumerate(self.power_spec.mu):
                p3d = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                
                p3d_tracer = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                            mu, pix_width, resolution, self._tracer) 
                                        for k in self.power_spec.k]) 

                p3d_var = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                
                p3d_tracer_var = np.array([self.covariance.compute_tracer_power_variance(k,mu) 
                                    for k in self.power_spec.k])
                
                p3d_z_k_mu[iz,:,i] = p3d
                p3d_tracer_z_k_mu[iz,:,i] = p3d_tracer

                p3d_var_z_k_mu[iz,:,i] = p3d_var
                p3d_tracer_var_z_k_mu[iz,:,i] = p3d_tracer_var

        p3d_info['p_lya'] = p3d_z_k_mu
        p3d_info['p_tracer'] = p3d_tracer_z_k_mu
        p3d_info['var_lya'] = p3d_var_z_k_mu
        p3d_info['var_tracer'] = p3d_tracer_var_z_k_mu

        return z_bin_centres, p3d_info, n_pk_z_lya, n_pk_z_tr

    def run_bao_forecast(self):
        print('Running BAO forecast')

        # areas = np.array(self.survey.area_deg2)
        # resolutions = np.array(self.survey.res_kms)
        # qso_densities = np.array(self.survey.qso_density)

        if self.covariance.per_mag:
            sigma_dh_lya = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_lya = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_dh_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_dh_cross = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_cross = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_dh_lya_lya_lya_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            sigma_da_lya_lya_lya_tracer = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
            corr_coef = np.zeros((self.survey.num_z_bins, self.survey.num_mag_bins))
        else:
            sigma_da_lya = np.zeros(self.survey.num_z_bins)
            sigma_dh_lya = np.zeros(self.survey.num_z_bins)
            sigma_da_tracer = np.zeros(self.survey.num_z_bins)
            sigma_dh_tracer = np.zeros(self.survey.num_z_bins)
            sigma_da_cross = np.zeros(self.survey.num_z_bins)
            sigma_dh_cross = np.zeros(self.survey.num_z_bins)
            sigma_da_lya_lya_lya_tracer = np.zeros(self.survey.num_z_bins)
            sigma_dh_lya_lya_lya_tracer = np.zeros(self.survey.num_z_bins)
            corr_coef = np.zeros(self.survey.num_z_bins)

        z_bin_edges,z_bin_centres = self._get_z_bins()
        for iz, zc in enumerate(z_bin_centres):
            print(f"z bin = [{z_bin_edges[0,iz]}-{z_bin_edges[1,iz]}], bin centre = {zc}")

            # observed wavelength range from redshift limits
            lmin = self.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
            lmax = self.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])

            #call function, setting bin width
            self.covariance(lmin,lmax)

            # some survey settings
            pix_width = self.covariance._pix_kms
            resolution = self.covariance._res_kms

            # this uses Luminosity, density, noise model
            self.covariance.compute_eff_density_and_noise()
                    
            #empty array for fisher information
            #if per magnitude add extra dimension
            if self.covariance.per_mag:
                fisher_matrix_lya = np.zeros((2,2,self.survey.num_mag_bins))
                fisher_matrix_tracer = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)
            else:
                fisher_matrix_lya = np.zeros((2,2))      
                fisher_matrix_tracer = np.zeros_like(fisher_matrix_lya)
                fisher_matrix_cross = np.zeros_like(fisher_matrix_lya)      

            
            for i, mu in enumerate(self.power_spec.mu):
                p3d_lya = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                                 mu, pix_width, resolution, 'lya') 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                p3d_tracer = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                            mu, pix_width, resolution, self._tracer) 
                                        for k in self.power_spec.k]) # (Mpc/h)**3
                p3d_cross = np.array([self.power_spec.compute_p3d_hmpc_smooth(zc, k,
                                            mu, pix_width, resolution, self._cross_tracer) 
                                            for k in self.power_spec.k]) # (Mpc/h)**3

                #var^2/P^2
                p3d_lya_var = np.array([self.covariance.compute_3d_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                p3d_tracer_var = np.array([self.covariance.compute_tracer_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6
                p3d_cross_var = np.array([self.covariance.compute_cross_power_variance(k,mu) 
                                    for k in self.power_spec.k]) # (Mpc/h)**6


                #get derivatives of p3d
                dp3d_lya_dlogk = self.get_dp_dlogk(p3d_lya,mu)
                dp3d_tracer_dlogk = self.get_dp_dlogk(p3d_tracer,mu)
                dp3d_cross_dlogk = self.get_dp_dlogk(-p3d_cross,mu)
                
                # k = sqrt( kp**2 + kt**2)
                # k'  = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
                # k' = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
                # dk/dap         = mu2 * k
                # dlog(k)/dap    = mu2
                # dlog(k)/dat    = (1-mu2)
                # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
                # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
                fisher_matrix_lya += self.get_fisher(mu,dp3d_lya_dlogk,p3d_lya_var)
                fisher_matrix_tracer += self.get_fisher(mu,dp3d_tracer_dlogk,p3d_tracer_var)
                fisher_matrix_cross += self.get_fisher(mu,dp3d_cross_dlogk,p3d_cross_var)
            

            sigma_dh_lya_z, sigma_da_lya_z, corr_coef_lya_z = self.print_bao(fisher_matrix_lya,'lya')
            sigma_dh_tracer_z, sigma_da_tracer_z, corr_coef_tracer_z = self.print_bao(fisher_matrix_tracer,self._tracer)
            sigma_dh_cross_z, sigma_da_cross_z, corr_coef_cross_z = self.print_bao(fisher_matrix_cross,'cross')

            sigma_dh_lya[iz] = sigma_dh_lya_z
            sigma_da_lya[iz] = sigma_da_lya_z
            corr_coef[iz] = corr_coef_lya_z

            sigma_dh_tracer[iz] = sigma_dh_tracer_z
            sigma_da_tracer[iz] = sigma_da_tracer_z

            sigma_dh_cross[iz] = sigma_dh_cross_z
            sigma_da_cross[iz] = sigma_da_cross_z

            bao_corr_coef = 0
            
            sigma_da_lya_lya_lya_tracer[iz] = self.combine_BAO(sigma_da_lya_z,sigma_da_cross_z,bao_corr_coef)
            sigma_dh_lya_lya_lya_tracer[iz] = self.combine_BAO(sigma_dh_lya_z,sigma_dh_cross_z,bao_corr_coef)

            print(f'at (lyaxlya + lyax{self._tracer}): {sigma_da_lya_lya_lya_tracer[iz]}, ap (lyaxlya + lyax{self._tracer}): {sigma_dh_lya_lya_lya_tracer[iz]}')
        
        
        if self.covariance.per_mag:
            sigma_da_combined_lya_m = 1./np.sqrt(np.sum(1./sigma_da_lya**2,axis=0))
            sigma_dh_combined_lya_m = 1./np.sqrt(np.sum(1./sigma_dh_lya**2,axis=0))
            sigma_da_combined_qso_m = 1./np.sqrt(np.sum(1./sigma_da_tracer**2,axis=0))
            sigma_dh_combined_qso_m = 1./np.sqrt(np.sum(1./sigma_dh_tracer**2,axis=0))
            sigma_da_combined_cross_m = 1./np.sqrt(np.sum(1./sigma_da_cross**2,axis=0))
            sigma_dh_combined_cross_m = 1./np.sqrt(np.sum(1./sigma_dh_cross**2,axis=0))
            sigma_da_combined_comb_m = 1./np.sqrt(np.sum(1./sigma_da_lya_lya_lya_tracer**2,axis=0))
            sigma_dh_combined_comb_m = 1./np.sqrt(np.sum(1./sigma_dh_lya_lya_lya_tracer**2,axis=0))

            sigma_da_combined = sigma_da_combined_lya_m[-1]
            sigma_dh_combined = sigma_dh_combined_lya_m[-1]
            sigma_da_combined_tracer = sigma_da_combined_qso_m[-1]
            sigma_dh_combined_tracer = sigma_dh_combined_qso_m[-1]
            sigma_da_combined_cross = sigma_da_combined_cross_m[-1]
            sigma_dh_combined_cross = sigma_dh_combined_cross_m[-1]
            sigma_da_combined_comb = sigma_da_combined_comb_m[-1]
            sigma_dh_combined_comb = sigma_dh_combined_comb_m[-1]

        else:
            sigma_da_combined = 1./np.sqrt(np.sum(1./sigma_da_lya**2))
            sigma_dh_combined = 1./np.sqrt(np.sum(1./sigma_dh_lya**2))
            sigma_da_combined_tracer = 1./np.sqrt(np.sum(1./sigma_da_tracer**2))
            sigma_dh_combined_tracer = 1./np.sqrt(np.sum(1./sigma_dh_tracer**2))
            sigma_da_combined_cross = 1./np.sqrt(np.sum(1./sigma_da_cross**2))
            sigma_dh_combined_cross = 1./np.sqrt(np.sum(1./sigma_dh_cross**2))
            sigma_da_combined_comb = 1./np.sqrt(np.sum(1./sigma_da_lya_lya_lya_tracer**2))
            sigma_dh_combined_comb = 1./np.sqrt(np.sum(1./sigma_dh_lya_lya_lya_tracer**2))



        print(f'\n Combined: at (lyaxlya)={sigma_da_combined}'
                    fr', ap (lyaxlya)={sigma_dh_combined}')
        print(f'\n Combined: at ({self._tracer})={sigma_da_combined_tracer}'
                    fr', ap ({self._tracer})={sigma_dh_combined_tracer}')
        print(f'\n Combined: at (lyaxtr)={sigma_da_combined_cross}'
                    fr', ap (lyaxtr)={sigma_dh_combined_cross}')
        print(f'\n Combined: at (lyaxlya + lyaxtr)={sigma_da_combined_comb}'
                    fr', ap (lyaxlya + lyaxtr)={sigma_dh_combined_comb}')
       
        data = {}
        data["redshifts"] = z_bin_centres
        data["mean redshift"] = self.cosmo.z_ref
        data["magnitudes"] = {self.survey.band:self.survey.maglist}
        # data["at_err_lya"] = sigma_da_combined
        # data["ap_err_lya"] = sigma_dh_combined
        # data["at_err_cross"] = sigma_da_combined
        # data["ap_err_cross"] = sigma_dh_combined
        data['ap_err_lya_z'] = sigma_dh_lya
        data['at_err_lya_z'] = sigma_da_lya
        data['ap_err_tracer_z'] = sigma_dh_tracer
        data['at_err_tracer_z'] = sigma_da_tracer
        data['ap_err_cross_z'] = sigma_dh_cross
        data['at_err_cross_z'] = sigma_da_cross
        data['ap_err_comb_z'] = sigma_dh_lya_lya_lya_tracer
        data['at_err_comb_z'] = sigma_da_lya_lya_lya_tracer
        
        if self.covariance.per_mag:
            data['ap_err_lya_m'] = sigma_dh_combined_lya_m
            data['at_err_lya_m'] = sigma_da_combined_lya_m
            data['ap_err_qso_m'] = sigma_dh_combined_qso_m
            data['at_err_qso_m'] = sigma_da_combined_qso_m
            data['ap_err_cross_m'] = sigma_dh_combined_cross_m
            data['at_err_cross_m'] = sigma_da_combined_cross_m

        # if forecast is not None:
        #     #load data to plots instance
        #     self.plots = Plots(forecast,data=data)

        #     if self.config['control'].getboolean('plot bao'):
        #         if self.covariance.per_mag:
        #             self.plots.plot_da_h_m()
        #             self.plots.fig.savefig(self.out_folder.joinpath('dap_dat_dm.png'))
        #         else:
        #             self.plots.plot_da_h_z()
        #             self.plots.fig.savefig(self.out_folder.joinpath('dap_dat_z.png'))

        return data
        
    def get_dp_dlogk(self,model,mu):
        """Return the differential of the a peak power spectrum component, 
            with respect to log k, add peak broadening."""
          
        pk = self.get_p_pk(model)
        
        pk = self.apply_peak_smoothing(pk,mu)

        # derivative of model wrt to log(k)
        dmodel = np.zeros(self.power_spec.k.size)
        dmodel[1:] = pk[1:]-pk[:-1]
        dmodel_dlk  = dmodel/self.power_spec.dlogk
                    
        return dmodel_dlk
    
    def get_p_pk(self,model):
        #get peak only component
        # smooth = get_pk_smooth(self.cosmo.results,self.power_spec.k,model)
        # pk = model - smooth
        x = self.power_spec.logk
        y = np.log(model)
        x -= np.mean(x)
        x /= (np.max(x)-np.min(x))
        w=np.ones(x.size)
        w[:3] *= 1.e8 
        coef=np.polyfit(x,y,8,w=w)
        pol=np.poly1d(coef)
        smooth = np.exp(pol(x))
        pk = model-smooth    

        return pk

    def apply_peak_smoothing(self,pk,mu):
        
        kp = mu * self.power_spec.k
        kt = np.sqrt(1-mu**2) * self.power_spec.k
        
        # Eisenstein, Seo, White, 2007, Eq. 12
        sig_nl_perp = 3.26 # Mpc/h
        f = self.cosmo.growth_rate # lograthmic growth (at z_ref)
        sig_nl_par = (1 + f) * sig_nl_perp # Mpc/h
        
        pk *= np.exp(-0.5 * ((sig_nl_par * kp)**2 + (sig_nl_perp * kt)**2))

        return pk

    def get_fisher(self,mu,dp_dlogk,var):
        """Compute fisher matrix for ap, at"""
        h = [mu**2,1 - mu**2]
        if self.covariance.per_mag:
            return np.outer(h,h)[:,:,None] * np.sum(dp_dlogk**2 / var.T, axis=1).T
        else:
            return np.outer(h,h) * np.sum(dp_dlogk**2 / var)
        
    def print_bao(self,fisher_matrix,which):
        """Print BAO results from Fisher matrix"""
        if self.covariance.per_mag:
            cov = np.linalg.inv(fisher_matrix.T)
            cov_diag = np.diagonal(cov.T,axis1=0,axis2=1)
            sigma_dh = np.sqrt(cov_diag.T[0])
            sigma_da = np.sqrt(cov_diag.T[1])
            corr_coef = cov.T[0,1]/np.sqrt(cov_diag.T[0]*cov_diag.T[1])

            print(f"ap ({which})={sigma_dh[-1]}, at ({which})={sigma_da[-1]},corr={corr_coef[-1]}")
        else:
            cov = np.linalg.inv(fisher_matrix)
            sigma_dh = np.sqrt(cov[0,0])
            sigma_da = np.sqrt(cov[1,1])    
            corr_coef = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

            print(f"ap ({which})={sigma_dh}, at ({which})={sigma_da},corr={corr_coef}")
            
        return sigma_dh, sigma_da, corr_coef


    def combine_BAO(self,dx1, dx2, rho):
        """
        Combine two BAO measurements (centred at 1) from different tracers, with correlation coefficient rho. This is a high noise approximation, 
        to be replaced by a full Fisher covariance matrix.
        
        Returns:
            dx_combined: uncertainty of the combined result
        """
        # Covariance matrix
        # C = np.array([
        #     [dx1**2, rho * dx1 * dx2],
        #     [rho * dx1 * dx2, dx2**2]
        # ])

        # # Inverse covariance matrix
        # C_inv = np.linalg.inv(C)

        # # Weight vector
        # ones = np.ones(2)

        # # Combined value and uncertainty
        # dx_combined = (1 / (ones @ C_inv @ ones))

        dx_comb = 1 / (1 / dx1 + 1 / dx2)

        return dx_comb



                
            


