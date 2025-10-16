#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
import numpy as np

def main():

    args = get_args()

    forecast = Forecast(args.config)

    """Compute effective redshift of measurements"""
    z_bin_edges,z_bin_centres = forecast.survey.z_bin_edges, forecast.survey.z_bin_centres
    #_get_z_bins()

    w_lya = np.zeros(z_bin_centres.size)
    w_cross = np.zeros(z_bin_centres.size)
    w_tracer = np.zeros(z_bin_centres.size)

    for iz, zc in enumerate(z_bin_centres):
        lmin = forecast.cosmo.LYA_REST * (1 + z_bin_edges[0,iz])
        lmax = forecast.cosmo.LYA_REST * (1 + z_bin_edges[1,iz])

        forecast.covariance(lmin, lmax)
        forecast.covariance.compute_eff_density_and_noise()
        forecast.covariance.compute_cross_power_variance(0.14, 0.6)

        w_lya[iz] = sum(forecast.covariance._w_lya)
        w_cross[iz] = sum(forecast.covariance._w_cross)
        w_tracer[iz] = sum(forecast.covariance._w_tracer)
    
    z_eff_lya = np.sum(z_bin_centres * w_lya)/sum(w_lya)
    z_eff_cross = np.sum(z_bin_centres * w_cross)/sum(w_cross)
    z_eff_tracer = np.sum(z_bin_centres * w_tracer)/sum(w_tracer)

    z_eff_lya_cross = np.sum(z_bin_centres * w_lya*w_cross)/(sum(w_lya*w_cross))

    print('zeff lya: ',round(z_eff_lya,3))
    print('zeff cross: ',round(z_eff_cross,3))
    print('zeff tracer: ',round(z_eff_tracer,3))
    print('zeff lya+cross: ',round(z_eff_lya_cross,3))

    
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Run Forecast.')

    parser.add_argument('--config', '-i',
                        type=str, 
                        default=None,
                        help='Config file')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()