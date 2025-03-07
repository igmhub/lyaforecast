#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots

def main():
    args = get_args()

    forecast = Forecast(args.config)

    plotter = Plots(forecast=forecast)

    z_bin_centres,_,_,n_pk_z_lya,n_pk_z_qso = forecast.compute_pk()

    plotter.plot_n_pk_z(z_bin_centres,n_pk_z_lya,n_pk_z_qso)

    plotter.fig.savefig(forecast.out_folder.joinpath('nP.png'))

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
