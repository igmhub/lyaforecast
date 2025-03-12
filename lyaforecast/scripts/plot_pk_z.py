#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots

def main():
    args = get_args()

    forecast = Forecast(args.config)

    plotter = Plots(forecast=forecast)

    z_bin_centres,info,_,_ = forecast.compute_pk()

    plotter.plot_pk_z(z_bin_centres,info)

    plotter.fig.savefig(forecast.out_folder.joinpath('pk_z.png'))

    plotter.plot_var_pk_z(z_bin_centres,info)

    plotter.fig.savefig(forecast.out_folder.joinpath('var_pk_z.png'))

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
