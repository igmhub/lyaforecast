#!/usr/bin/env python
import argparse
from lyaforecast import Forecast
from lyaforecast.plots import Plots

def main():
    args = get_args()

    forecast = Forecast(args.config)

    plotter = Plots(forecast)

    z_bins,weights,_ = forecast.compute_weights()

    plotter.plot_weights(weights,z_bins)

    plotter.fig.savefig(forecast.out_folder.joinpath('weights.png'))

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
