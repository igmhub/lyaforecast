#!/usr/bin/env python
import argparse
import numpy as np
from lyaforecast import Forecast

def main():
    args = get_args()

    forecast = Forecast(args.configs[0])

    forecast.run_forecast()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Run Forecast.')

    parser.add_argument('--configs', '-i',
                        type=str, 
                        nargs='+',
                        default=None, 
                        help='Config file')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
