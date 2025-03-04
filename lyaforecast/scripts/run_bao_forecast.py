#!/usr/bin/env python
import argparse
import numpy as np
from lyaforecast import Forecast

def main():
    args = get_args()

    forecast = Forecast(args.config)

    forecast.run_bao_forecast(forecast)

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
