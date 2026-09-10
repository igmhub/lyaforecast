"""Command-line entry point for the supported multi-tracer forecast."""
import argparse


def get_args(argv=None):
    """Parse one required INI configuration path."""
    parser = argparse.ArgumentParser(description="Run a multi-tracer BAO forecast.")
    parser.add_argument("--configs", "-i", required=True, metavar="CONFIG", help="INI configuration")
    return parser.parse_args(argv)


def main():
    """Run the same workflow exposed by NewForecast.new_run_forecast."""
    args = get_args()
    from lyaforecast.forecast_new import NewForecast

    NewForecast(args.configs).new_run_forecast()


if __name__ == "__main__":
    main()
