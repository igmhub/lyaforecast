from pathlib import Path
import os.path
import logging


def check_file(input_path):
    """Verify a file exists and return it, raising an error otherwise.

    Parameters
    ----------
    input_path : Path
        Absolute path to the file to check.

    Returns
    -------
    Path
        The same path if it exists.

    Raises
    ------
    RuntimeError
        If the path does not point to an existing file.
    """
    if input_path.is_file():
        return input_path
    else:
        raise RuntimeError('The path does not exist: ', input_path)


def get_file(path):
    """Resolve a file path, searching the lyaforecast resource directories.

    Checks in order: absolute path, resources/, resources/data/,
    resources/default_configs/, resources/camb_configs/.

    Parameters
    ----------
    path : str
        Input path; can be absolute or relative to the lyaforecast package.

    Returns
    -------
    Path
        Resolved absolute path to the file.

    Raises
    ------
    RuntimeError
        If the file cannot be found in any search location.
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_file():
        return input_path
    # Locate the installed package resources.
    package_path = Path(__file__).resolve().parent

    # Check if it's a resource
    resource = package_path / 'resources' / input_path
    if resource.is_file():
        return resource

    # Check if it's a data source
    data = package_path / 'resources/data' / input_path
    if data.is_file():
        return data

    # Check if it's a default config
    default_cfg = package_path / 'resources/default_configs' / input_path
    if default_cfg.is_file():
        return default_cfg

    # Check if it's a camb config
    camb_cfg = package_path / 'resources/camb_configs' / input_path
    if camb_cfg.is_file():
        return camb_cfg

    raise RuntimeError('The path does not exist: ', input_path, 'or', resource)


def get_dir(path):
    """Resolve a directory path, searching the lyaforecast resource directories.

    Checks in order: absolute path, resources/, resources/data/.

    Parameters
    ----------
    path : str
        Input path; can be absolute or relative to the lyaforecast package.

    Returns
    -------
    Path
        Resolved absolute path to the directory.

    Raises
    ------
    RuntimeError
        If the directory cannot be found in any search location.
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_dir():
        return input_path

    # Locate the installed package resources.
    package_path = Path(__file__).resolve().parent

    # Check if it's a resource
    resource = package_path / 'resources' / input_path
    if resource.is_dir():
        return resource

    # Check if it's a data source (folder)
    data = package_path / 'resources/data' / input_path
    if data.is_dir():
        return data

    raise RuntimeError('The directory does not exist: ', input_path)


def setup_logger(out_folder):
    """Configure console and file logging for one output directory.

    Repeated calls for the same directory reuse handlers. Application root
    logging is left unchanged, and messages do not propagate to root handlers.

    Parameters
    ----------
    out_folder : str or Path
        Directory for forecast.log; created if missing.

    Returns
    -------
    logging.Logger
        Logger shared by forecasts writing to this directory.
    """
    out_folder = Path(out_folder).resolve()
    out_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{__name__}.{out_folder}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        file_handler = logging.FileHandler(out_folder / 'forecast.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
