from pathlib import Path
import os.path
import logging

import lyaforecast

import numpy as np
import mcfit

def check_file(input_path):
    """ Verify a file exists, if not raise error

    Parameters
    ----------
    path : string
        Input path. Only absolute.

    """
    # First check if it's an absolute path
    if input_path.is_file():
        return input_path
    else:
        raise RuntimeError('The path does not exist: ', input_path)

def get_file(path):
    """ Find files on the system.

    Checks if it's an absolute or relative path inside LyaCast

    Parameters
    ----------
    path : string
        Input path. Can be absolute or relative to lyacast
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_file():
        return input_path
    # Get the lyacast path and check inside lyacast (this returns LyaCast/lyacast)
    lyacast_path = Path(os.path.dirname(lyaforecast.__file__))

    # Check if it's a resource
    resource = lyacast_path / 'resources' / input_path
    if resource.is_file():
        return resource
    
    # Check if it's a data source
    data = lyacast_path / 'resources/data' / input_path
    if data.is_file():
        return data
    
    # Check if it's a default config
    default_cfg = lyacast_path / 'resources/default_configs' / input_path
    if default_cfg.is_file():
        return default_cfg
    
    # Check if it's a camb config
    camb_cfg = lyacast_path / 'resources/camb_configs' / input_path
    if camb_cfg.is_file():
        return camb_cfg

    raise RuntimeError('The path does not exist: ', input_path, 'or', resource)

def get_dir(path):
    """ Find directory on the system.

    Checks if it's an absolute or relative path inside LyaCast

    Parameters
    ----------
    path : string
        Input path. Can be absolute or relative to lyacast
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_dir():
        return input_path

    # Get the lyacast path and check inside lyacast (this returns LyaCast/lyacast)
    lyacast_path = Path(os.path.dirname(lyaforecast.__file__))

    # Check if it's a resource
    resource = lyacast_path / 'resources' / input_path
    if resource.is_dir():
        return resource
    
    # Check if it's a data source (folder)
    data = lyacast_path / 'resources/data' / input_path
    if data.is_dir():
        return data

    raise RuntimeError('The directory does not exist: ', input_path)

def setup_logger(out_folder):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
        )

    # File handler (WARNING and above)
    file_handler = logging.FileHandler(f"{out_folder}/forecast.log")
    file_handler.setLevel(logging.WARNING)

    return logger

