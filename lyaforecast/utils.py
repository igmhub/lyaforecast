from pathlib import Path
import os.path
import lyaforecast


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

    raise RuntimeError('The directory does not exist: ', input_path)
