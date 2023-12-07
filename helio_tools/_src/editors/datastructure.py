from sunpy.map import Map
import numpy as np


def map_to_ndarray(s_map: Map) -> np.ndarray:
    """Converts a sunpy map to a numpy array.

    Args:
        s_map (Map): the sunpy map to be converted

    Returns:
        np.ndarray: the numpy array

    Usage:

    >>> import helio_tools as ht
    >>> s_map = ht.load_fits_to_map(filename)
    >>> data = ht.map_to_ndarray(s_map)

    """
    return s_map.data
