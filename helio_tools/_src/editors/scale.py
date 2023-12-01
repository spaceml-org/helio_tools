"""
scale.py

This module provides functions for normalizing and scaling solar maps. It includes functionality to normalize data using a provided function, normalize the radius of a solar map, and crop a solar map to a specified resolution.

Functions:
- normalize_fn(data: np.ndarray, fn: Callable) -> np.ndarray:
    Normalizes the data array using the provided function. The function should take a numpy array as input and return a numpy array.

- normalize_radius(s_map: Map, resolution: int, padding_factor: float = 0.1, crop: bool = True) -> Map:
    Normalizes the radius of the solar map to a specified resolution. The padding factor and crop parameters are optional.

- _crop_resolution(s_map: Map, resolution: int) -> Map:
    Crops the solar map to the specified resolution. This is a helper function used within normalize_radius.

Dependencies:
- sunpy.map.Map: For handling solar data.
- numpy: For numerical operations.
- astropy: For handling astronomical quantities and coordinates.
- warnings: For suppressing warnings during certain operations.

Note: This module is designed to work with solar data and may not be applicable to other types of astronomical data.
"""

from typing import Callable
from sunpy.map import Map
import warnings
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def normalize_fn(data: np.ndarray, fn: Callable) -> np.ndarray:
    """Normalizes the data array using the provided function. 
    The function should take a numpy array as input and return a numpy array.

    Args:
        data (np.ndarray): the data array to be normalized
        fn (Callable): the function to be applied to the data array

    Returns:
        np.ndarray: the normalized data array

    Usage:
    >>> import helio_tools as ht
    >>> import numpy as np
    >>> data = np.random.rand(100, 100)
    >>> fn = np.sin
    >>> norm_data = ht.normalize_fn(data, fn)    
    """
    return 2 * fn(data).data - 1


def normalize_radius(
        s_map: Map,
        resolution: int,
        padding_factor: float = 0.1,
        crop: bool = True,
) -> Map:
    """ 
    Normalizes the radius of the solar map to a specified resolution.

    Args:
        s_map (Map): the solar map to be normalized
        resolution (int): the desired resolution of the output map
        padding_factor (float, optional): the padding factor for the solar radius. Defaults to 0.1.
        crop (bool, optional): whether to crop the map to the specified resolution. Defaults to True.

    Returns:
        Map: the normalized solar map

    Usage:
    >>> import helio_tools as ht
    >>> import numpy as np
    >>> s_map = ht.load_fits_to_map(filename)
    >>> norm_map = ht.normalize_radius(s_map, 512)    
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + padding_factor) * r_obs_pix
        scale_factor = resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(
            recenter=True, scale=scale_factor, missing=0, order=4)
        if crop:
            s_map = _crop_resolution(s_map, resolution=resolution)
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']

        return s_map


def _crop_resolution(
        s_map: Map,
        resolution: int
) -> Map:
    """Crops the solar map to the specified resolution. 
    This is a helper function used within normalize_radius.

    Args:
        s_map (Map): the solar map to be cropped
        resolution (int): the desired resolution of the output map

    Returns:
        Map: the cropped solar map

    Usage:
    >>> import helio_tools as ht
    >>> import numpy as np
    >>> s_map = ht.load_fits_to_map(filename)
    >>> s_map = ht.normalize_radius(s_map, 512)
    >>> s_map = ht._crop_resolution(s_map, 512) # crop to 512x512 pixels

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arcs_frame = (resolution / 2) * s_map.scale[0].value
        s_map = s_map.submap(
            bottom_left=SkyCoord(-arcs_frame * u.arcsec, -
                                 arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
            top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
        pad_x = s_map.data.shape[0] - resolution
        pad_y = s_map.data.shape[1] - resolution
        s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                             top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)

        return s_map
