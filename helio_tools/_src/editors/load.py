"""
load.py

This module provides a function to load FITS files into sunpy's Map objects. It is designed to handle solar data in FITS format.

Functions:
- load_fits_to_map(filename: str) -> Map:
    Opens a FITS file and returns a sunpy.map.Map object. All warnings during the loading process are ignored.

Dependencies:
- sunpy.map.Map: For handling solar data.
- warnings: For suppressing warnings during the loading process.

Note: This module is designed to work with solar data in FITS format and may not be applicable to other types of astronomical data.

Usage:

>>> import helio_tools as ht
>>> s_map = ht.load_fits_to_map(filename)

For more details on sunpy's Map object, see https://docs.sunpy.org/en/stable/reference/map.html
"""

from sunpy.map import Map
import warnings


def load_fits_to_map(filename: str) -> Map:
    """This opens a file and returns a sunpy.map.Map
    objects. All warnings are squashed

    Args:
        filename (str): the path to a file

    Returns:
        spmap (sunpy.map.Map): a sunpy map object.
         See https://docs.sunpy.org/en/stable/reference/map.html
         for details

    Usage: 

    >>> import helio_tools as ht
    >>> s_map = ht.load_fits_to_map(filename)

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_map = Map(filename)
        s_map.meta["timesys"] = "tai"  # fix leap seconds
        return s_map
