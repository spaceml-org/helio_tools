"""
calibration.py

This module provides functions for calibrating solar maps. It includes functionality to correct degradation in solar images and to fetch calibration tables.

Functions:
- get_auto_calibration_table() -> pd.DataFrame:
    Fetches the auto calibration table from a remote server and caches it locally. If the table is already cached, it returns the cached version.

- get_local_correction_table() -> astropy.table.Table:
    Fetches the local correction table. If the table doesn't exist, it creates one.

- correct_degregation(s_map: Map, method: Optional[str]="auto", **kwargs) -> Map:
    Corrects the degradation in a solar map. The method of correction can be specified.

Dependencies:
- sunpy.map.Map: For handling solar data.
- aiapy.calibrate: For correcting degradation in solar images.
- pandas: For handling data frames.
- astropy: For handling astronomical quantities and coordinates.
- os, pathlib, urllib.request: For handling file paths and HTTP requests.

Note: This module is designed to work with solar data and may not be applicable to other types of astronomical data.
"""

from typing import Optional
import numpy as np
from sunpy.map import Map
from aiapy.calibrate import correct_degradation as aiapy_correct_degregation
from aiapy.calibrate.util import get_correction_table
from functools import lru_cache
import os
from pathlib import Path
from urllib import request
import pandas as pd
import astropy


@lru_cache(maxsize=None)
def get_auto_calibration_table():
    """
    Fetches the auto calibration table from a remote server and caches it locally. If the table is already cached, it returns the cached version.

    Returns:
    - pd.DataFrame: The auto calibration table.

    Usage:
    >>> table = get_auto_calibration_table()


    """
    table_path = os.path.join(Path.home(), '.iti', 'sdo_autocal_table.csv')
    os.makedirs(os.path.join(Path.home(), '.iti'), exist_ok=True)
    if not os.path.exists(table_path):
        request.urlretrieve(
            'http://kanzelhohe.uni-graz.at/iti/sdo_autocal_table.csv', filename=table_path)
    return pd.read_csv(table_path, parse_dates=['DATE'], index_col=0)


@lru_cache(maxsize=None)
def get_local_correction_table():
    """
    Fetches the local correction table. If the table doesn't exist, it creates one.

    Returns:
    - astropy.table.Table: The local correction table.

    Usage:
    >>> table = get_local_correction_table()
    """
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    astropy.io.ascii.write(correction_table, path)
    return correction_table


def correct_degregation(
    s_map: Map,
    method: Optional[str] = "auto",
    **kwargs
) -> Map:
    """
    Corrects the degradation in a solar map. The method of correction can be specified.

    Parameters:
    - s_map (Map): The solar map to be corrected.
    - method (Optional[str]): The method of correction. Can be "auto" or "aiapy". Defaults to "auto".
    - **kwargs: Additional keyword arguments to be passed to the aiapy.calibrate.correct_degradation() function.

    Returns:
    - Map: The corrected solar map.

    Usage:
    >>> import helio_tools as ht
    >>> s_map = ht.load_fits_to_map(filename)
    >>> s_map = ht.correct_degradation(s_map, method="auto")

    """

    if method == "auto":
        correction_table = get_auto_calibration_table()
        index = correction_table["DATE"].sub(
            s_map.date.datetime).abs().idxmin()
        num = s_map.meta["wavelnth"]
        s_map = Map(
            s_map.data / correction_table.iloc[index][f"{int(num):04}"], s_map.meta)
    elif method == "aiapy":
        correction_table = get_local_correction_table()
        s_map = aiapy_correct_degregation(
            smap=s_map, correction_table=correction_table, **kwargs)
    elif method is None:
        pass
    else:
        msg = "Unrecognized method"
        msg += f"Must be 'auto' or 'aiapy'. User input: {method}"
        raise ValueError(msg)

    # TODO: check why this is here and not something special...
    data = np.nan_to_num(s_map.data)
    data /= s_map.meta["exptime"]
    return Map(data.astype(np.float32), s_map.meta)
