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
    table_path = os.path.join(Path.home(), '.iti', 'sdo_autocal_table.csv')
    os.makedirs(os.path.join(Path.home(), '.iti'), exist_ok=True)
    if not os.path.exists(table_path):
        request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/sdo_autocal_table.csv', filename=table_path)
    return pd.read_csv(table_path, parse_dates=['DATE'], index_col=0)


@lru_cache(maxsize=None)
def get_local_correction_table():
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    astropy.io.ascii.write(correction_table, path)
    return correction_table


def correct_degregation(
        s_map: Map,
        method: Optional[str]="auto",
        **kwargs
) -> Map:

    if method == "auto":
        correction_table = get_auto_calibration_table()
        index = correction_table["DATE"].sub(s_map.date.datetime).abs().idxmin()
        num = s_map.meta["wavelnth"]
        s_map = Map(s_map.data / correction_table.iloc[index][f"{int(num):04}"], s_map.meta)
    elif method == "aiapy":
        correction_table = get_local_correction_table()
        s_map = aiapy_correct_degregation(smap=s_map, correction_table=correction_table, **kwargs)
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
