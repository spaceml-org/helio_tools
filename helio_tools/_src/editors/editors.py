from typing import Optional
import numpy as np


def constrast_normalize(
    data,
    shift: Optional[float] = None,
    use_median: bool = False,
    normalization: Optional[float] = None,
):
"""
    Normalizes the data by subtracting the shift and dividing by the normalization. 
    If shift is None, it is set to the median of the data. 
    If normalization is None, it is set to the standard deviation of the data.

    Args:

    - data (np.ndarray): The data to be normalized.
    - shift (float, optional): The value to be subtracted from the data. Defaults to None.

    - use_median (bool, optional): If True, the median of the data is used as the shift. Otherwise, the mean is used. Defaults to False.
    - normalization (float, optional): The value to be divided by. Defaults to None.

    Returns:

    - np.ndarray: The normalized data.

    Usage:

    >>> import helio_tools as ht
    >>> data = ht.load_fits_to_ndarray(filename)
    >>> data = ht.contrast_normalize(data)
"""
    if shift is None:
        shift = np.nanmedian(data) if use_median else np.nanmean(data)

    if normalization is None:
        normalization = np.nanstd(data)

    data = (data - shift) / (normalization + 10e-8)

    return data
