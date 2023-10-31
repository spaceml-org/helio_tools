from typing import Optional

import numpy as np


def constrast_normalize(
    data,
    shift: Optional[float] = None,
    use_median: bool = False,
    normalization: Optional[float] = None,
):
    if shift is None:
        shift = np.nanmedian(data) if use_median else np.nanmean(data)

    if normalization is None:
        normalization = np.nanstd(data)

    data = (data - shift) / (normalization + 10e-8)

    return data
