from typing import Callable
from sunpy.map import Map
import warnings
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def normalize_fn(data: np.ndarray, fn: Callable) -> np.ndarray:
    return 2 * fn(data).data - 1

def normalize_radius(
        s_map: Map,
        resolution: int,
        padding_factor: float = 0.1,
        crop: bool = True,
) -> Map:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + padding_factor) * r_obs_pix
        scale_factor = resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=4)
        if crop:
            s_map = _crop_resolution(s_map, resolution=resolution)
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']

        return s_map


def _crop_resolution(
        s_map: Map,
        resolution: int
) -> Map:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arcs_frame = (resolution / 2) * s_map.scale[0].value
        s_map = s_map.submap(
            bottom_left=SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
            top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
        pad_x = s_map.data.shape[0] - resolution
        pad_y = s_map.data.shape[1] - resolution
        s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                             top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)

        return s_map