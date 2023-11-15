from sunpy.map import Map
import numpy as np

def map_to_ndarray(s_map: Map) -> np.ndarray:
    return s_map.data