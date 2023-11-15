from sunpy.map import Map
import numpy as np

def change_map_dtype(s_map: Map, dtype) -> Map:
    return Map(s_map.astype(dtype), s_map.meta)