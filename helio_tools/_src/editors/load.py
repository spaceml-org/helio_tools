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
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_map = Map(filename)
        s_map.meta["timesys"] = "tai" # fix leap seconds
        return s_map