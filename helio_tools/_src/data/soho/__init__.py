from typing import Optional, List
from dataclasses import dataclass

DEFAULT_WAVELENGTHS = [171, 193, 211, 304]



class SDOData:
    email: str
    base_path: str
    wavelengths: List[str | int | float]=DEFAULT_WAVELENGTHS
    n_workers: int

def download_soho(
        email: str, base_path: str,
        wavelengths: List[str | int | float]=DEFAULT_WAVELENGTHS,
        n_workers: int=5
) -> None:
    """A simple download script do down

    Args:
        email (str): the email account needed
        base_path (str):
        wavelength (list[int|str|float]): the wavelengths we would like to download
        n_workers (int): the number of workers for the download.
    """
    pass