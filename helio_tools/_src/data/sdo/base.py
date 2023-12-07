"""
Script to download data products from the SDO database at http://jsoc.stanford.edu/ 
to a local directory. Uses the drms python package, the default package for downloading SDO data. 

Documentation for DRMS: https://docs.sunpy.org/projects/drms/en/latest/
"""
from typing import Optional
import argparse
import logging
import multiprocessing
import os
from datetime import timedelta, datetime
from urllib import request

import tqdm
import warnings
import os
import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from sunpy.io._fits import header_to_fits
from sunpy.util import MetaDict
from helio_tools._src.utils.time import check_datetime_format
import typer
from loguru import logger

DEFAULT_WAVELENGTHS = [171, 193, 211, 304]


class SDODownloader:
    def __init__(
        self,
        base_path: str = None,
        email: str = None,
        wavelengths: list[str | int | float] = DEFAULT_WAVELENGTHS,
        n_workers: int = 5,
    ) -> None:
        """The SDO Downloader is an efficent way to download data from the SDO database.

        Args:
            base_path (str): the base path where the data should be downloaded to.
            email (str): the email account needed
            wavelength (list[int|str|float]): the wavelengths we would like to download
            n_workers (int): the number of workers for the download.

        Example Usage:

            >>> downloader_sdo = SDODownloader(...)
            >>> downloader_sdo.downloadDate(datetime(2022, 3, 1))

        """
        self.ds_path = base_path
        self.wavelengths = [str(wl) for wl in wavelengths]
        self.n_workers = n_workers
        [
            os.makedirs(os.path.join(base_path, wl), exist_ok=True)
            for wl in self.wavelengths + ["6173"]
        ]

        self.drms_client = drms.Client(email=email)

    def downloadDate(self, date: datetime):
        """Download FITS data for a specific date."""
        id = date.isoformat()
        logging.info("Start download: %s" % id)
        time_param = "%sZ" % date.isoformat("_", timespec="seconds")

        # query Magnetogram Instrument
        ds_hmi = "hmi.M_720s[%s]{magnetogram}" % time_param
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_hmi, segment_hmi = self.drms_client.query(
            ds_hmi, key=",".join(keys_hmi), seg="magnetogram"
        )
        if len(header_hmi) != 1 or np.any(header_hmi.QUALITY != 0):
            self.fetchDataFallback(date)
            return

        # query EUV Instrument
        ds_euv = "aia.lev1_euv_12s[%s][%s]{image}" % (
            time_param,
            ",".join(self.wavelengths),
        )
        keys_euv = self.drms_client.keys(ds_euv)
        header_euv, segment_euv = self.drms_client.query(
            ds_euv, key=",".join(keys_euv), seg="image"
        )
        if len(header_euv) != len(self.wavelengths) or np.any(header_euv.QUALITY != 0):
            self.fetchDataFallback(date)
            return

        queue = []
        for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.magnetogram):
            queue += [(h.to_dict(), s, date)]
        for (idx, h), s in zip(header_euv.iterrows(), segment_euv.image):
            queue += [(h.to_dict(), s, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)
        logging.info("Finished: %s" % id)

    def download(self, sample: tuple[dict, str, datetime]):
        header, segment, t = sample
        try:
            dir = os.path.join(self.ds_path, "%d" % header["WAVELNTH"])
            map_path = os.path.join(
                dir, "%s.fits" % t.isoformat("T", timespec="seconds")
            )
            if os.path.exists(map_path):
                return map_path
            # load map
            url = "http://jsoc.stanford.edu" + segment
            request.urlretrieve(url, filename=map_path)

            header["DATE_OBS"] = header["DATE__OBS"]
            header = header_to_fits(MetaDict(header))
            with fits.open(map_path, "update") as f:
                hdr = f[1].header
                for k, v in header.items():
                    if pd.isna(v):
                        continue
                    hdr[k] = v
                f.verify("silentfix")

            return map_path
        except Exception as ex:
            logging.info("Download failed: %s (requeue)" % header["DATE__OBS"])
            logging.info(ex)
            raise ex

    def fetchDataFallback(self, date: datetime):
        id = date.isoformat()

        logging.info("Fallback download: %s" % id)
        # query Magnetogram
        t = date - timedelta(hours=24)
        ds_hmi = "hmi.M_720s[%sZ/12h@720s]{magnetogram}" % t.replace(
            tzinfo=None
        ).isoformat("_", timespec="seconds")
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_tmp, segment_tmp = self.drms_client.query(
            ds_hmi, key=",".join(keys_hmi), seg="magnetogram"
        )
        assert len(header_tmp) != 0, "No data found!"
        date_str = (
            header_tmp["DATE__OBS"].replace("MISSING", "").str.replace("60", "59")
        )  # fix date format
        date_diff = np.abs(pd.to_datetime(date_str).dt.tz_localize(None) - date)
        # sort and filter
        header_tmp["date_diff"] = date_diff
        header_tmp.sort_values("date_diff")
        segment_tmp["date_diff"] = date_diff
        segment_tmp.sort_values("date_diff")
        cond_tmp = header_tmp.QUALITY == 0
        header_tmp = header_tmp[cond_tmp]
        segment_tmp = segment_tmp[cond_tmp]
        assert len(header_tmp) > 0, "No valid quality flag found"
        # replace invalid
        header_hmi = header_tmp.iloc[0].drop("date_diff")
        segment_hmi = segment_tmp.iloc[0].drop("date_diff")
        ############################################################
        # query EUV
        header_euv, segment_euv = [], []
        t = date - timedelta(hours=6)
        for wl in self.wavelengths:
            euv_ds = "aia.lev1_euv_12s[%sZ/12h@12s][%s]{image}" % (
                t.replace(tzinfo=None).isoformat("_", timespec="seconds"),
                wl,
            )
            keys_euv = self.drms_client.keys(euv_ds)
            header_tmp, segment_tmp = self.drms_client.query(
                euv_ds, key=",".join(keys_euv), seg="image"
            )
            assert len(header_tmp) != 0, "No data found!"
            date_str = (
                header_tmp["DATE__OBS"].replace("MISSING", "").str.replace("60", "59")
            )  # fix date format
            date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - date).abs()
            # sort and filter
            header_tmp["date_diff"] = date_diff
            header_tmp.sort_values("date_diff")
            segment_tmp["date_diff"] = date_diff
            segment_tmp.sort_values("date_diff")
            cond_tmp = header_tmp.QUALITY == 0
            header_tmp = header_tmp[cond_tmp]
            segment_tmp = segment_tmp[cond_tmp]
            assert len(header_tmp) > 0, "No valid quality flag found"
            # replace invalid
            header_euv.append(header_tmp.iloc[0].drop("date_diff"))
            segment_euv.append(segment_tmp.iloc[0].drop("date_diff"))

        queue = []
        queue += [(header_hmi.to_dict(), segment_hmi.magnetogram, date)]
        for h, s in zip(header_euv, segment_euv):
            queue += [(h.to_dict(), s.image, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)

        logging.info("Finished: %s" % id)


def download_sdo_data(
    start_date: str = "2022-3-1",
    end_date: str = "2023-3-2",
    email: Optional[str] = None,
    base_path: Optional[str] = None,
    n_workers: int = 8,
):
    if base_path is None:
        base_path = os.path.join(os.path.expanduser("~"), "sdo-path")

    logger.info(f"BasePath: {base_path}")

    # check datetime object
    start_date: datetime = check_datetime_format(start_date)
    end_date: datetime = check_datetime_format(end_date)

    logger.info(f"Period: {start_date}-{end_date}")

    if email is None:
        email = os.getenv("SDO_EMAIL")
    logger.info(f"Email: {email}")
    downloader_sdo = SDODownloader(
        base_path=base_path, email=email, n_workers=n_workers
    )

    dates = [
        start_date + i * timedelta(hours=12)
        for i in range((end_date - start_date) // timedelta(hours=12))
    ]

    pbar = tqdm.tqdm(dates)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for idate in pbar:
            pbar.set_description(f"Date: {idate}")
            downloader_sdo.downloadDate(idate)


if __name__ == "__main__":
    typer.run(download_sdo_data)
