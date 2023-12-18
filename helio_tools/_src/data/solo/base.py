"""
Script to download data products from the SolO database using Fido, a sunpy module meant for retrieving data from solar missions.

In this case it is downloading images from the EUI (extreme ultraviolet imager) from the Solar Orbiter (SolO) mission.

"""
from typing import Optional
import argparse
import logging
import os
import shutil
from datetime import timedelta, datetime
from multiprocessing import Pool
from urllib.request import urlopen
from warnings import simplefilter
from random import sample

import drms
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io.fits import getheader, HDUList
from dateutil.relativedelta import relativedelta
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
import sunpy_soar
from tqdm import tqdm
from loguru import logger
import typer
from helio_tools._src.utils.time import (
    check_datetime_format,
    get_num_months,
    get_month_dates,
)


DEFAULT_WAVELENGTHS = ["eui-hri174-image", "eui-fsi174-image", "eui-fsi304-image"]


class SOLODownloader:
    def __init__(
        self,
        base_path: str = None,
        wavelengths: list[str | int | float] = DEFAULT_WAVELENGTHS,
    ) -> None:
        self.base_path = base_path
        self.wavelengths = wavelengths

        # create directories if they don't exist for each wavelength
        [
            os.makedirs(os.path.join(base_path, wl), exist_ok=True)
            for wl in self.wavelengths
        ]

    def downloadDate(self, date: datetime, download_hri: bool = False):
        """Download FITS data for a specific date.

        Args:
            date (datetime): date to download
            download_hri (bool): whether to download HRI data or not

        Example Usage:

                >>> downloader_solo = SOLODownloader(...)
                >>> downloader_solo.downloadDate(datetime(2022, 3, 1))

        """
        files = []
        try:
            # Download FSI Sensor data
            for wl in self.wavelengths[1::]:
                files += [self.downloadFSI(date, wl)]

            # Download HRI Sensor data
            if download_hri:
                for wl in self.wavelengths[0]:
                    files += [self.downloadHRI(date, wl)]
            logging.info("Download complete %s" % date.isoformat())
        except Exception as ex:
            logging.error("Unable to download %s: %s" % (date.isoformat(), str(ex)))
            [os.remove(f) for f in files if os.path.exists(f)]

    def downloadFSI(self, query_date, wl):
        """
        Download FSI (full sun) data for a specific date.

        Args:
            query_date (datetime): date to download
            wl (str): wavelength to download
        Returns:
            file_path (str): path to downloaded file

        Example Usage:

            >>> downloader_solo = SOLODownloader(...)
            >>> downloader_solo.downloadFSI(datetime(2022, 3, 1), 'eui-fsi174-image')

        """
        file_path = os.path.join(
            self.base_path,
            wl,
            "%s.fits" % query_date.isoformat("T", timespec="seconds"),
        )
        if os.path.exists(file_path):
            return file_path

        search = Fido.search(
            a.Time(
                query_date - timedelta(minutes=15), query_date + timedelta(minutes=15)
            ),
            a.Instrument("EUI"),
            a.soar.Product(wl),
            a.Level(2),
        )
        assert search.file_num > 0, "No data found for %s (%s)" % (
            query_date.isoformat(),
            wl,
        )
        search = sorted(
            search["soar"],
            key=lambda x: abs(
                pd.to_datetime(x["Start time"]) - query_date
            ).total_seconds(),
        )

        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]

            # Clean data with header info or add printing meta data info
            header = getheader(file, 1)
            if header["CDELT1"] != 4.44012445:
                os.remove(file)
                continue

            shutil.move(file, file_path)
            return file_path

        raise Exception(
            "No valid file found for %s (%s)!" % (query_date.isoformat(), wl)
        )

    def downloadHRI(self, query_date, wl):
        """
        Currently unused, but retriever for HRI (high res) data
        """
        file_path = os.path.join(
            self.base_path,
            wl,
            "%s.fits" % query_date.isoformat("T", timespec="seconds"),
        )
        if os.path.exists(file_path):
            return file_path
        #
        search = Fido.search(
            a.Time(
                query_date + timedelta(minutes=15), query_date + timedelta(minutes=15)
            ),
            a.Instrument("EUI"),
            a.soar.Product(wl),
            a.Level(2),
        )
        assert search.file_num > 0, "No data found for %s (%s)" % (
            query_date.isoformat(),
            wl,
        )
        search = sorted(
            search["soar"],
            key=lambda x: abs(
                pd.to_datetime(x["Start Time"]) - query_date
            ).total_seconds(),
        )
        #
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]
            # header = Map(file.meta)
            shutil.move(file, file_path)
            return file_path

        raise Exception(
            "No valid file found for %s (%s)!" % (query_date.isoformat(), wl)
        )


def download_solo_data(
    start_date: str = "2021-02-22 00:00",
    end_date: str = "2021-06-2 00:00",
    base_path: Optional[str] = None,
):
    start_date = check_datetime_format(start_date, sensor="solo")
    end_date = check_datetime_format(end_date, sensor="solo")
    logger.info(f"Period: {start_date} --- {end_date}")
    if base_path is None:
        base_path = os.getcwd() + "/solo-data"

    solo_downloader = SOLODownloader(base_path=base_path)

    logger.info(f"BasePath: {base_path}")

    num_months = get_num_months(start_date, end_date)
    logger.info(f"Number of Months: {num_months}")
    month_dates = get_month_dates(start_date, num_months)
    logger.info(f"Number of Month Dates: {len(month_dates)}")

    pbar = tqdm(month_dates)

    for idate in pbar:
        pbar.set_description(f"Date: {idate}")
        search = Fido.search(
            a.Time(idate, idate + relativedelta(days=1)),
            a.Instrument("EUI"),
            a.soar.Product("eui-fsi174-image"),
            a.Level(2),
        )
        if search.file_num == 0:
            continue
        dates = search["soar"]["Start time"]
        dates = pd.to_datetime(dates)
        step = int(np.floor(len(dates) / 60)) if len(dates) > 60 else 1

        spbar = tqdm(dates[::step], leave=False)

        for isubdate in spbar:
            spbar.set_description(f"Date: {isubdate}")
            solo_downloader.downloadDate(isubdate)


if __name__ == "__main__":
    typer.run(download_solo_data)
