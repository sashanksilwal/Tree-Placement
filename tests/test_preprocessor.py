# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Failure-path tests for meteorological preprocessing."""

import datetime

import pytest

netcdf4 = pytest.importorskip("netCDF4")

from utherm.preprocessor import _select_wrf_files, process_metfiles


def _empty_netcdf(path):
    with netcdf4.Dataset(path, "w"):
        pass


def test_process_metfiles_requires_raster_tiles(tmp_path):
    forcing = tmp_path / "forcing.nc"
    _empty_netcdf(forcing)
    rasters = tmp_path / "rasters"
    rasters.mkdir()
    with pytest.raises(FileNotFoundError, match="no GeoTIFF tiles"):
        process_metfiles(forcing, rasters, tmp_path, "2023-07-15", tmp_path / "processed")


def test_process_metfiles_reports_missing_forcing_variables(tmp_path):
    forcing = tmp_path / "forcing.nc"
    _empty_netcdf(forcing)
    rasters = tmp_path / "rasters"
    rasters.mkdir()
    (rasters / "DEM_0_0.tif").touch()
    with pytest.raises(ValueError, match="missing variables"):
        process_metfiles(forcing, rasters, tmp_path, "2023-07-15", tmp_path / "processed")


def test_wrf_selection_requires_one_domain(tmp_path):
    (tmp_path / "wrfout_d01_2023-07-15_12").touch()
    (tmp_path / "wrfout_d02_2023-07-15_12").touch()
    start = end = datetime.datetime(2023, 7, 15, 12)
    with pytest.raises(ValueError, match="one domain"):
        _select_wrf_files(tmp_path, start, end)


def test_wrf_selection_requires_every_requested_hour(tmp_path):
    (tmp_path / "wrfout_d01_2023-07-15_12").touch()
    (tmp_path / "wrfout_d01_2023-07-15_14").touch()
    start = datetime.datetime(2023, 7, 15, 12)
    end = datetime.datetime(2023, 7, 15, 14)
    with pytest.raises(ValueError, match="missing=.*13:00:00"):
        _select_wrf_files(tmp_path, start, end)


def test_wrf_selection_excludes_files_outside_interval(tmp_path):
    for hour in (11, 12, 13, 14):
        (tmp_path / f"wrfout_d01_2023-07-15_{hour:02d}").touch()
    start = datetime.datetime(2023, 7, 15, 12)
    end = datetime.datetime(2023, 7, 15, 13)
    files, times = _select_wrf_files(tmp_path, start, end)
    assert files == [
        "wrfout_d01_2023-07-15_12",
        "wrfout_d01_2023-07-15_13",
    ]
    assert times == [start, end]
