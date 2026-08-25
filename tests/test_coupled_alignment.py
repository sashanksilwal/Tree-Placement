# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from types import SimpleNamespace

import pytest

osgeo = pytest.importorskip("osgeo", reason="alignment check needs GDAL")
from utherm.utci_process import _validate_coupled_geometry_alignment


WKT_32611 = 'PROJCS["WGS 84 / UTM zone 11N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-117],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'
WKT_32612 = 'PROJCS["WGS 84 / UTM zone 12N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-111],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'


class _Raster:
    def __init__(self, projection, transform):
        self.projection = projection
        self.transform = transform

    def GetProjection(self):
        return self.projection

    def GetGeoTransform(self):
        return self.transform


def test_coupled_geometry_alignment_accepts_exact_grid():
    transform = (660000.0, 1.0, 0.0, 4000000.0, 0.0, -1.0)
    bundle = SimpleNamespace(
        crs_wkt=WKT_32611,
        geotransform=transform,
        source_path=Path("geometry.npz"),
    )
    _validate_coupled_geometry_alignment(bundle, _Raster(WKT_32611, transform))


def test_coupled_geometry_alignment_rejects_crs_and_transform_mismatch():
    transform = (660000.0, 1.0, 0.0, 4000000.0, 0.0, -1.0)
    bundle = SimpleNamespace(
        crs_wkt=WKT_32612,
        geotransform=transform,
        source_path=Path("geometry.npz"),
    )
    raster = _Raster(WKT_32611, transform)
    with pytest.raises(ValueError, match="CRS does not match"):
        _validate_coupled_geometry_alignment(bundle, raster)
    bundle.crs_wkt = WKT_32611
    bundle.geotransform = (660001.0, 1.0, 0.0, 4000000.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="transform does not match"):
        _validate_coupled_geometry_alignment(bundle, raster)
