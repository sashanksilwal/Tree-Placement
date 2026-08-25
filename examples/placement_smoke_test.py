"""Run representative pixel and crown strategies on a synthetic neighbourhood."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_code.geospatial_preprocessing.osm_eligibility import REQUIRED_LAYERS
from study_code.tree_placement import PlacementConfig, run_city


def write(path: Path, data: np.ndarray) -> None:
    array = data[None] if data.ndim == 2 else data
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype="float32",
        crs="EPSG:32616",
        transform=from_origin(500000, 4500000, 1, 1),
    ) as dst:
        dst.write(array.astype("float32"))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tree-placement-") as directory:
        city = Path(directory)
        base = city / "output/base/output_folder/0_0"
        base.mkdir(parents=True)
        shape = (96, 96)
        dem = np.zeros(shape, dtype="float32")
        building = dem.copy()
        building[40:50, 40:50] = 12
        canopy = np.zeros(shape, dtype="float32")
        canopy[12:20, 12:20] = 8
        tsfc = np.full((24, *shape), 300, dtype="float32")
        tsfc[:, 20:78, 20:78] = 305
        tsfc[:, 58:80, 58:80] = 309
        svf = np.full(shape, 0.35, dtype="float32")
        svf[12:38, 52:84] = 0.8
        write(city / "DEM.tif", dem)
        write(city / "Building_DSM.tif", building)
        write(city / "CDSM.tif", canopy)
        write(city / "landcover.tif", np.ones(shape, dtype="float32"))
        write(base / "Tsfc_0_0.tif", tsfc)
        write(base / "SVF_0_0.tif", svf)
        osm_dir = city / "osm"
        osm_dir.mkdir()
        empty_layer = json.dumps({"type": "FeatureCollection", "features": []})
        for layer in REQUIRED_LAYERS:
            (osm_dir / f"{layer}.geojson").write_text(empty_layer)
        result = {}
        for geometry, strategy in (
            ("pixel", "random"),
            ("crown", "adaptive"),
            ("crown", "hotspot_spread"),
        ):
            config = PlacementConfig(
                dose_pp=1.0,
                analysis_buffer_px=8,
                placement_geometry=geometry,
            )
            key = f"{strategy}_{geometry}"
            result[key] = run_city(city, city / key, strategy, config)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
