#!/usr/bin/env python3
# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utherm.view_factors import ViewFactorConfig, trace_geometry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_aligned(
    paths: list[Path],
    read_window: Window,
    zero_fill_nodata_indices: frozenset[int] = frozenset(),
) -> tuple[list[np.ndarray], object]:
    arrays = []
    reference = None
    for index, path in enumerate(paths):
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(f"{path} must contain one band")
            signature = (dataset.width, dataset.height, dataset.crs, tuple(dataset.transform))
            if reference is None:
                reference = signature
                crs = dataset.crs
            elif signature != reference:
                raise ValueError(f"raster grid mismatch: {path}")
            masked = dataset.read(1, window=read_window, masked=True)
            zero_nodata = (
                index in zero_fill_nodata_indices
                and dataset.nodata is not None
                and np.isclose(dataset.nodata, 0.0)
            )
            fill_value = 0.0 if zero_nodata else np.nan
            array = masked.filled(fill_value).astype(np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"{path} contains nodata or non-finite cells")
            arrays.append(array)
    return arrays, crs


def _window(value: str | None, rows: int, cols: int) -> tuple[int, int, int, int]:
    if value is None:
        return 0, 0, rows, cols
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--window must be row,column,height,width") from exc
    if len(values) != 4:
        raise ValueError("--window must be row,column,height,width")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a ray-traced coupled geometry bundle")
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--building-dsm", type=Path, required=True)
    parser.add_argument("--canopy-height", type=Path, required=True)
    parser.add_argument("--landcover", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", help="row,column,height,width; default is the complete raster")
    parser.add_argument("--max-distance", type=float, default=60.0)
    parser.add_argument("--ray-step", type=float, default=0.5)
    parser.add_argument("--surface-rays", type=int, default=128)
    parser.add_argument("--body-rays", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = [
        args.dem.resolve(),
        args.building_dsm.resolve(),
        args.canopy_height.resolve(),
        args.landcover.resolve(),
    ]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing input raster(s): {missing}")
    output = args.output.resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite to replace it")
    with rasterio.open(paths[0]) as reference_dataset:
        rows, cols = reference_dataset.height, reference_dataset.width
        transform = reference_dataset.transform
        if reference_dataset.crs is None or not reference_dataset.crs.is_projected:
            raise ValueError("input rasters must use a projected CRS")
        if abs(float(transform.b)) > 1.0e-12 or abs(float(transform.d)) > 1.0e-12:
            raise ValueError("rotated or sheared raster grids are not supported")
        if float(transform.a) <= 0.0 or float(transform.e) >= 0.0:
            raise ValueError("input rasters must use a north-up transform")
        if not np.isclose(abs(float(transform.a)), abs(float(transform.e)), atol=1.0e-9):
            raise ValueError("input raster pixels must be square")
        pixel_size = abs(float(transform.a))
    source_window = _window(args.window, rows, cols)
    source_row, source_col, source_height, source_width = source_window
    if (
        source_row < 0
        or source_col < 0
        or source_height < 1
        or source_width < 1
        or source_row + source_height > rows
        or source_col + source_width > cols
    ):
        raise ValueError("--window extends outside the raster or has an invalid size")
    config = ViewFactorConfig(
        pixel_size_m=pixel_size,
        max_distance_m=args.max_distance,
        ray_step_m=args.ray_step,
        surface_ray_count=args.surface_rays,
        body_ray_count=args.body_rays,
    )
    config.validate()
    halo = int(np.ceil(config.max_distance_m / pixel_size)) + 2
    read_row = max(0, source_row - halo)
    read_col = max(0, source_col - halo)
    read_end_row = min(rows, source_row + source_height + halo)
    read_end_col = min(cols, source_col + source_width + halo)
    read_window = Window(
        read_col,
        read_row,
        read_end_col - read_col,
        read_end_row - read_row,
    )
    arrays, crs = _read_aligned(
        paths,
        read_window,
        zero_fill_nodata_indices=frozenset({2}),
    )
    dem, building_dsm, canopy_height, landcover_float = arrays
    if not np.equal(landcover_float, np.rint(landcover_float)).all():
        raise ValueError("land-cover values must be integers")
    landcover = np.rint(landcover_float).astype(np.int16)
    if np.any((landcover < 1) | (landcover > 7)):
        raise ValueError("land-cover values must use UTherm classes 1 through 7")
    if np.isin(landcover, (3, 4)).any():
        raise ValueError(
            "land-cover classes 3 and 4 describe canopy; derive the ground-cover "
            "class beneath vegetation first"
        )
    local_window = (
        source_row - read_row,
        source_col - read_col,
        source_height,
        source_width,
    )
    result = trace_geometry(
        dem,
        building_dsm,
        canopy_height,
        window=local_window,
        config=config,
    )
    _, _, height, width = result.window
    output_transform = rasterio.windows.transform(
        Window(source_col, source_row, source_width, source_height), transform
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        area=result.area,
        sky_view_area=result.sky_view_area,
        exchange_area=result.exchange_area,
        body_view_factor=result.body_view_factor,
        body_sky_view_factor=result.body_sky_view_factor,
        raw_surface_sky_view_factor=result.raw_surface_sky_view_factor,
        facet_origin_x=result.facet_origin_x,
        facet_origin_y=result.facet_origin_y,
        facet_origin_z=result.facet_origin_z,
        trace_dem=dem.astype(np.float32),
        trace_building_dsm=np.maximum(building_dsm, dem).astype(np.float32),
        trace_canopy_height=canopy_height.astype(np.float32),
        trace_landcover=landcover,
        trace_output_window=np.asarray(local_window, dtype=np.int64),
        crs_wkt=np.asarray(crs.to_wkt()),
        geotransform=np.asarray(output_transform.to_gdal(), dtype=np.float64),
        source_window=np.asarray(source_window, dtype=np.int64),
        dsm_clamped_cell_count=np.asarray(result.dsm_clamped_cell_count, dtype=np.int64),
        maximum_dsm_clamp_m=np.asarray(result.maximum_dsm_clamp_m, dtype=np.float64),
        representation=np.asarray("local reciprocal seven-facet enclosure derived from height-field ray tracing"),
        config_json=np.asarray(json.dumps(config.__dict__, sort_keys=True)),
        input_sha256_json=np.asarray(json.dumps({path.name: _sha256(path) for path in paths}, sort_keys=True)),
    )
    active = result.area > 0.0
    reciprocal_sky_factor = np.divide(
        result.sky_view_area,
        result.area,
        out=np.ones_like(result.sky_view_area),
        where=active,
    )
    sky_adjustment = np.abs(
        reciprocal_sky_factor[active] - result.raw_surface_sky_view_factor[active]
    )
    area_weighted_sky_adjustment = float(
        (
            np.abs(reciprocal_sky_factor - result.raw_surface_sky_view_factor)
            * result.area
        ).sum()
        / result.area.sum()
    )
    report = {
        "output": str(output),
        "shape": [height, width],
        "window": list(source_window),
        "read_window": [
            int(read_window.row_off),
            int(read_window.col_off),
            int(read_window.height),
            int(read_window.width),
        ],
        "crs": str(crs),
        "maximum_reciprocity_error": float(
            np.max(np.abs(result.exchange_area - result.exchange_area.swapaxes(0, 1)))
        ),
        "maximum_closure_error": float(
            np.max(
                np.abs(
                    result.sky_view_area
                    + result.exchange_area.sum(axis=1)
                    - result.area
                )
            )
        ),
        "maximum_body_closure_error": float(
            np.max(
                np.abs(
                    result.body_view_factor.sum(axis=0)
                    + result.body_sky_view_factor
                    - 1.0
                )
            )
        ),
        "mean_body_sky_view_factor": float(result.body_sky_view_factor.mean()),
        "dsm_clamped_cell_count": result.dsm_clamped_cell_count,
        "maximum_dsm_clamp_m": result.maximum_dsm_clamp_m,
        "mean_reciprocal_sky_factor_adjustment": float(sky_adjustment.mean()),
        "area_weighted_reciprocal_sky_factor_adjustment": area_weighted_sky_adjustment,
        "maximum_reciprocal_sky_factor_adjustment": float(sky_adjustment.max()),
        "active_fraction_by_facet": [
            float((result.area[index] > 0.0).mean()) for index in range(result.area.shape[0])
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
