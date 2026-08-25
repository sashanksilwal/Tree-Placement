#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_ground_cover(
    landcover: np.ndarray,
    pixel_size_m: float,
    vegetation_classes: tuple[int, ...] = (3, 4),
    observed_ground_classes: tuple[int, ...] = (1, 5, 6, 7),
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(landcover)
    if values.ndim != 2:
        raise ValueError("land cover must be two-dimensional")
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("land cover must contain finite integer classes")
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0.0:
        raise ValueError("pixel_size_m must be finite and positive")
    classes = np.rint(values).astype(np.int16)
    vegetation = np.isin(classes, vegetation_classes)
    observed_ground = np.isin(classes, observed_ground_classes)
    if vegetation.any() and not observed_ground.any():
        raise ValueError("no observed ground class is available to fill vegetation")
    output = classes.copy()
    distance = np.zeros(classes.shape, dtype=np.float64)
    if vegetation.any():
        nearest_distance, nearest_indices = distance_transform_edt(
            ~observed_ground,
            sampling=pixel_size_m,
            return_indices=True,
        )
        output[vegetation] = classes[
            nearest_indices[0][vegetation],
            nearest_indices[1][vegetation],
        ]
        distance[vegetation] = nearest_distance[vegetation]
    return output, distance


def main() -> None:
    import rasterio

    parser = argparse.ArgumentParser(
        description="Replace UMEP canopy classes with the nearest observed ground class"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite to replace it")
    with rasterio.open(source) as dataset:
        if dataset.count != 1:
            raise ValueError("land-cover raster must contain one band")
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError("land-cover raster must use a projected CRS")
        if not np.isclose(abs(dataset.transform.a), abs(dataset.transform.e), atol=1.0e-9):
            raise ValueError("land-cover pixels must be square")
        values = dataset.read(1)
        profile = dataset.profile.copy()
        pixel_size = abs(float(dataset.transform.a))
    derived, fill_distance = derive_ground_cover(values, pixel_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="int16", nodata=None)
    with rasterio.open(output, "w", **profile) as dataset:
        dataset.write(derived, 1)
        dataset.update_tags(
            derivation="nearest observed ground class beneath UMEP canopy classes 3 and 4",
            observed_ground_classes="1,5,6,7",
            source_sha256=_sha256(source),
        )
    filled = fill_distance > 0.0
    report = {
        "input": str(source),
        "output": str(output),
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "filled_cells": int(filled.sum()),
        "maximum_fill_distance_m": float(fill_distance.max()),
        "p95_fill_distance_m": (
            float(np.percentile(fill_distance[filled], 95.0)) if filled.any() else 0.0
        ),
        "output_class_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(derived, return_counts=True))
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
