#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "area",
            "sky_view_area",
            "exchange_area",
            "body_view_factor",
            "body_sky_view_factor",
            "raw_surface_sky_view_factor",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _absolute_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    return {
        "mean_absolute_difference": float(difference.mean()),
        "p95_absolute_difference": float(np.percentile(difference, 95.0)),
        "maximum_absolute_difference": float(difference.max()),
    }


def _bundle_checks(bundle: dict[str, np.ndarray]) -> dict[str, object]:
    area = bundle["area"]
    exchange = bundle["exchange_area"]
    sky = bundle["sky_view_area"]
    body = bundle["body_view_factor"]
    body_sky = bundle["body_sky_view_factor"]
    active = area > 0.0
    reciprocal_sky = np.divide(sky, area, out=np.ones_like(sky), where=active)
    sky_adjustment = np.abs(reciprocal_sky[active] - bundle["raw_surface_sky_view_factor"][active])
    area_weighted_adjustment = float(
        (np.abs(reciprocal_sky - bundle["raw_surface_sky_view_factor"]) * area).sum()
        / area.sum()
    )
    return {
        "shape": list(area.shape[1:]),
        "maximum_reciprocity_error": float(np.max(np.abs(exchange - exchange.swapaxes(0, 1)))),
        "maximum_closure_error": float(np.max(np.abs(sky + exchange.sum(axis=1) - area))),
        "maximum_body_closure_error": float(np.max(np.abs(body.sum(axis=0) + body_sky - 1.0))),
        "mean_body_sky_view_factor": float(body_sky.mean()),
        "mean_reciprocal_sky_factor_adjustment": float(sky_adjustment.mean()),
        "area_weighted_reciprocal_sky_factor_adjustment": area_weighted_adjustment,
        "p99_reciprocal_sky_factor_adjustment": float(np.percentile(sky_adjustment, 99.0)),
        "maximum_reciprocal_sky_factor_adjustment": float(sky_adjustment.max()),
        "fraction_active_facets_adjusted_over_0_05": float((sky_adjustment > 0.05).mean()),
    }


def _convergence(
    candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, object]:
    if candidate["area"].shape != reference["area"].shape:
        raise ValueError("candidate and reference bundle shapes differ")
    np.testing.assert_allclose(candidate["area"], reference["area"], atol=1.0e-6, rtol=0.0)
    area = candidate["area"]
    active = area > 0.0
    candidate_sky = np.divide(
        candidate["sky_view_area"], area, out=np.ones_like(area), where=active
    )
    reference_sky = np.divide(
        reference["sky_view_area"], area, out=np.ones_like(area), where=active
    )
    candidate_view = np.divide(
        candidate["exchange_area"],
        area[:, None],
        out=np.zeros_like(candidate["exchange_area"]),
        where=area[:, None] > 0.0,
    )
    reference_view = np.divide(
        reference["exchange_area"],
        area[:, None],
        out=np.zeros_like(reference["exchange_area"]),
        where=area[:, None] > 0.0,
    )
    active_exchange = np.broadcast_to(active[:, None], candidate_view.shape)
    return {
        "body_view_factor": _absolute_metrics(
            candidate["body_view_factor"], reference["body_view_factor"]
        ),
        "body_sky_view_factor": _absolute_metrics(
            candidate["body_sky_view_factor"], reference["body_sky_view_factor"]
        ),
        "raw_surface_sky_view_factor": _absolute_metrics(
            candidate["raw_surface_sky_view_factor"][active],
            reference["raw_surface_sky_view_factor"][active],
        ),
        "reciprocal_surface_sky_view_factor": _absolute_metrics(
            candidate_sky[active], reference_sky[active]
        ),
        "surface_exchange_view_factor": _absolute_metrics(
            candidate_view[active_exchange],
            reference_view[active_exchange],
        ),
        "surface_exchange_area": _absolute_metrics(
            candidate["exchange_area"], reference["exchange_area"]
        ),
    }


def _svf_metrics(traced: np.ndarray, svf: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = mask & np.isfinite(svf) & np.isfinite(traced) & (svf >= 0.0) & (svf <= 1.0)
    if not bool(valid.any()):
        raise ValueError("no valid SOLWEIG SVF cells remain after masking")
    difference = traced[valid] - svf[valid]
    return {
        "cells": int(valid.sum()),
        "traced_mean": float(traced[valid].mean()),
        "solweig_mean": float(svf[valid].mean()),
        "mean_bias": float(difference.mean()),
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "spatial_spearman": float(spearmanr(traced[valid], svf[valid]).statistic),
    }


def _solweig_svf(
    bundle: dict[str, np.ndarray],
    path: Path,
    dem_path: Path | None,
    building_path: Path | None,
    canopy_path: Path | None,
    building_threshold_m: float,
    canopy_threshold_m: float,
) -> dict[str, object]:
    import rasterio
    from rasterio.windows import Window

    if "source_window" not in bundle:
        raise ValueError("bundle does not record source_window")
    row, col, height, width = (int(value) for value in bundle["source_window"])
    with rasterio.open(path) as dataset:
        svf = dataset.read(1, window=Window(col, row, width, height)).astype(np.float64)
    traced = bundle["raw_surface_sky_view_factor"][0].astype(np.float64)
    result: dict[str, object] = {
        "comparison_definition": (
            "The saved SOLWEIG SVF is building SVF corrected with fixed 3% tree "
            "transmission. The tracer uses LAI-based Beer-Lambert transmission; "
            "open ground without canopy is the like-for-like geometry comparison."
        ),
        "all_cells": _svf_metrics(traced, svf, np.ones_like(svf, dtype=bool))
    }
    if (dem_path is None) != (building_path is None):
        raise ValueError("--dem and --building-dsm must be supplied together")
    if dem_path is not None and building_path is not None:
        with rasterio.open(dem_path) as dataset:
            dem = dataset.read(1, window=Window(col, row, width, height)).astype(np.float64)
        with rasterio.open(building_path) as dataset:
            building = dataset.read(1, window=Window(col, row, width, height)).astype(np.float64)
        open_ground = building - dem < building_threshold_m
        result["open_ground"] = _svf_metrics(traced, svf, open_ground)
        if canopy_path is not None:
            with rasterio.open(canopy_path) as dataset:
                canopy = dataset.read(1, window=Window(col, row, width, height)).astype(np.float64)
            canopy_present = canopy >= canopy_threshold_m
            result["open_ground_without_canopy"] = _svf_metrics(
                traced, svf, open_ground & ~canopy_present
            )
            result["open_ground_beneath_canopy"] = _svf_metrics(
                traced, svf, open_ground & canopy_present
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a coupled geometry bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--reference-bundle", type=Path)
    parser.add_argument("--solweig-svf", type=Path)
    parser.add_argument("--dem", type=Path)
    parser.add_argument("--building-dsm", type=Path)
    parser.add_argument("--canopy-height", type=Path)
    parser.add_argument("--building-threshold", type=float, default=2.0)
    parser.add_argument("--canopy-threshold", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = _load(args.bundle)
    report: dict[str, object] = {
        "bundle": str(args.bundle.resolve()),
        "checks": _bundle_checks(candidate),
    }
    if args.reference_bundle is not None:
        reference = _load(args.reference_bundle)
        report["reference_bundle"] = str(args.reference_bundle.resolve())
        report["ray_count_convergence"] = _convergence(candidate, reference)
    if args.solweig_svf is not None:
        report["solweig_ground_sky_view_comparison"] = _solweig_svf(
            candidate,
            args.solweig_svf,
            args.dem,
            args.building_dsm,
            args.canopy_height,
            args.building_threshold,
            args.canopy_threshold,
        )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"output exists: {output}")
        output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
