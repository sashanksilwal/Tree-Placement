#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utherm.coupled_pipeline import (
    CoupledRadiationBridge,
    ground_material_ids_from_trace,
    load_geometry_bundle,
)
from utherm.energy_balance import CanopyProperties, MaterialProperties


LABELS = (
    "ground",
    "roof",
    "north_wall",
    "east_wall",
    "south_wall",
    "west_wall",
    "canopy",
)


def _summary(field: torch.Tensor, area: torch.Tensor) -> dict[str, float]:
    active = area > 0.0
    values = field[active]
    weights = area[active]
    return {
        "active_cells": int(active.sum().item()),
        "minimum_w_m2_surface": float(values.min().item()),
        "maximum_w_m2_surface": float(values.max().item()),
        "area_weighted_mean_w_m2_surface": float(
            (values * weights).sum().item() / weights.sum().item()
        ),
        "mean_w_per_m2_plan": float((field * area).mean().item()),
    }


def _plot(fields: np.ndarray, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(12.5, 6.2), constrained_layout=True)
    maximum = max(float(np.percentile(fields, 99.0)), 1.0)
    for axis, field, label in zip(axes.flat, fields, LABELS):
        image = axis.imshow(field, vmin=0.0, vmax=maximum, cmap="inferno")
        axis.set_title(f"{label.replace('_', ' ').title()}\nmean={field.mean():.1f} W m$^{{-2}}$")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    axes.flat[-1].axis("off")
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _directional_sweep(
    bridge: CoupledRadiationBridge,
    area: torch.Tensor,
    direct_normal: float,
    diffuse_horizontal: float,
) -> dict[str, object]:
    records = []
    for altitude in (5.0, 15.0, 30.0, 60.0, 85.0):
        for azimuth in range(0, 360, 45):
            field = bridge.facet_shortwave_irradiance(
                direct_normal_irradiance=direct_normal,
                diffuse_horizontal_irradiance=diffuse_horizontal,
                solar_altitude_degrees=altitude,
                solar_azimuth_degrees=float(azimuth),
            )
            plan_flux = float((field * area).sum(dim=0).mean().item())
            target = direct_normal * math.sin(math.radians(altitude)) + diffuse_horizontal
            records.append(
                {
                    "solar_altitude_degrees": altitude,
                    "solar_azimuth_degrees": azimuth,
                    "closure_error_w_m2_plan": plan_flux - target,
                    "maximum_surface_flux_w_m2": float(field.max().item()),
                }
            )
    worst = max(records, key=lambda item: item["maximum_surface_flux_w_m2"])
    return {
        "cases": len(records),
        "maximum_absolute_closure_error_w_m2_plan": max(
            abs(item["closure_error_w_m2_plan"]) for item in records
        ),
        "maximum_surface_flux_w_m2": worst["maximum_surface_flux_w_m2"],
        "direct_plus_diffuse_upper_bound_w_m2": direct_normal + diffuse_horizontal,
        "worst_case": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate traced facet shortwave on a geometry bundle"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--altitude", type=float, required=True)
    parser.add_argument("--azimuth", type=float, required=True)
    parser.add_argument("--direct-normal", type=float, required=True)
    parser.add_argument("--diffuse-horizontal", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--sweep-cardinals", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    with np.load(args.bundle, allow_pickle=False) as archive:
        rows, cols = archive["body_sky_view_factor"].shape
    device = torch.device(args.device)
    bundle = load_geometry_bundle(args.bundle, rows, cols, device)
    bridge = CoupledRadiationBridge(
        bundle,
        ground_material=MaterialProperties.asphalt(),
        roof_material=MaterialProperties.roof(),
        canopy_properties=CanopyProperties.deciduous(),
        dt=3600.0,
        spinup_max_cycles=2,
        strict_convergence=False,
    )
    trace = bundle.solar_trace
    if trace is None:
        raise ValueError("bundle lacks solar-trace geometry")
    row, col, height, width = trace.output_window
    tile_landcover = trace.landcover[row : row + height, col : col + width]
    ground_material_ids = ground_material_ids_from_trace(bundle, tile_landcover)
    classes, counts = np.unique(ground_material_ids, return_counts=True)
    with torch.no_grad():
        direct = bridge.facet_shortwave_irradiance(
            direct_normal_irradiance=args.direct_normal,
            diffuse_horizontal_irradiance=0.0,
            solar_altitude_degrees=args.altitude,
            solar_azimuth_degrees=args.azimuth,
        )
        diffuse = bridge.facet_shortwave_irradiance(
            direct_normal_irradiance=0.0,
            diffuse_horizontal_irradiance=args.diffuse_horizontal,
            solar_altitude_degrees=args.altitude,
            solar_azimuth_degrees=args.azimuth,
        )
        total = direct + diffuse
    report = {
        "bundle": args.bundle.name,
        "device": args.device,
        "shape": [rows, cols],
        "method": (
            "facet-normal direct beam with height-field obstruction, isotropic diffuse sky exposure, "
            "Beer-Lambert canopy interception and bounded domain-integrated plan-area conservation"
        ),
        "forcing": {
            "solar_altitude_degrees": args.altitude,
            "solar_azimuth_degrees": args.azimuth,
            "direct_normal_irradiance_w_m2": args.direct_normal,
            "diffuse_horizontal_irradiance_w_m2": args.diffuse_horizontal,
        },
        "ground_material_provenance": {
            "representative_origin_class_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(classes, counts)
            },
            "cells_whose_ground_origin_class_differs_from_plan_cell": int(
                np.count_nonzero(ground_material_ids != tile_landcover)
            ),
        },
        "domain_energy_closure": {
            "direct_w_per_m2_plan": float(
                (direct * bundle.geometry.area).sum(dim=0).mean().item()
            ),
            "diffuse_w_per_m2_plan": float(
                (diffuse * bundle.geometry.area).sum(dim=0).mean().item()
            ),
            "total_w_per_m2_plan": float(
                (total * bundle.geometry.area).sum(dim=0).mean().item()
            ),
        },
        "facets": {
            label: {
                "direct": _summary(direct[index], bundle.geometry.area[index]),
                "diffuse": _summary(diffuse[index], bundle.geometry.area[index]),
                "total": _summary(total[index], bundle.geometry.area[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }
    if args.sweep_cardinals:
        report["directional_sweep"] = _directional_sweep(
            bridge,
            bundle.geometry.area,
            args.direct_normal,
            args.diffuse_horizontal,
        )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.plot is not None:
        _plot(
            total.cpu().numpy(),
            args.plot,
            f"Facet irradiance: altitude {args.altitude:g}°, azimuth {args.azimuth:g}°",
        )
    print(text)


if __name__ == "__main__":
    main()
