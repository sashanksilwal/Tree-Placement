#!/usr/bin/env python3
"""Run reproducible physical and numerical sensitivity cases on one real AOI."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utherm import thermal_comfort
import utherm.coupled_pipeline as coupled_pipeline


CASES = {
    "baseline": {},
    "canyon_height_4m": {"canyon_height": 4.0},
    "canyon_height_12m": {"canyon_height": 12.0},
    "ventilation_0p075": {"ventilation_coefficient": 0.075},
    "ventilation_0p225": {"ventilation_coefficient": 0.225},
    "interior_291p15k": {"interior_temperature": 291.15},
    "interior_299p15k": {"interior_temperature": 299.15},
    "deep_ground_283p15k": {"ground_deep_temperature": 283.15},
    "deep_ground_293p15k": {"ground_deep_temperature": 293.15},
    "relaxation_0p35": {"relaxation": 0.35},
    "relaxation_0p95": {"relaxation": 0.95},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_output(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        values = dataset.read().astype(np.float64)
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise RuntimeError(f"invalid sensitivity output: {path}")
    return values


def _summary(values: np.ndarray) -> dict[str, object]:
    spatial_mean = np.nanmean(values, axis=(1, 2))
    return {
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "peak_spatial_mean": float(np.nanmax(spatial_mean)),
        "spatial_mean_by_band": [float(value) for value in spatial_mean],
    }


def _differences(values: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    difference = values - baseline
    return {
        "mean_difference": float(np.nanmean(difference)),
        "mean_absolute_difference": float(np.nanmean(np.abs(difference))),
        "maximum_absolute_difference": float(np.nanmax(np.abs(difference))),
        "maximum_absolute_spatial_mean_difference": float(
            np.nanmax(np.abs(np.nanmean(difference, axis=(1, 2))))
        ),
    }


def _run_case(
    name: str,
    changes: dict[str, float],
    *,
    inputs: Path,
    geometry: Path,
    work_root: Path,
    selected_date: str,
    spinup_cycles: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    work = work_root / name
    work.mkdir()
    original_bridge = coupled_pipeline.CoupledRadiationBridge
    instances = []

    class SensitivityBridge(original_bridge):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model.config = replace(self.model.config, **changes)
            instances.append(self)

    coupled_pipeline.CoupledRadiationBridge = SensitivityBridge
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        thermal_comfort(
            base_path=str(work),
            selected_date_str=selected_date,
            building_dsm_filename=str(inputs / "Building_DSM.tif"),
            dem_filename=str(inputs / "DEM.tif"),
            trees_filename=str(inputs / "Trees.tif"),
            landcover_filename=str(inputs / "landcover.tif"),
            tile_size=2000,
            overlap=20,
            use_own_met=True,
            own_met_file=str(inputs / "met_multiday.txt"),
            use_coupled_eb=True,
            coupled_geometry_path=str(geometry),
            coupled_spinup_max_cycles=spinup_cycles,
            coupled_strict_convergence=True,
            save_tmrt=True,
            save_lup=True,
            save_t_leaf=True,
            save_tair_canyon=True,
        )
    finally:
        coupled_pipeline.CoupledRadiationBridge = original_bridge
    if len(instances) != 1 or instances[0].last_spinup is None:
        raise RuntimeError(f"case {name} did not execute exactly one coupled tile")
    bridge = instances[0]
    spinup = bridge.last_spinup
    if not spinup.converged:
        raise RuntimeError(f"case {name} did not converge")
    output = work / "output_folder" / "0_0"
    arrays = {
        field: _read_output(output / f"{field}_0_0.tif")
        for field in ("TMRT", "Lup", "TLeaf", "TairCanyon")
    }
    report = {
        "status": "passed",
        "configuration_changes": changes,
        "elapsed_seconds": time.time() - started,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "spinup": {
            "cycles": spinup.cycles,
            "converged": spinup.converged,
            "maximum_temperature_drift_k": spinup.maximum_temperature_drift,
            "maximum_moisture_drift_kg_m2": spinup.maximum_moisture_drift,
            "maximum_specific_humidity_drift_kg_kg": (
                spinup.maximum_specific_humidity_drift
            ),
        },
        "outputs": {field: _summary(values) for field, values in arrays.items()},
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--selected-date", required=True)
    parser.add_argument("--spinup-cycles", type=int, default=80)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    inputs = args.inputs.expanduser().resolve()
    geometry = args.geometry.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    required = [
        inputs / name
        for name in (
            "DEM.tif",
            "Building_DSM.tif",
            "Trees.tif",
            "landcover.tif",
            "met_multiday.txt",
        )
    ] + [geometry]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("one or more sensitivity inputs are missing")
    if work_root.exists():
        raise FileExistsError(f"work root already exists: {work_root}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sensitivity run")
    work_root.mkdir(parents=True)
    reports = {}
    arrays = {}
    for name, changes in CASES.items():
        try:
            reports[name], arrays[name] = _run_case(
                name,
                changes,
                inputs=inputs,
                geometry=geometry,
                work_root=work_root,
                selected_date=args.selected_date,
                spinup_cycles=args.spinup_cycles,
            )
        except Exception as exc:
            reports[name] = {
                "status": "failed",
                "configuration_changes": changes,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(json.dumps({"failed": name, "error": str(exc)}), flush=True)
        else:
            print(
                json.dumps({"completed": name, "spinup": reports[name]["spinup"]}),
                flush=True,
            )
    if "baseline" not in arrays:
        raise RuntimeError("baseline sensitivity case failed; comparisons are undefined")
    baseline = arrays["baseline"]
    for name in arrays:
        reports[name]["difference_from_baseline"] = {
            field: _differences(arrays[name][field], baseline[field])
            for field in baseline
        }
    report = {
        "scenario": "real-AOI coupled physical and numerical sensitivity",
        "inputs": {path.name: str(path) for path in required[:-1]},
        "input_sha256": {path.name: _sha256(path) for path in required[:-1]},
        "geometry": str(geometry),
        "geometry_sha256": _sha256(geometry),
        "selected_date": args.selected_date,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cases": reports,
    }
    report_path = work_root / "QA_REAL_AOI_SENSITIVITY.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "cases": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
