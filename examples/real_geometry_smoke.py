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

from utherm.coupled_pipeline import CoupledRadiationBridge, load_geometry_bundle
from utherm.energy_balance import CanopyProperties, MaterialProperties
from utherm.energy_balance.coupled import UrbanForcing


def _forcing_cycle(bundle, bridge):
    forcings = []
    body_shortwave = []
    for hour in range(24):
        solar = max(math.sin(math.pi * (hour - 6.0) / 12.0), 0.0)
        altitude = 60.0 * solar
        azimuth = 90.0 + 15.0 * (hour - 6.0)
        direct_normal = 850.0 if solar > 0.0 else 0.0
        diffuse_horizontal = 120.0 * solar
        shortwave = bridge.facet_shortwave_irradiance(
            direct_normal_irradiance=direct_normal,
            diffuse_horizontal_irradiance=diffuse_horizontal,
            solar_altitude_degrees=altitude,
            solar_azimuth_degrees=azimuth,
        )
        air = 296.15 + 5.0 * math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
        forcings.append(
            UrbanForcing(
                air_temperature=air,
                vapor_pressure_kpa=1.8,
                pressure_kpa=101.3,
                wind_speed=2.0,
                sky_longwave=370.0 + 15.0 * solar,
                shortwave_irradiance=shortwave,
            )
        )
        body_shortwave.append(
            torch.full(
                (shortwave.shape[1], shortwave.shape[2]),
                0.7 * (0.25 * direct_normal + diffuse_horizontal),
                dtype=shortwave.dtype,
                device=shortwave.device,
            )
        )
    return forcings, body_shortwave


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the coupled solver on a traced real-geometry bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--spinup-cycles", type=int, default=80)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    with torch.no_grad():
        with np.load(args.bundle, allow_pickle=False) as archive:
            rows, cols = archive["body_sky_view_factor"].shape
        bundle = load_geometry_bundle(args.bundle, rows, cols, device)
        bridge = CoupledRadiationBridge(
            bundle,
            ground_material=MaterialProperties.asphalt(),
            roof_material=MaterialProperties.roof(),
            canopy_properties=CanopyProperties.deciduous(),
            dt=3600.0,
            spinup_max_cycles=args.spinup_cycles,
            strict_convergence=True,
        )
        forcing, body_shortwave = _forcing_cycle(bundle, bridge)
        results, radiation = bridge.solve_cycle(forcing, body_shortwave)
    spinup = bridge.last_spinup
    report = {
        "device": args.device,
        "shape": [rows, cols],
        "spinup_cycles": spinup.cycles,
        "spinup_converged": spinup.converged,
        "spinup_temperature_drift_k": spinup.maximum_temperature_drift,
        "spinup_moisture_drift_kg_m2": spinup.maximum_moisture_drift,
        "maximum_energy_residual_w_m2": max(result.max_energy_residual for result in results),
        "surface_temperature_range_k": [
            min(float(result.state.surface_temperature.min().item()) for result in results),
            max(float(result.state.surface_temperature.max().item()) for result in results),
        ],
        "tmrt_range_c": [
            min(float(value.tmrt_celsius.min().item()) for value in radiation),
            max(float(value.tmrt_celsius.max().item()) for value in radiation),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
