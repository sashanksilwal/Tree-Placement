#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utherm.energy_balance import CanopyProperties, MaterialProperties
from utherm.energy_balance.coupled import (
    CANOPY,
    GROUND,
    N_FACETS,
    ROOF,
    WALL_EAST,
    WALL_NORTH,
    WALL_SOUTH,
    WALL_WEST,
    CoupledUrbanEBConfig,
    CoupledUrbanEnergyBalance,
    UrbanFacetGeometry,
    UrbanForcing,
)


def geometry(rows, cols, device):
    facet_area = torch.tensor(
        [0.7, 0.3, 0.25, 0.25, 0.25, 0.25, 0.8],
        dtype=torch.float32,
        device=device,
    )
    area = facet_area.view(N_FACETS, 1, 1).expand(-1, rows, cols).clone()
    exchange = torch.zeros(
        N_FACETS, N_FACETS, rows, cols, dtype=torch.float32, device=device
    )
    for first in range(N_FACETS):
        for second in range(first + 1, N_FACETS):
            value = 0.02 * min(float(facet_area[first]), float(facet_area[second]))
            exchange[first, second] = value
            exchange[second, first] = value
    sky = area - exchange.sum(dim=1)
    return UrbanFacetGeometry(area, sky, exchange)


def forcing_cycle(area):
    result = []
    for hour in range(24):
        solar = max(math.sin(math.pi * (hour - 6.0) / 12.0), 0.0)
        azimuth = 15.0 * hour
        shortwave = torch.zeros_like(area)
        shortwave[GROUND] = 650.0 * solar
        shortwave[ROOF] = 850.0 * solar
        shortwave[CANOPY] = 500.0 * solar
        wall_sun = {
            WALL_NORTH: max(math.cos(math.radians(azimuth)), 0.0),
            WALL_EAST: max(math.sin(math.radians(azimuth)), 0.0),
            WALL_SOUTH: max(-math.cos(math.radians(azimuth)), 0.0),
            WALL_WEST: max(-math.sin(math.radians(azimuth)), 0.0),
        }
        for facet, incidence in wall_sun.items():
            shortwave[facet] = (100.0 + 500.0 * incidence) * solar
        air = 296.15 + 5.0 * math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
        result.append(
            UrbanForcing(
                air_temperature=air,
                vapor_pressure_kpa=1.8,
                pressure_kpa=101.3,
                wind_speed=2.0,
                sky_longwave=370.0 + 15.0 * solar,
                shortwave_irradiance=shortwave,
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--max-spinup-cycles", type=int, default=40)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.rows < 1 or args.cols < 1 or args.max_spinup_cycles < 2:
        raise SystemExit("rows and cols must be positive; max spin-up cycles must be at least 2")

    device = torch.device(args.device)
    urban_geometry = geometry(args.rows, args.cols, device)
    water_capacity = torch.tensor(
        [20.0, 1.0, 0.2, 0.2, 0.2, 0.2, 1.5],
        dtype=torch.float32,
        device=device,
    ).view(N_FACETS, 1, 1).expand_as(urban_geometry.area).clone()
    materials = [MaterialProperties.soil(), MaterialProperties.roof()] + [
        MaterialProperties.brick()
    ] * 4
    config = CoupledUrbanEBConfig(
        max_coupling_iterations=120,
        spinup_max_cycles=args.max_spinup_cycles,
        spinup_temperature_tolerance=0.1,
        spinup_moisture_tolerance=0.02,
        strict_convergence=True,
    )
    model = CoupledUrbanEnergyBalance(
        urban_geometry,
        materials,
        CanopyProperties.deciduous(),
        water_capacity,
        config,
    )
    forcings = forcing_cycle(urban_geometry.area)
    spinup = model.spin_up(forcings)
    results = model.run(forcings, state=spinup.state, spin_up=False)
    peak_residual = max(item.max_energy_residual for item in results)
    maximum_temperature = max(
        float(item.state.surface_temperature.max().item()) for item in results
    )
    minimum_temperature = min(
        float(item.state.surface_temperature.min().item()) for item in results
    )
    print(
        json.dumps(
            {
                "device": args.device,
                "shape": [args.rows, args.cols],
                "spinup_cycles": spinup.cycles,
                "spinup_temperature_drift_K": spinup.maximum_temperature_drift,
                "spinup_moisture_drift_kg_m2": spinup.maximum_moisture_drift,
                "spinup_humidity_drift_kg_kg": spinup.maximum_specific_humidity_drift,
                "peak_energy_residual_W_m2": peak_residual,
                "surface_temperature_range_K": [minimum_temperature, maximum_temperature],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
