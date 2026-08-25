#UTherm: GPU-accelerated urban microclimate model
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import os
from .core import thermal_comfort
from . import __version__

def str2bool(v):
    """
    Convert string to boolean for argparse.
    
    Args:
        v: Input value (str or bool)
    
    Returns:
        bool: Converted boolean value
    
    Raises:
        argparse.ArgumentTypeError: If value cannot be converted to boolean
    """
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'): return True
    elif v.lower() in ('no', 'false', 'f', '0'): return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

def main():
    """
    Command-line interface for UTherm thermal comfort modeling.
    
    Parses command-line arguments and runs the thermal_comfort function.
    This is the entry point for the 'thermal_comfort' console script.
    
    Usage:
        thermal_comfort --base_path /path/to/input --date 2020-08-13 [options]
    
    For full help:
        thermal_comfort --help
    """
    parser = argparse.ArgumentParser(description="Run UTherm urban microclimate model.")
    parser.add_argument('--version', action='version', version=f'utherm {__version__}')

    # Required arguments
    parser.add_argument('--base_path', required=True, help='Base directory containing input data')
    parser.add_argument('--date', required=True, help='Date for which thermal comfort is computed (e.g., 2021-07-01)')

    # Raster inputs
    parser.add_argument('--building_dsm', default='Building_DSM.tif', help='Building DSM raster filename')
    parser.add_argument('--dem', default='DEM.tif', help='DEM raster filename')
    parser.add_argument('--trees', default='Trees.tif', help='Trees raster filename')
    parser.add_argument('--landcover', default=None, help='Landcover raster filename (optional)')

    # Tiling config
    parser.add_argument('--tile_size', type=int, default=3600, help='Tile size in pixels (e.g., 100–4000)')
    parser.add_argument('--overlap', type=int, default=20, help='Tile overlap in pixels (less than tile_size)')

    # Meteorological inputs
    parser.add_argument('--use_own_met', type=str2bool, default=True, help='Use your own meteorological file (True/False)')
    parser.add_argument('--own_metfile', default=None, help='Path to a UMEP-format meteorological text file')
    parser.add_argument('--data_source_type', default=None, help='Meteorological source (e.g., ERA5, WRF)')
    parser.add_argument('--data_folder', default=None, help='Directory containing ERA5/WRF data files')

    # Optional time range (required if using data_source_type)
    parser.add_argument('--start', default=None, help="Start time (e.g., '2020-08-12 00:00:00')")
    parser.add_argument('--end', default=None, help="End time (e.g., '2020-08-12 23:00:00')")

    # Output options
    parser.add_argument('--save_tmrt', type=str2bool, default=False, help='Save mean radiant temperature output')
    parser.add_argument('--save_svf', type=str2bool, default=False, help='Save sky view factor output')
    parser.add_argument('--save_kup', type=str2bool, default=False, help='Save upward shortwave radiation output')
    parser.add_argument('--save_kdown', type=str2bool, default=False, help='Save downward shortwave radiation output')
    parser.add_argument('--save_lup', type=str2bool, default=False, help='Save upward longwave radiation output')
    parser.add_argument('--save_ldown', type=str2bool, default=False, help='Save downward longwave radiation output')
    parser.add_argument('--save_shadow', type=str2bool, default=False, help='Save shadow map output')

    # Energy balance
    parser.add_argument('--use_energy_balance', type=str2bool, default=False, help='Enable physics-based surface energy balance')
    parser.add_argument('--spinup_days', type=int, default=0, help='Periodic forcing days used to initialize energy-balance state (default: 0)')
    parser.add_argument('--use_coupled_eb', type=str2bool, default=False, help='Enable the seven-facet coupled energy balance')
    parser.add_argument('--coupled_geometry', default=None, help='Coupled facet-geometry .npz file or per-tile directory')
    parser.add_argument('--coupled_spinup_max_cycles', type=int, default=30, help='Maximum periodic spin-up cycles for the coupled solver')
    parser.add_argument('--coupled_strict_convergence', type=str2bool, default=True, help='Fail when coupled timestep or spin-up convergence is not reached')
    parser.add_argument('--z0', type=float, default=0.01, help='Roughness length in meters (default: 0.01)')
    parser.add_argument('--z_wind', type=float, default=None, help='Wind measurement height in meters')
    parser.add_argument('--ground_material_type', default='asphalt', choices=['asphalt', 'concrete', 'brick', 'soil', 'grass', 'dry_soil', 'water'], help='Ground material preset')
    parser.add_argument('--save_qh', type=str2bool, default=False, help='Save sensible heat flux output')
    parser.add_argument('--save_qe', type=str2bool, default=False, help='Save latent heat flux output')
    parser.add_argument('--save_tsfc', type=str2bool, default=False, help='Save ground surface temperature output')

    # Roof energy balance
    parser.add_argument('--use_roof_eb', type=str2bool, default=False, help='Enable roof energy balance')
    parser.add_argument('--roof_material_type', default='roof', choices=['roof', 'green_roof'], help='Roof material preset')
    parser.add_argument('--save_tsfc_roof', type=str2bool, default=False, help='Save roof surface temperature output')

    # Canopy energy balance
    parser.add_argument('--use_canopy_eb', type=str2bool, default=False, help='Enable canopy energy balance')
    parser.add_argument('--canopy_type', default='deciduous', choices=['deciduous', 'evergreen'], help='Canopy type')
    parser.add_argument('--canopy_lai', type=float, default=3.0, help='Leaf area index (default: 3.0)')
    parser.add_argument('--save_t_leaf', type=str2bool, default=False, help='Save leaf temperature output')
    parser.add_argument('--save_canopy_qh', type=str2bool, default=False, help='Save canopy sensible heat flux output')
    parser.add_argument('--save_canopy_qe', type=str2bool, default=False, help='Save canopy latent heat flux (moisture assumptions differ by solver)')

    # Canyon air temperature
    parser.add_argument('--use_canyon_air_temp', type=str2bool, default=False, help='Enable the canyon air-temperature diagnostic')
    parser.add_argument('--tair_method', default='conductance-blend', choices=['conductance-blend', 'log-profile', 'advdiff', 'target'], help='Air-temperature diagnostic; target is a legacy alias for conductance-blend')
    parser.add_argument('--save_tair', type=str2bool, default=False, help='Save air temperature output')
    parser.add_argument('--save_tair_canyon', type=str2bool, default=False, help='Save canyon air temperature output')
    parser.add_argument('--save_wall_temperature', type=str2bool, default=False, help='Save four coupled wall-facet temperature rasters')

    # Wind field
    parser.add_argument('--use_wind_field', type=str2bool, default=False, help='Enable diagnostic spatial wind field model')
    parser.add_argument('--save_wind_speed', type=str2bool, default=False, help='Save wind speed output')
    parser.add_argument('--save_wind_direction', type=str2bool, default=False, help='Save wind direction output')

    args = parser.parse_args()

    # Validation logic
    if args.use_own_met:
        if not args.own_metfile:
            parser.error("--own_metfile is required when --use_own_met=True")
        if not os.path.isfile(args.own_metfile):
            parser.error(f"File not found: {args.own_metfile}")
    else:
        if not args.data_source_type:
            parser.error("--data_source_type is required when --use_own_met=False")
        if not args.data_folder:
            parser.error("--data_folder is required when --use_own_met=False")
        if not os.path.isdir(args.data_folder):
            parser.error(f"Directory not found: {args.data_folder}")
        if not args.start or not args.end:
            parser.error("--start and --end are required when using --data_source_type")

    # Run main function
    thermal_comfort(
        base_path=args.base_path,
        selected_date_str=args.date,
        building_dsm_filename=args.building_dsm,
        dem_filename=args.dem,
        trees_filename=args.trees,
        landcover_filename=args.landcover,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_own_met=args.use_own_met,
        own_met_file=args.own_metfile,
        data_source_type=args.data_source_type,
        data_folder=args.data_folder,
        start_time=args.start,
        end_time=args.end,
        save_tmrt=args.save_tmrt,
        save_svf=args.save_svf,
        save_kup=args.save_kup,
        save_kdown=args.save_kdown,
        save_lup=args.save_lup,
        save_ldown=args.save_ldown,
        save_shadow=args.save_shadow,
        use_energy_balance=args.use_energy_balance,
        use_coupled_eb=args.use_coupled_eb,
        coupled_geometry_path=args.coupled_geometry,
        coupled_spinup_max_cycles=args.coupled_spinup_max_cycles,
        coupled_strict_convergence=args.coupled_strict_convergence,
        z0=args.z0,
        z_wind=args.z_wind,
        ground_material_type=args.ground_material_type,
        save_qh=args.save_qh,
        save_qe=args.save_qe,
        use_roof_eb=args.use_roof_eb,
        roof_material_type=args.roof_material_type,
        use_canopy_eb=args.use_canopy_eb,
        canopy_type=args.canopy_type,
        canopy_lai=args.canopy_lai,
        use_canyon_air_temp=args.use_canyon_air_temp,
        save_tsfc=args.save_tsfc,
        save_tsfc_roof=args.save_tsfc_roof,
        save_t_leaf=args.save_t_leaf,
        save_canopy_qh=args.save_canopy_qh,
        save_canopy_qe=args.save_canopy_qe,
        save_tair=args.save_tair,
        save_tair_canyon=args.save_tair_canyon,
        save_wall_temperature=args.save_wall_temperature,
        use_wind_field=args.use_wind_field,
        save_wind_speed=args.save_wind_speed,
        save_wind_direction=args.save_wind_direction,
        tair_method=args.tair_method,
        spinup_days=args.spinup_days,
    )


if __name__ == "__main__":
    main()
