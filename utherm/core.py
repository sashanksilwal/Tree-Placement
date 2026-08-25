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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Optional


def _resolve_raster_path(base_path, raster_path):
    """Resolve a raster argument using the same rules as preprocessing."""
    path = Path(raster_path).expanduser()
    return path if path.is_absolute() else Path(base_path).expanduser() / path


def _validate_run_arguments(selected_date_str, tile_size, overlap, spinup_days=0):
    """Reject invalid run settings before preprocessing creates any files."""
    try:
        datetime.strptime(selected_date_str, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError("selected_date_str must use YYYY-MM-DD") from exc
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise ValueError("overlap must be an integer")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if (
        not isinstance(spinup_days, int)
        or isinstance(spinup_days, bool)
        or spinup_days < 0
    ):
        raise ValueError("spinup_days must be a nonnegative integer")


def _validate_model_arguments(
    *,
    use_own_met,
    own_met_file,
    data_source_type,
    data_folder,
    start_time,
    end_time,
    use_energy_balance,
    use_coupled_eb,
    coupled_geometry_path,
    coupled_spinup_max_cycles,
    coupled_strict_convergence,
    z0,
    z_wind,
    ground_material_type,
    use_roof_eb,
    roof_material_type,
    use_canopy_eb,
    canopy_type,
    canopy_lai,
    use_canyon_air_temp,
    use_wind_field,
    tair_method,
    ground_wetness,
    canopy_gmax_fraction,
    requested_outputs,
):
    """Validate coupled model options before preprocessing starts."""
    import math

    if use_own_met:
        if own_met_file is None:
            raise ValueError("own_met_file is required when use_own_met=True")
        if not Path(own_met_file).expanduser().is_file():
            raise FileNotFoundError(f"meteorological file not found: {own_met_file}")
    else:
        if data_source_type not in {"ERA5", "wrfout"}:
            raise ValueError("data_source_type must be 'ERA5' or 'wrfout'")
        if data_folder is None or not Path(data_folder).expanduser().is_dir():
            raise FileNotFoundError(f"meteorological data folder not found: {data_folder}")
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time are required for ERA5/WRF input")
    if not math.isfinite(z0) or z0 <= 0:
        raise ValueError("z0 must be finite and positive")
    if z_wind is not None and (not math.isfinite(z_wind) or z_wind <= 0):
        raise ValueError("z_wind must be finite and positive")
    if ground_material_type not in {
        "asphalt", "concrete", "brick", "soil", "grass", "dry_soil", "water"
    }:
        raise ValueError("unknown ground_material_type")
    if roof_material_type not in {"roof", "green_roof"}:
        raise ValueError("unknown roof_material_type")
    if canopy_type not in {"deciduous", "evergreen"}:
        raise ValueError("unknown canopy_type")
    if not math.isfinite(canopy_lai) or canopy_lai <= 0:
        raise ValueError("canopy_lai must be finite and positive")
    if tair_method not in {"conductance-blend", "target", "log-profile", "advdiff"}:
        raise ValueError("unknown tair_method")
    for name, value in (
        ("ground_wetness", ground_wetness),
        ("canopy_gmax_fraction", canopy_gmax_fraction),
    ):
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1")
    coupled = {
        "use_roof_eb": use_roof_eb,
        "use_canopy_eb": use_canopy_eb,
        "use_canyon_air_temp": use_canyon_air_temp,
    }
    enabled_without_eb = [
        name for name, enabled in coupled.items()
        if enabled and not (use_energy_balance or use_coupled_eb)
    ]
    if enabled_without_eb:
        raise ValueError(
            "energy-balance options require use_energy_balance=True or "
            f"use_coupled_eb=True: {enabled_without_eb}"
        )
    if not isinstance(use_coupled_eb, bool):
        raise ValueError("use_coupled_eb must be boolean")
    if use_coupled_eb and use_energy_balance:
        raise ValueError(
            "use_coupled_eb and use_energy_balance are alternative solvers and "
            "cannot both be enabled"
        )
    if use_coupled_eb:
        if coupled_geometry_path is None:
            raise ValueError(
                "coupled_geometry_path is required when use_coupled_eb=True"
            )
        if not Path(coupled_geometry_path).expanduser().exists():
            raise FileNotFoundError(
                f"coupled geometry path not found: {coupled_geometry_path}"
            )
        incompatible = [name for name, enabled in coupled.items() if enabled]
        if incompatible:
            raise ValueError(
                "legacy facet switches cannot be combined with use_coupled_eb=True: "
                f"{incompatible}"
            )
    if (
        not isinstance(coupled_spinup_max_cycles, int)
        or isinstance(coupled_spinup_max_cycles, bool)
        or coupled_spinup_max_cycles < 2
    ):
        raise ValueError("coupled_spinup_max_cycles must be an integer of at least two")
    if not isinstance(coupled_strict_convergence, bool):
        raise ValueError("coupled_strict_convergence must be boolean")
    requirements = {
        "save_qh": use_energy_balance or use_coupled_eb,
        "save_qe": use_energy_balance or use_coupled_eb,
        "save_tsfc": use_energy_balance or use_coupled_eb,
        "save_tsfc_roof": use_roof_eb or use_coupled_eb,
        "save_t_leaf": use_canopy_eb or use_coupled_eb,
        "save_canopy_qh": use_canopy_eb or use_coupled_eb,
        "save_canopy_qe": use_canopy_eb or use_coupled_eb,
        "save_tair_canyon": use_canyon_air_temp or use_coupled_eb,
        "save_wall_temperature": use_coupled_eb,
        "save_wind_speed": use_wind_field,
        "save_wind_direction": use_wind_field,
    }
    invalid_outputs = [name for name, requested in requested_outputs.items() if requested and not requirements[name]]
    if invalid_outputs:
        raise ValueError(f"requested outputs are not enabled by their model options: {invalid_outputs}")


def _validate_tile_maps(tile_maps: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Require every generated input to contain the same, nonempty tile set."""
    if not tile_maps:
        raise ValueError("no tile maps were provided")
    reference_label, reference_map = next(iter(tile_maps.items()))
    reference_keys = set(reference_map)
    if not reference_keys:
        raise RuntimeError(f"no generated tiles found for {reference_label}")
    for label, mapping in list(tile_maps.items())[1:]:
        keys = set(mapping)
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise RuntimeError(
                f"generated tile keys do not match for {label}: "
                f"missing={missing}, extra={extra}"
            )

    def numeric_key(key: str):
        try:
            x, y = key.split("_")
            return int(x), int(y)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid generated tile key: {key!r}") from exc

    return sorted(reference_keys, key=numeric_key)


def _load_met_file(path, selected_date_str):
    """Load and validate a UMEP-format meteorological text file."""
    import numpy as np

    met_path = Path(path)
    try:
        data = np.loadtxt(met_path, skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read meteorological file {met_path}: {exc}") from exc
    if data.shape[0] == 0 or data.shape[1] < 23:
        raise ValueError(
            f"meteorological file {met_path} must contain at least one row and 23 columns"
        )
    required_columns = (0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 21, 22)
    if not np.isfinite(data[:, required_columns]).all():
        raise ValueError(f"meteorological file {met_path} has non-finite required values")
    selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")
    calendar_columns = data[:, :4]
    if not np.equal(calendar_columns, np.floor(calendar_columns)).all():
        raise ValueError(f"meteorological file {met_path} has non-integer date or time fields")
    years, days, hours, minutes = calendar_columns.astype(int).T
    if not ((1 <= days) & (days <= 366)).all():
        raise ValueError(f"meteorological file {met_path} has day-of-year outside 1..366")
    if not ((0 <= hours) & (hours <= 23)).all():
        raise ValueError(f"meteorological file {met_path} has hour outside 0..23")
    if not ((0 <= minutes) & (minutes <= 59)).all():
        raise ValueError(f"meteorological file {met_path} has minute outside 0..59")
    expected_day = selected_date.timetuple().tm_yday
    if not ((years == selected_date.year) & (days == expected_day)).all():
        raise ValueError(
            f"meteorological file {met_path} does not match {selected_date_str}"
        )
    timestamps = np.column_stack((years, days, hours, minutes))
    if np.unique(timestamps, axis=0).shape[0] != timestamps.shape[0]:
        raise ValueError(f"meteorological file {met_path} has duplicate timestamps")
    if data.shape[0] > 1:
        minutes_since_midnight = hours * 60 + minutes
        if not np.all(np.diff(minutes_since_midnight) > 0):
            raise ValueError(f"meteorological file {met_path} timestamps are not increasing")
    if (data[:, 9] < 0.0).any():
        raise ValueError(f"meteorological file {met_path} has negative wind speed")
    if ((data[:, 10] < 0.0) | (data[:, 10] > 100.0)).any():
        raise ValueError(f"meteorological file {met_path} has relative humidity outside 0..100")
    if ((data[:, 11] < -100.0) | (data[:, 11] > 70.0)).any():
        raise ValueError(f"meteorological file {met_path} has air temperature outside -100..70 C")
    if ((data[:, 12] < 20.0) | (data[:, 12] > 120.0)).any():
        raise ValueError(f"meteorological file {met_path} has pressure outside 20..120 kPa")
    if (data[:, 14] < 0.0).any():
        raise ValueError(f"meteorological file {met_path} has negative global radiation")
    if (data[:, 13] < 0.0).any():
        raise ValueError(f"meteorological file {met_path} has negative precipitation")
    for column, label in ((21, "diffuse radiation"), (22, "direct radiation")):
        values = data[:, column]
        if ((values < 0.0) & (values != -999.0)).any():
            raise ValueError(
                f"meteorological file {met_path} has invalid {label}; use -999 for missing data"
            )
    if data.shape[1] >= 24:
        direction = data[:, 23]
        invalid_direction = ((direction < 0.0) | (direction > 360.0)) & (direction != -999.0)
        if not np.isfinite(direction).all() or invalid_direction.any():
            raise ValueError(
                f"meteorological file {met_path} has invalid wind direction; "
                "use -999 for missing data"
            )
    return data


def _expand_met_for_spinup(data, selected_date_str, spinup_days):
    """Prepend periodic copies of one forcing day for thermal-state spin-up.

    Meteorological values are repeated, while year and day-of-year are changed
    to the preceding calendar dates so solar geometry remains chronological.
    The returned start index marks the first requested-day record.
    """
    import numpy as np

    _validate_run_arguments(selected_date_str, 1, 0, spinup_days)
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 4:
        raise ValueError("spin-up forcing must be a nonempty two-dimensional table")
    if spinup_days == 0:
        return values.copy(), 0

    selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")
    cycles = []
    for days_before in range(spinup_days, -1, -1):
        cycle_date = selected_date - timedelta(days=days_before)
        cycle = values.copy()
        cycle[:, 0] = cycle_date.year
        cycle[:, 1] = cycle_date.timetuple().tm_yday
        cycles.append(cycle)
    return np.concatenate(cycles, axis=0), spinup_days * values.shape[0]


def _require_wind_directions(data, path):
    """Require usable directions when a spatial wind calculation is enabled."""
    if data.shape[1] < 24 or ((data[:, 23] < 0.0) | (data[:, 23] > 360.0)).any():
        raise ValueError(
            f"meteorological file {path} needs wind direction in column 24 "
            "with values from 0 to 360 degrees"
        )


def _validate_input_rasters(
    base_path,
    building_dsm_filename,
    dem_filename,
    trees_filename,
    landcover_filename=None,
):
    """Validate the public API's raster alignment contract before writing output."""
    import numpy as np

    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Raster validation requires the 'geo' dependencies; install utherm[geo]."
        ) from exc

    inputs = [
        ("building DSM", _resolve_raster_path(base_path, building_dsm_filename)),
        ("DEM", _resolve_raster_path(base_path, dem_filename)),
        ("trees", _resolve_raster_path(base_path, trees_filename)),
    ]
    if landcover_filename is not None:
        inputs.append(("land cover", _resolve_raster_path(base_path, landcover_filename)))

    for label, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{label} raster not found: {path}")

    reference_label, reference_path = inputs[0]
    with rasterio.open(reference_path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"{reference_label} raster must have one band: {reference_path}")
        values = dataset.read(1, masked=True)
        if bool(np.ma.getmaskarray(values).any()) or not np.isfinite(values.filled(np.nan)).all():
            raise ValueError(f"{reference_label} raster contains nodata or non-finite values")
        reference_shape = (dataset.height, dataset.width)
        reference_crs = dataset.crs
        reference_transform = dataset.transform
    if reference_crs is None:
        raise ValueError(f"{reference_label} raster has no CRS: {reference_path}")

    for label, path in inputs[1:]:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(f"{label} raster must have one band: {path}")
            values = dataset.read(1, masked=True)
            if bool(np.ma.getmaskarray(values).any()) or not np.isfinite(values.filled(np.nan)).all():
                raise ValueError(f"{label} raster contains nodata or non-finite values")
            shape = (dataset.height, dataset.width)
            crs = dataset.crs
            transform = dataset.transform
        if shape != reference_shape:
            raise ValueError(
                f"Raster dimensions do not match: {label} {shape}, "
                f"{reference_label} {reference_shape}."
            )
        if crs != reference_crs:
            raise ValueError(
                f"Raster CRS does not match: {label} {crs}, "
                f"{reference_label} {reference_crs}."
            )
        if transform != reference_transform:
            raise ValueError(
                f"Raster transform does not match: {label} {transform}, "
                f"{reference_label} {reference_transform}."
            )

def thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: Optional[str] = None,
    tile_size=3600,
    overlap = 20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    use_energy_balance=False,
    use_coupled_eb=False,
    coupled_geometry_path: Optional[str] = None,
    coupled_spinup_max_cycles=30,
    coupled_strict_convergence=True,
    z0=0.01,
    z_wind=None,
    ground_material_type='asphalt',
    save_qh=False,
    save_qe=False,
    use_roof_eb=False,
    roof_material_type='roof',
    use_canopy_eb=False,
    canopy_type='deciduous',
    canopy_lai=3.0,
    use_canyon_air_temp=False,
    save_tsfc=False,
    save_tsfc_roof=False,
    save_t_leaf=False,
    save_canopy_qh=False,
    save_canopy_qe=False,
    save_tair=False,
    save_tair_canyon=False,
    save_wall_temperature=False,
    use_wind_field=False,
    save_wind_speed=False,
    save_wind_direction=False,
    tair_method='conductance-blend',
    ground_wetness=None,
    canopy_gmax_fraction=None,
    spinup_days=0,
):
    """
    Main function to compute urban thermal comfort using the UTherm model.

    This function orchestrates the complete workflow:
    1. Preprocesses input rasters (tiling, validation)
    2. Processes meteorological data (ERA5, WRF, or custom)
    3. Calculates wall heights and aspects (parallel CPU)
    4. Computes shadows, radiation, and SVF (GPU-accelerated)
    5. Calculates UTCI thermal comfort index
    6. Saves outputs as georeferenced rasters

    Args:
        base_path (str): Base directory for output_folder/ and processed_inputs/; also used to resolve relative raster paths. For outputs elsewhere, set this to the desired output directory and pass complete paths for the raster arguments.
        selected_date_str (str): Simulation date in format 'YYYY-MM-DD'
        building_dsm_filename (str): Building+terrain DSM path or filename (relative to base_path). Default: 'Building_DSM.tif'. Can be a complete path if rasters live elsewhere.
        dem_filename (str): DEM path or filename (relative to base_path). Default: 'DEM.tif'
        trees_filename (str): Vegetation DSM path or filename (relative to base_path). Default: 'Trees.tif'
        landcover_filename (str, optional): Land cover raster path or filename. Default: None
        tile_size (int): Tile size in pixels. Default: 3600. Adjust based on GPU RAM.
        overlap (int): Overlap between tiles in pixels. Default: 20. Used for shadow transfer.
        use_own_met (bool): Use custom meteorological file. Default: True
        start_time (str, optional): Start datetime 'YYYY-MM-DD HH:MM:SS' (UTC for ERA5/WRF)
        end_time (str, optional): End datetime 'YYYY-MM-DD HH:MM:SS' (UTC for ERA5/WRF)
        data_source_type (str, optional): 'ERA5' or 'wrfout' if not using own met file
        data_folder (str, optional): Folder containing ERA5/WRF NetCDF files
        own_met_file (str, optional): Path to custom meteorological text file
        spinup_days (int): Number of periodic preceding forcing days used to
            initialize thermal state. Only the requested day is written.
        save_tmrt (bool): Save mean radiant temperature output. Default: True
        save_svf (bool): Save sky view factor output. Default: False
        save_kup (bool): Save upward shortwave radiation. Default: False
        save_kdown (bool): Save downward shortwave radiation. Default: False
        save_lup (bool): Save upward longwave radiation. Default: False
        save_ldown (bool): Save downward longwave radiation. Default: False
        save_shadow (bool): Save shadow maps. Default: False
        use_energy_balance (bool): Enable physics-based energy balance. Default: False
        use_coupled_eb (bool): Enable the seven-facet coupled energy balance.
            This replaces, rather than augments, use_energy_balance.
        coupled_geometry_path (str, optional): Reciprocal facet geometry bundle,
            required when use_coupled_eb=True.
        z0 (float): Roughness length (m). Default: 0.01
        z_wind (float, optional): Wind measurement height (m)
        ground_material_type (str): Ground material preset. Default: 'asphalt'

    Returns:
        None: Outputs are saved to `{base_path}/output_folder/` directory

    Output Structure:
        - {base_path}/processed_inputs/ - All preprocessing files
          - Building_DSM/ - Preprocessing tiles
          - DEM/ - Preprocessing tiles
          - Trees/ - Preprocessing tiles
          - metfiles/ - Meteorological files
          - walls/ - Wall height rasters
          - aspect/ - Wall aspect rasters
          - Outfile.nc - Processed NetCDF (if using ERA5/WRF)
        - output_folder/{tile_key}/ - One folder per tile (tile_key e.g. "0_0", "1000_0")
          - UTCI_{tile_key}.tif - Universal Thermal Climate Index (always saved)
          - TMRT_{tile_key}.tif - Mean radiant temperature (if save_tmrt=True)
          - SVF_{tile_key}.tif - Sky view factor (if save_svf=True)
          - Kup_{tile_key}.tif - Upward shortwave (if save_kup=True)
          - Kdown_{tile_key}.tif - Downward shortwave (if save_kdown=True)
          - Lup_{tile_key}.tif - Upward longwave (if save_lup=True)
          - Ldown_{tile_key}.tif - Downward longwave (if save_ldown=True)
          - Shadow_{tile_key}.tif - Shadow maps (if save_shadow=True)

    Notes:
        - Automatically uses GPU if available, falls back to CPU
        - Processes tiles in parallel for large domains
        - UTC to local time conversion handled automatically
        - Multi-band rasters: one band per hour

    Example:
        >>> from utherm import thermal_comfort
        >>> thermal_comfort(
        ...     base_path='/path/to/input',
        ...     selected_date_str='2020-08-13',
        ...     tile_size=1000,
        ...     overlap=100,
        ...     use_own_met=True,
        ...     own_met_file='/path/to/met.txt'
        ... )

    Raises:
        ValueError: If input rasters have mismatched dimensions, CRS, or pixel sizes
        FileNotFoundError: If the required input files are missing
    """

    _validate_run_arguments(selected_date_str, tile_size, overlap, spinup_days)
    if spinup_days and not use_energy_balance:
        raise ValueError(
            "spinup_days applies to the modular energy balance; the coupled solver "
            "uses convergence-based periodic spin-up"
        )
    _validate_input_rasters(
        base_path,
        building_dsm_filename,
        dem_filename,
        trees_filename,
        landcover_filename,
    )
    _validate_model_arguments(
        use_own_met=use_own_met,
        own_met_file=own_met_file,
        data_source_type=data_source_type,
        data_folder=data_folder,
        start_time=start_time,
        end_time=end_time,
        use_energy_balance=use_energy_balance,
        use_coupled_eb=use_coupled_eb,
        coupled_geometry_path=coupled_geometry_path,
        coupled_spinup_max_cycles=coupled_spinup_max_cycles,
        coupled_strict_convergence=coupled_strict_convergence,
        z0=z0,
        z_wind=z_wind,
        ground_material_type=ground_material_type,
        use_roof_eb=use_roof_eb,
        roof_material_type=roof_material_type,
        use_canopy_eb=use_canopy_eb,
        canopy_type=canopy_type,
        canopy_lai=canopy_lai,
        use_canyon_air_temp=use_canyon_air_temp,
        use_wind_field=use_wind_field,
        tair_method=tair_method,
        ground_wetness=ground_wetness,
        canopy_gmax_fraction=canopy_gmax_fraction,
        requested_outputs={
            "save_qh": save_qh,
            "save_qe": save_qe,
            "save_tsfc": save_tsfc,
            "save_tsfc_roof": save_tsfc_roof,
            "save_t_leaf": save_t_leaf,
            "save_canopy_qh": save_canopy_qh,
            "save_canopy_qe": save_canopy_qe,
            "save_tair_canyon": save_tair_canyon,
            "save_wall_temperature": save_wall_temperature,
            "save_wind_speed": save_wind_speed,
            "save_wind_direction": save_wind_direction,
        },
    )
    from .preprocessor import ppr
    from .utci_process import compute_utci, map_files_by_key
    from .walls_aspect import run_parallel_processing
    import os
    import torch
    # Create preprocessing outputs directory
    preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    ppr(
        base_path, building_dsm_filename, dem_filename, trees_filename,
        landcover_filename, tile_size, overlap, selected_date_str, use_own_met,
        start_time, end_time, data_source_type, data_folder, own_met_file,
         preprocess_dir=preprocess_dir
    )

    base_output_path = os.path.join(base_path, "output_folder")
    inputMet = os.path.join(preprocess_dir, "metfiles")
    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    tree_dir = os.path.join(preprocess_dir, "Trees")
    dem_dir = os.path.join(preprocess_dir, "DEM")
    landcover_dir = os.path.join(preprocess_dir, "Landcover") if landcover_filename is not None else None
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")

    run_parallel_processing(building_dsm_dir, walls_dir, aspect_dir)

    if torch.cuda.is_available():
        print("[GPU] GPU acceleration enabled")
    print("Running UTherm ...")

    building_dsm_map = map_files_by_key(building_dsm_dir, ".tif")
    tree_map = map_files_by_key(tree_dir, ".tif")
    dem_map = map_files_by_key(dem_dir, ".tif")
    landcover_map = map_files_by_key(landcover_dir, ".tif") if landcover_dir else {}
    walls_map = map_files_by_key(walls_dir, ".tif")
    aspect_map = map_files_by_key(aspect_dir, ".tif")
    met_map = map_files_by_key(inputMet, ".txt", is_metfile=True)
    tile_maps = {
        "building DSM": building_dsm_map,
        "trees": tree_map,
        "DEM": dem_map,
        "walls": walls_map,
        "wall aspect": aspect_map,
        "meteorology": met_map,
    }
    if landcover_dir:
        tile_maps["land cover"] = landcover_map
    tile_keys = _validate_tile_maps(tile_maps)

    for key in tile_keys:

        building_dsm_path = building_dsm_map[key]
        tree_path = tree_map[key]
        dem_path = dem_map[key]
        landcover_path = landcover_map.get(key) if landcover_dir else None
        walls_path = walls_map[key]
        aspect_path = aspect_map[key]
        met_file_path = met_map[key]

        output_folder = os.path.join(base_output_path, key)
        os.makedirs(output_folder, exist_ok=True)

        met_file_data = _load_met_file(met_file_path, selected_date_str)
        if use_wind_field or tair_method == "advdiff":
            _require_wind_directions(met_file_data, met_file_path)
        if use_coupled_eb:
            output_start_index = 0
        else:
            met_file_data, output_start_index = _expand_met_for_spinup(
                met_file_data, selected_date_str, spinup_days
            )

        compute_utci(
            building_dsm_path,
            tree_path,
            dem_path,
            walls_path,
            aspect_path,
            landcover_path,
            met_file_data,
            output_folder,
            key,
            selected_date_str,
            save_tmrt=save_tmrt,
            save_svf=save_svf,
            save_kup=save_kup,
            save_kdown=save_kdown,
            save_lup=save_lup,
            save_ldown=save_ldown,
            save_shadow=save_shadow,
            use_energy_balance=use_energy_balance,
            use_coupled_eb=use_coupled_eb,
            coupled_geometry_path=coupled_geometry_path,
            coupled_spinup_max_cycles=coupled_spinup_max_cycles,
            coupled_strict_convergence=coupled_strict_convergence,
            z0=z0,
            z_wind=z_wind,
            ground_material_type=ground_material_type,
            save_qh=save_qh,
            save_qe=save_qe,
            use_roof_eb=use_roof_eb,
            roof_material_type=roof_material_type,
            use_canopy_eb=use_canopy_eb,
            canopy_type=canopy_type,
            canopy_lai=canopy_lai,
            use_canyon_air_temp=use_canyon_air_temp,
            save_tsfc=save_tsfc,
            save_tsfc_roof=save_tsfc_roof,
            save_t_leaf=save_t_leaf,
            save_canopy_qh=save_canopy_qh,
            save_canopy_qe=save_canopy_qe,
            save_tair=save_tair,
            save_tair_canyon=save_tair_canyon,
            save_wall_temperature=save_wall_temperature,
            use_wind_field=use_wind_field,
            save_wind_speed=save_wind_speed,
            save_wind_direction=save_wind_direction,
            tair_method=tair_method,
            ground_wetness=ground_wetness,
            canopy_gmax_fraction=canopy_gmax_fraction,
            output_start_index=output_start_index,
        )

        # Free GPU memory between tiles
        torch.cuda.empty_cache()
