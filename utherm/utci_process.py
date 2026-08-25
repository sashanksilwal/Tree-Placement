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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import replace
import math
import numpy as np
from osgeo import gdal, osr
import torch
import torch.nn.functional as F
import time
from timezonefinder import TimezoneFinder
import pytz
import datetime
from .Tgmaps_v1 import Tgmaps_v1
from .sun_position import Solweig_2015a_metdata_noload
from .shadow import svf_calculator, create_patches
from .radiation import Solweig_2022a_calc, clearnessindex_2013b
from .calculate_utci import utci_calculator
import os
import re
gdal.UseExceptions()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(__file__)
landcover_classes_path = os.path.join(script_dir, 'landcoverclasses_2016a.txt')


def _dry_coupled_spinup_forcings(forcings):
    """Return a dry-antecedent copy without changing target-cycle forcing."""
    return [replace(forcing, precipitation_rate=0.0) for forcing in forcings]

# Wall and ground emissivity and albedo
albedo_b = 0.2
albedo_g = 0.15
ewall = 0.9
eground = 0.95
absK = 0.7
absL = 0.95

# Standing position
Fside = 0.22
Fup = 0.06
Fcyl = 0.28

cyl = True
elvis = 0
usevegdem = 1
onlyglobal = 1

firstdayleaf = 97
lastdayleaf = 300
conifer_bool = False


def _indices_for_current_day(dectime, index):
    """Return indices sharing the calendar day of ``dectime[index]``."""
    values = np.asarray(dectime, dtype=float)
    if values.ndim != 1 or not 0 <= index < values.size:
        raise ValueError("dectime must be one-dimensional and index must be valid")
    if not np.isfinite(values).all():
        raise ValueError("dectime must contain only finite values")
    return np.flatnonzero(np.floor(values) == np.floor(values[index]))


def _daily_mean_air_temperature(air_temperature, dectime, index):
    """Mean air temperature for the calendar day containing ``index``."""
    values = np.asarray(air_temperature, dtype=float)
    indices = _indices_for_current_day(dectime, index)
    if values.ndim != 1 or values.size != np.asarray(dectime).size:
        raise ValueError("air_temperature and dectime must be equal-length vectors")
    daily_values = values[indices]
    if daily_values.size == 0 or not np.isfinite(daily_values).all():
        raise ValueError("daily air temperatures must be present and finite")
    return float(np.mean(daily_values))


def load_raster_to_tensor(dem_path):
    """
    Load a GeoTIFF raster file into a PyTorch tensor.
    
    Args:
        dem_path (str): Path to GeoTIFF file
    
    Returns:
        tuple: (tensor, dataset) where:
            - tensor: PyTorch tensor on GPU/CPU with raster data
            - dataset: GDAL dataset object (for accessing metadata)
    """
    dataset = gdal.Open(dem_path)
    band = dataset.GetRasterBand(1)
    array = np.array(band.ReadAsArray(), dtype=np.float32)
    return torch.from_numpy(array).to(device), dataset

def extract_key(filename, is_metfile=False):
    """
    Extract numerical key from filename for tile matching.
    
    Args:
        filename (str): Filename to parse
        is_metfile (bool): True if filename is a metfile, False if raster tile
    
    Returns:
        str: Extracted key (e.g., "0_0" from "Building_DSM_0_0.tif")
    """

    if is_metfile:
        # Two writers produce met files under different conventions:
        # generated forcing -> metfile_X_Y_YYYY-MM-DD.txt (preprocessor.py:771)
        # user-supplied met -> metfile_X_Y.txt            (preprocessor.py:906)
        # The trailing date is therefore optional.
        match = re.search(r'metfile_(\d+)_(\d+)(?:_\d{4}-\d{2}-\d{2})?', filename)
    else:
        # look for ..._X_Y.tif
        match = re.search(r'_(\d+)_(\d+)', filename)

    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

# Function to list matching files in a directory
def get_matching_files(directory, extension):
    """
    Get sorted list of files with given extension from directory.
    
    Args:
        directory (str): Directory path to search
        extension (str): File extension to filter (e.g., '.tif')
    
    Returns:
        list: Sorted list of filenames matching extension
    """
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def map_files_by_key(directory, extension, is_metfile=False):
    """
    Create mapping of tile keys to filenames.
    
    Groups files by their tile coordinates (e.g., "0_0", "1000_0") to match
    corresponding raster tiles with their meteorological files.
    
    Args:
        directory (str): Directory containing files
        extension (str): File extension to filter
        is_metfile (bool): True if files are metfiles
    
    Returns:
        dict: Dictionary mapping keys to filenames
    """
    files = get_matching_files(directory, extension)
    mapping = {}
    for f in files:
        key = extract_key(f, is_metfile=is_metfile)
        if key:
            if key in mapping:
                raise RuntimeError(
                    f"duplicate tile key {key!r} in {directory}: "
                    f"{os.path.basename(mapping[key])!r} and {f!r}"
                )
            mapping[key] = os.path.join(directory, f)
    return mapping


def _met_timestep_seconds(met_file):
    """Return the regular forcing interval in seconds; use one hour for one row."""
    if met_file.shape[0] == 1:
        return 3600.0
    timestamps = []
    for row in met_file:
        year, day, hour, minute = (int(row[index]) for index in range(4))
        timestamps.append(
            datetime.datetime(year, 1, 1)
            + datetime.timedelta(days=day - 1, hours=hour, minutes=minute)
        )
    intervals = np.diff(np.asarray(timestamps, dtype="datetime64[s]")).astype(
        "timedelta64[s]"
    ).astype(np.float64)
    if not np.isfinite(intervals).all() or (intervals <= 0.0).any():
        raise ValueError("meteorological forcing intervals must be finite and positive")
    if not np.allclose(intervals, intervals[0], rtol=0.0, atol=1.0):
        raise ValueError("coupled energy balance requires a regular forcing interval")
    return float(intervals[0])


def _validate_coupled_geometry_alignment(bundle, raster_dataset):
    geometry_crs = osr.SpatialReference()
    tile_crs = osr.SpatialReference()
    if geometry_crs.ImportFromWkt(bundle.crs_wkt) != 0:
        raise ValueError(f"invalid CRS WKT in coupled geometry bundle {bundle.source_path}")
    if tile_crs.ImportFromWkt(raster_dataset.GetProjection()) != 0:
        raise ValueError("processed building raster has invalid CRS WKT")
    if not bool(geometry_crs.IsSame(tile_crs)):
        raise ValueError(
            f"coupled geometry CRS does not match processed tile: {bundle.source_path}"
        )
    tile_transform = np.asarray(raster_dataset.GetGeoTransform(), dtype=float)
    geometry_transform = np.asarray(bundle.geotransform, dtype=float)
    if not np.allclose(geometry_transform, tile_transform, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"coupled geometry transform does not match processed tile: "
            f"geometry={tuple(geometry_transform)}, tile={tuple(tile_transform)}"
        )

def extract_number_from_filename(filename):
    """
    Extract tile number from Building_DSM filename.
    
    Args:
        filename (str): Filename in format "Building_DSM_X_Y.tif"
    
    Returns:
        str: Extracted number portion (e.g., "0_0")
    """
    number = filename[13:-4] # change according to the naming of building DSM files
    return number


def compute_utci(building_dsm_path, tree_path, dem_path, walls_path, aspect_path, landcover_path, met_file,
                output_path,number,selected_date_str,save_tmrt=False,save_svf=False, save_kup=False,save_kdown=False,save_lup=False,save_ldown=False,save_shadow=False,
                use_energy_balance=False, use_coupled_eb=False,
                coupled_geometry_path=None, coupled_spinup_max_cycles=30,
                coupled_strict_convergence=True,
                z0=0.01, z_wind=None, ground_material_type='asphalt',
                save_qh=False, save_qe=False,
                use_roof_eb=False, roof_material_type='roof',
                use_canopy_eb=False, canopy_type='deciduous', canopy_lai=3.0,
                use_canyon_air_temp=False,
                save_tsfc=False, save_tsfc_roof=False, save_t_leaf=False,
                save_canopy_qh=False, save_canopy_qe=False,
                save_tair=False, save_tair_canyon=False,
                save_wall_temperature=False,
                use_wind_field=False, save_wind_speed=False,
                save_wind_direction=False,
                tair_method='conductance-blend',
                ground_wetness=None, canopy_gmax_fraction=None,
                output_start_index=0):
    """
    Compute UTCI and related thermal comfort outputs for a single tile.

    This is the main computation function that integrates shadow modeling, radiation
    calculations, and UTCI computation for urban microclimate analysis.

    Args:
        building_dsm_path (str): Path to Building DSM raster
        tree_path (str): Path to tree/vegetation DSM raster
        dem_path (str): Path to Digital Elevation Model raster
        walls_path (str): Path to wall height raster
        aspect_path (str): Path to wall aspect raster
        landcover_path (str): Path to land cover raster (can be None)
        met_file (str): Path to meteorological forcing file
        output_path (str): Directory for saving output rasters
        number (str): Tile identifier (e.g., "0_0")
        selected_date_str (str): Date string (YYYY-MM-DD)
        save_tmrt (bool): Save mean radiant temperature output
        save_svf (bool): Save sky view factor output
        save_kup (bool): Save upward shortwave radiation
        save_kdown (bool): Save downward shortwave radiation
        save_lup (bool): Save upward longwave radiation
        save_ldown (bool): Save downward longwave radiation
        save_shadow (bool): Save shadow maps

    Returns:
        None: Outputs are saved as GeoTIFF files in output_path

    Notes:
        - Automatically uses GPU if available
        - Outputs are multi-band rasters (one band per hour)
        - UTCI is always computed and saved
        - Other outputs are optional based on save_* flags
    """
    if (
        not isinstance(output_start_index, int)
        or isinstance(output_start_index, bool)
        or output_start_index < 0
        or output_start_index >= len(met_file)
    ):
        raise ValueError("output_start_index must identify a row in the forcing table")
    a, dataset = load_raster_to_tensor(building_dsm_path)
    temp1, dataset2 = load_raster_to_tensor(tree_path)
    temp2, dataset3 = load_raster_to_tensor(dem_path)
    walls, dataset4 = load_raster_to_tensor(walls_path)
    dirwalls, dataset5 = load_raster_to_tensor(aspect_path)
 
    landcover = 0

    if landcover_path is not None:
        landcover = 1
        lcgrid_torch, dataset6 = load_raster_to_tensor(landcover_path)
        lcgrid_raw = lcgrid_torch.detach().cpu().numpy()
        if not np.isfinite(lcgrid_raw).all() or not np.equal(lcgrid_raw, np.rint(lcgrid_raw)).all():
            raise ValueError("land-cover values must be finite integer class codes")
        lcgrid_np = lcgrid_raw.astype(np.int64)
        if ((lcgrid_np < 1) | (lcgrid_np > 7)).any():
            raise ValueError("land-cover class codes must be between 1 and 7")
        mask_vegetation = (lcgrid_np == 3) | (lcgrid_np == 4)
        if mask_vegetation.any():
            raise ValueError(
                "land-cover classes 3 and 4 describe canopy; provide the ground-cover "
                "class beneath vegetation instead"
            )
        lcgrid_torch = torch.from_numpy(lcgrid_np).to(device)

        with open(landcover_classes_path) as f:
            lines = f.readlines()[1:]                            # skip header line
        lc_class = np.empty((len(lines), 6), dtype=float)
        for i, ln in enumerate(lines):
            lc_class[i, :] = [float(x) for x in ln.split()[1:]]  # cols 1-6
    base_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
    rows, cols = a.shape
    geotransform = dataset.GetGeoTransform()
    scale = 1 / geotransform[1]
    projection_wkt = dataset.GetProjection()
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(projection_wkt) 
    old_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs84_wkt = """GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    new_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    widthx = dataset.RasterXSize
    heightx = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    centre_x = geotransform[0] + geotransform[1] * widthx  / 2.0
    centre_y = geotransform[3] + geotransform[5] * heightx / 2.0
    lon, lat = transform.TransformPoint(centre_x, centre_y)[:2]
    alt = float(torch.median(temp2).item())
    if not math.isfinite(alt):
        raise ValueError("DEM median elevation is not finite")
    location = {'longitude': lon, 'latitude': lat, 'altitude': alt}
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    local_tz = pytz.timezone(timezone_name)
    local_dt = local_tz.localize(base_date)
    utc = local_dt.utcoffset().total_seconds() / 3600
    print(f"[INFO] Timezone: {timezone_name}, UTC offset: {utc} hours")
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(met_file, location, utc)
    # SOLWEIG returns day-of-year decimal time. Convert it to a continuous
    # calendar ordinal so multi-day spin-up remains increasing across New Year;
    # the radiation routines use only its integer day grouping and fraction.
    dectime = np.asarray([
        (
            datetime.datetime(int(row[0]), 1, 1)
            + datetime.timedelta(days=int(row[1]) - 1)
        ).toordinal()
        + row[2] / 24.0
        + row[3] / (60.0 * 24.0)
        for row in met_file
    ])
    temp1[temp1 < 0.] = 0.
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = torch.maximum(a.max(), vegdem.max())
    buildings = a - temp2
    buildings[buildings < 2.] = 1.
    buildings[buildings >= 2.] = 0.
    valid_mask = (buildings == 1)
    Knight = torch.zeros((rows, cols), device=device)
    Tgmap1 = torch.zeros((rows, cols), device=device)
    Tgmap1E = torch.zeros((rows, cols), device=device)
    Tgmap1S = torch.zeros((rows, cols), device=device)
    Tgmap1W = torch.zeros((rows, cols), device=device)
    Tgmap1N = torch.zeros((rows, cols), device=device)
    TgOut1 = torch.zeros((rows, cols), device=device)
    
    if landcover == 1:                                     
        (TgK_np, Tstart_np, alb_np, emis_np, TgK_wall_np, Tstart_wall_np, TmaxLST_np,
         TmaxLST_wall_np) = Tgmaps_v1(lcgrid_np, lc_class)
           
        TgK           = torch.from_numpy(TgK_np).to(device).float()
        Tstart        = torch.from_numpy(Tstart_np).to(device).float()
        alb_grid      = torch.from_numpy(alb_np).to(device).float()
        emis_grid     = torch.from_numpy(emis_np).to(device).float()
        TgK_wall      = torch.tensor(float(np.ravel(TgK_wall_np)[0])     , device=device)
        Tstart_wall   = torch.tensor(float(np.ravel(Tstart_wall_np)[0])  , device=device)
        TmaxLST       = torch.from_numpy(TmaxLST_np ).to(device).float()
        TmaxLST_wall  = torch.tensor(float(np.ravel(TmaxLST_wall_np)[0]) , device=device)
    else:
        TgK = Knight + 0.37
        Tstart = Knight - 3.41
        alb_grid = Knight + albedo_g
        emis_grid = Knight + eground
        TgK_wall = 0.37
        Tstart_wall = -3.41
        TmaxLST = 15.
        TmaxLST_wall = 15.
    # SOLWEIG vegetation transmission used without canopy energy balance.
    transVeg = 3. / 100.
    if landcover == 1:
        lcgrid = lcgrid_torch
    else:
        lcgrid = False
    anisotropic_sky = 1
    patch_option = 2
    DOY = torch.tensor(met_file[:, 1], device=device)
    hours = torch.tensor(met_file[:, 2], device=device)
    minu = torch.tensor(met_file[:, 3], device=device)
    Ta = torch.tensor(met_file[:, 11], device=device)
    RH = torch.tensor(met_file[:, 10], device=device)
    radG = torch.tensor(met_file[:, 14], device=device)
    radD = torch.tensor(met_file[:, 21], device=device)
    radI = torch.tensor(met_file[:, 22], device=device)
    # Prefer measured diffuse and direct radiation when both are available.
    _has_measured_beam = bool(
        (radD > -998.0).all() and (radI > -998.0).all()
        and (radI > 0.0).any() and (radD > 0.0).any()
    )
    onlyglobal = 0 if _has_measured_beam else 1
    P = torch.tensor(met_file[:, 12], device=device)
    Rain = torch.tensor(met_file[:, 13], device=device)
    Ws = torch.tensor(met_file[:, 9], device=device)
    # Wind direction (column 23) — needed for spatial wind field
    Wd = torch.tensor(met_file[:, 23], device=device) if met_file.shape[1] > 23 else None
    # Prepare leafon based on vegetation type
    if conifer_bool:
        leafon = torch.ones((1, DOY.shape[0]), device=device)
    else:
        leafon = torch.zeros((1, DOY.shape[0]), device=device)
        if firstdayleaf > lastdayleaf:
            leaf_bool = ((DOY > firstdayleaf) | (DOY < lastdayleaf))
        else:
            leaf_bool = ((DOY > firstdayleaf) & (DOY < lastdayleaf))
        leafon[0, leaf_bool] = 1
    canopy_props = None
    _directional_canopy_optics = bool(use_energy_balance and use_canopy_eb)
    if _directional_canopy_optics:
        from .energy_balance.config import CanopyProperties
        from .energy_balance.canopy import (
            canopy_gap_transmittance,
            canopy_hemispherical_transmittance,
        )

        can_factory = {
            'deciduous': CanopyProperties.deciduous,
            'evergreen': CanopyProperties.evergreen,
        }
        canopy_props = can_factory[canopy_type]()
        # altitude/sky-patch angles arrive as NumPy, so these helpers default to
        # CPU. Pin them to the model device before mixing with device tensors.
        direct_tau = canopy_gap_transmittance(
            canopy_lai, canopy_props.clumping_factor, altitude[0],
        ).to(device)
        psi = torch.where(leafon > 0, direct_tau.unsqueeze(0), 0.5)
    else:
        psi = leafon * transVeg
        psi[leafon == 0] = 0.5
    Twater = []
    height = 1.1
    height = torch.tensor(height, device=device)
    #first = torch.round(torch.tensor(height, device=device))
    first = torch.round(height.clone().detach().to(device))
    if first == 0.:
        first = torch.tensor(1., device=device)
    second = torch.round(height * 20.)
    if len(Ta) == 1:
        timestepdec = 0
    else:
        timestepdec = dectime[1] - dectime[0]
    timeadd = 0.
    firstdaytime = 1.
    start_time = time.time()
    # Calculate SVF and related parameters.
    svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal = svf_calculator(patch_option, amaxvalue, a, vegdsm, vegdsm2, bush, scale)
    asvf = torch.acos(torch.sqrt(svf))
    diffsh = torch.zeros((rows, cols, shmat.shape[2]), device=device)
    diffsh_leaf_off = None
    if _directional_canopy_optics:
        sky_patch_altitude, _, _, _, _, _, _ = create_patches(patch_option)
        patch_tau = canopy_gap_transmittance(
            canopy_lai, canopy_props.clumping_factor, sky_patch_altitude,
        ).to(device)
        diffuse_tau = canopy_hemispherical_transmittance(
            canopy_lai, canopy_props.clumping_factor,
        ).to(device)
        svfbuveg = svf - (1.0 - svfveg) * (1.0 - diffuse_tau)
        svfbuveg_leaf_off = svf - (1.0 - svfveg) * 0.5
        diffsh_leaf_off = torch.zeros_like(diffsh)
        for patch_idx in range(shmat.shape[2]):
            vegetation_block = 1.0 - vegshmat[:, :, patch_idx]
            diffsh[:, :, patch_idx] = (
                shmat[:, :, patch_idx]
                - vegetation_block * (1.0 - patch_tau[patch_idx])
            ).clamp(0.0, 1.0)
            diffsh_leaf_off[:, :, patch_idx] = (
                shmat[:, :, patch_idx] - vegetation_block * 0.5
            ).clamp(0.0, 1.0)
    else:
        svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)
        svfbuveg_leaf_off = svfbuveg
        for patch_idx in range(shmat.shape[2]):
            diffsh[:, :, patch_idx] = (
                shmat[:, :, patch_idx]
                - (1.0 - vegshmat[:, :, patch_idx]) * (1.0 - transVeg)
            ).clamp(0.0, 1.0)
    tmp = svf + svfveg - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = torch.asin(torch.exp(torch.log(1.0 - tmp) / 2.0))

    # Initialize energy balance bridge (if enabled)
    eb_bridge = None
    if use_energy_balance:
        from .eb_bridge import EBBridge
        from .energy_balance import MaterialProperties, CanopyProperties, CanyonAirTempConfig

        # Resolve ground material preset (fallback for pixels without landcover)
        mat_factory = {
            'asphalt': MaterialProperties.asphalt,
            'concrete': MaterialProperties.concrete,
            'brick': MaterialProperties.brick,
            'grass': MaterialProperties.grass,
            'soil': MaterialProperties.soil,
            'dry_soil': MaterialProperties.dry_soil,
            'water': MaterialProperties.water,
        }
        mat_fn = mat_factory[ground_material_type]
        ground_mat = mat_fn()

        # Build per-pixel material table + IDs from landcover (same mapping as CPU)
        eb_material_ids = None
        material_table = None
        if landcover == 1:
            material_table = [
                MaterialProperties.asphalt(),   # 0: default
                MaterialProperties.asphalt(),   # 1: paved
                MaterialProperties.concrete(),  # 2: buildings (unused on ground)
                MaterialProperties.grass(),     # 3: evergreen trees
                MaterialProperties.grass(),     # 4: deciduous trees
                MaterialProperties.grass(),     # 5: grass
                MaterialProperties.dry_soil(),  # 6: bare soil
                MaterialProperties.water(),     # 7: water
            ]
            eb_material_ids = torch.from_numpy(lcgrid_np.astype(np.int64)).to(device)
            print(f"[EB] Landcover-driven materials: {len(material_table)} classes")

        # Roof material
        roof_mat = None
        if use_roof_eb:
            roof_factory = {
                'roof': MaterialProperties.roof,
                'green_roof': MaterialProperties.green_roof,
            }
            roof_fn = roof_factory[roof_material_type]
            roof_mat = roof_fn()
            print(f"[EB] Roof EB enabled: material={roof_material_type}")

        # Canopy properties
        canopy_mask_gpu = None
        lai_grid_gpu = None
        if use_canopy_eb:
            if canopy_props is None:
                can_factory = {
                    'deciduous': CanopyProperties.deciduous,
                    'evergreen': CanopyProperties.evergreen,
                }
                canopy_props = can_factory[canopy_type]()
            if canopy_gmax_fraction is not None:
                canopy_props.max_stomatal_conductance *= canopy_gmax_fraction
                print(f"[EB] canopy gmax scaled x{canopy_gmax_fraction} -> {canopy_props.max_stomatal_conductance:.3f} mm/s")
            # Build canopy mask from CDSM (tree heights > 0) — stays on GPU
            canopy_mask_gpu = temp1 > 0.0
            lai_grid_gpu = torch.where(canopy_mask_gpu, torch.full_like(temp1, canopy_lai), torch.zeros_like(temp1))
            n_canopy = int(canopy_mask_gpu.sum().item())
            print(f"[EB] Canopy EB enabled: type={canopy_type}, LAI={canopy_lai}, pixels={n_canopy}")

        # Canyon air temp config
        canyon_cfg = None
        if use_canyon_air_temp:
            bldg_mask = buildings < 0.5
            if bldg_mask.any():
                bldg_heights = (a - temp2)[bldg_mask]
                valid_heights = bldg_heights[bldg_heights > 2]
                z_h = float(valid_heights.mean().item()) if valid_heights.numel() > 0 else 10.0
            else:
                z_h = 10.0
            canyon_cfg = CanyonAirTempConfig(
                z_h=z_h, canyon_air_height=z_h * 0.6,
                z_ref=max(z_h + 5.0, 20.0),
            )
            print(f"[EB] Canyon air temp enabled: z_h={z_h:.1f}m")

        # Pass GPU tensors directly — no CPU transfer needed
        eb_bridge = EBBridge(
            ground_material=ground_mat,
            buildings=buildings,
            alb_grid=alb_grid,
            emis_grid=emis_grid,
            svf=svf,
            svfveg=svfveg,
            svfaveg=svfaveg,
            z0=z0,
            z_wind=z_wind,
            ewall=ewall,
            material_ids=eb_material_ids,
            material_table=material_table,
            enable_roof_eb=use_roof_eb,
            roof_material=roof_mat,
            canopy_properties=canopy_props,
            canopy_mask=canopy_mask_gpu,
            lai_grid=lai_grid_gpu,
            canyon_config=canyon_cfg,
            ground_wetness=ground_wetness,
            device=device,
        )
        lc_str = "landcover-driven" if eb_material_ids is not None else ground_material_type
        print(f"[EB] Energy balance enabled: material={lc_str}, z0={z0}, z_wind={z_wind}")

    coupled_bridge = None
    coupled_forcings = []
    coupled_body_shortwave = []
    if use_coupled_eb:
        from .coupled_pipeline import (
            CoupledRadiationBridge,
            ground_material_ids_from_trace,
            load_geometry_bundle,
            resolve_geometry_path,
        )
        from .energy_balance import CanopyProperties, MaterialProperties
        from .energy_balance.coupled import (
            CANOPY,
            GROUND,
            N_FACETS,
            ROOF,
            WALL_EAST,
            WALL_NORTH,
            WALL_SOUTH,
            WALL_WEST,
            UrbanForcing,
        )

        geometry_file = resolve_geometry_path(coupled_geometry_path, number)
        geometry_bundle = load_geometry_bundle(geometry_file, rows, cols, device)
        _validate_coupled_geometry_alignment(geometry_bundle, dataset)
        ground_factory = {
            'asphalt': MaterialProperties.asphalt,
            'concrete': MaterialProperties.concrete,
            'brick': MaterialProperties.brick,
            'grass': MaterialProperties.grass,
            'soil': MaterialProperties.soil,
            'dry_soil': MaterialProperties.dry_soil,
            'water': MaterialProperties.water,
        }
        roof_factory = {
            'roof': MaterialProperties.roof,
            'green_roof': MaterialProperties.green_roof,
        }
        canopy_factory = {
            'deciduous': CanopyProperties.deciduous,
            'evergreen': CanopyProperties.evergreen,
        }
        coupled_material_ids = None
        coupled_material_table = None
        wall_material = MaterialProperties.brick()
        wall_material.albedo = 0.20
        wall_material.emissivity = float(ewall)
        if landcover == 1:
            coupled_material_table = [
                MaterialProperties.asphalt(),
                MaterialProperties.asphalt(),
                MaterialProperties.concrete(),
                MaterialProperties.grass(),
                MaterialProperties.grass(),
                MaterialProperties.grass(),
                MaterialProperties.dry_soil(),
                MaterialProperties.water(),
            ]
            for class_id in range(1, 8):
                class_mask = lcgrid_torch == class_id
                if bool(class_mask.any()):
                    coupled_material_table[class_id].albedo = float(
                        alb_grid[class_mask][0].item()
                    )
                    coupled_material_table[class_id].emissivity = float(
                        emis_grid[class_mask][0].item()
                    )
            roof_index = len(coupled_material_table)
            coupled_material_table.append(roof_factory[roof_material_type]())
            wall_index = len(coupled_material_table)
            coupled_material_table.append(wall_material)
            coupled_material_ids = torch.full(
                (6, rows, cols), wall_index, dtype=torch.long, device=device
            )
            ground_material_ids = ground_material_ids_from_trace(
                geometry_bundle,
                lcgrid_np,
            )
            coupled_material_ids[GROUND] = torch.as_tensor(
                ground_material_ids,
                dtype=torch.long,
                device=device,
            )
            coupled_material_ids[ROOF] = roof_index
            print("[Coupled EB] per-pixel ground materials enabled from land cover")
        coupled_bridge = CoupledRadiationBridge(
            geometry_bundle,
            ground_material=ground_factory[ground_material_type](),
            roof_material=roof_factory[roof_material_type](),
            canopy_properties=canopy_factory[canopy_type](),
            dt=_met_timestep_seconds(met_file),
            spinup_max_cycles=coupled_spinup_max_cycles,
            strict_convergence=coupled_strict_convergence,
            wall_material=wall_material,
            material_ids=coupled_material_ids,
            material_table=coupled_material_table,
        )
        print(
            f"[Coupled EB] geometry={geometry_file}, "
            f"spinup_max_cycles={coupled_spinup_max_cycles}, "
            f"strict={coupled_strict_convergence}"
        )

    # Initialize diagnostic wind field model (if enabled)
    wind_model = None
    if use_wind_field and Wd is not None:
        from .wind_field import WindFieldModel, WindFieldConfig
        wind_cfg = WindFieldConfig()
        if z_wind is not None:
            wind_cfg.z_wind = z_wind
        wind_model = WindFieldModel(a, temp2, temp1, wind_cfg, device)
        print(f"[Wind] Diagnostic wind field enabled "
              f"({int(wind_model.building_mask.sum().item())} building pixels, "
              f"{int(wind_model.veg_mask.sum().item())} vegetation pixels)")

    # Prepare lists to store results for all time steps
    UTCI_all  = []
    TMRT_all  = []
    Kup_all   = []
    Kdown_all = []
    Lup_all   = []
    Ldown_all = []
    Shadow_all= []
    QH_all    = []
    QE_all    = []
    Tsfc_all = []
    TsfcRoof_all = []
    TLeaf_all = []
    CanopyQH_all = []
    CanopyQE_all = []
    Tair_all = []
    TairCanyon_all = []
    WindSpeed_all = []
    WindDir_all = []
    WallNorth_all = []
    WallEast_all = []
    WallSouth_all = []
    WallWest_all = []

    # Precompute the static terms for a TARGET-inspired conductance blend.
    # This is a pixel-scale diagnostic, not the complete TARGET model.
    _use_target = (tair_method in {'conductance-blend', 'target'}
                   and use_energy_balance and eb_bridge is not None)
    if _use_target:
        svf_clamped = svf.clamp(0.05, 0.999)
        _target_H_W = torch.tan(torch.acos(svf_clamped))
        _target_rho_cp = 1.2 * 1004.0
        _target_kappa = 0.4
        _target_z_ref = max(canyon_cfg.z_ref, 20.0) if use_canyon_air_temp and canyon_cfg is not None else 20.0
        _target_z_h = canyon_cfg.z_h if use_canyon_air_temp and canyon_cfg is not None else 10.0
        _target_z0_ca = max(0.1 * _target_z_h, 0.1)
        _target_ln_term = math.log(_target_z_ref / _target_z0_ca)
        print(f"[conductance blend] z_ref={_target_z_ref:.1f}m, z0_ca={_target_z0_ca:.1f}m, "
              f"z_h={_target_z_h:.1f}m, ln_term={_target_ln_term:.2f}")
        # Precompute Gaussian kernel for horizontal air mixing (~30m)
        _gk = 31 // 2
        _gx = torch.arange(-_gk, _gk + 1, dtype=torch.float32, device=device)
        _gkernel = torch.exp(-_gx ** 2 / (2 * 10.0 ** 2))
        _gkernel = _gkernel / _gkernel.sum()
        _gkernel_h = _gkernel.view(1, 1, 1, -1)
        _gkernel_v = _gkernel.view(1, 1, -1, 1)

    CI = 1.0
    for i in np.arange(0, Ta.__len__()):
        if landcover == 1:
            if np.isclose(dectime[i] - np.floor(dectime[i]), 0.0) or i == 0:
                Twater = _daily_mean_air_temperature(
                    Ta.cpu().numpy(), dectime, int(i)
                )
        if np.isclose(dectime[i] - np.floor(dectime[i]), 0.0):
            day_indices = _indices_for_current_day(dectime, int(i))
            above_horizon = np.flatnonzero(altitude[0, day_indices] > 1.0)
            if day_indices.size > 1 and above_horizon.size:
                sample_position = min(
                    int(above_horizon[0]) + 1, day_indices.size - 1
                )
                sample_index = int(day_indices[sample_position])
                [_, CI, _, _, _] = clearnessindex_2013b(
                    torch.as_tensor(
                        zen[0, sample_index], device=Ta.device, dtype=Ta.dtype
                    ),
                    DOY[sample_index],
                    Ta[sample_index], RH[sample_index] / 100.,
                    radG[sample_index], location, P[sample_index]
                )
                if (CI > 1.) or (CI == np.inf):
                    CI = 1.
            else:
                CI = 1.
        # Compute spatial wind field for this timestep (or use scalar)
        # advdiff air-temperature needs the wind *direction* field to advect along.
        _need_uv = save_wind_direction or tair_method == 'advdiff'
        wd_field_2d = None
        if wind_model is not None and Wd is not None:
            if _need_uv:
                ws_field_2d, wd_field_2d = wind_model.compute_wind_field_uv(float(Ws[i]), float(Wd[i]))
            else:
                ws_field_2d = wind_model.compute_wind_field(float(Ws[i]), float(Wd[i]))
            ws_for_calc = ws_field_2d   # 2D tensor for EB
        else:
            ws_field_2d = None
            ws_for_calc = Ws[i]
        if _directional_canopy_optics and leafon[0, i] <= 0:
            diffsh_timestep = diffsh_leaf_off
            svfbuveg_timestep = svfbuveg_leaf_off
        else:
            diffsh_timestep = diffsh
            svfbuveg_timestep = svfbuveg
        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, \
        KsideI, TgOut1, TgOut, radIout, radDout, Lside, Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside, \
        _qh_np, _qe_np = Solweig_2022a_calc(
            i, a, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl,
            altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location, psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], dirwalls, walls, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
            amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall, first, second, svfalfa, svfbuveg_timestep, firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N,
            CI, TgOut1, diffsh_timestep, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option,
            eb_bridge=eb_bridge, Ws=ws_for_calc)
        if coupled_bridge is not None:
            facet_shortwave = coupled_bridge.facet_shortwave_irradiance(
                direct_normal_irradiance=max(float(radIout.item()), 0.0),
                diffuse_horizontal_irradiance=max(float(radDout.item()), 0.0),
                solar_altitude_degrees=float(altitude[0][i]),
                solar_azimuth_degrees=float(azimuth[0][i]),
            )
            body_shortwave = absK * (
                torch.nan_to_num(Kside, nan=0.0) * Fcyl
                + (torch.nan_to_num(Kdown, nan=0.0) + torch.nan_to_num(Kup, nan=0.0)) * Fup
                + (
                    torch.nan_to_num(Knorth, nan=0.0)
                    + torch.nan_to_num(Keast, nan=0.0)
                    + torch.nan_to_num(Ksouth, nan=0.0)
                    + torch.nan_to_num(Kwest, nan=0.0)
                ) * Fside
            )
            coupled_dt = coupled_bridge.model.config.dt
            rain_rate = max(float(Rain[i].item()), 0.0) / coupled_dt
            coupled_forcings.append(
                UrbanForcing(
                    air_temperature=float(Ta[i].item()) + 273.15,
                    vapor_pressure_kpa=float(ea.item() if isinstance(ea, torch.Tensor) else ea) * 0.1,
                    pressure_kpa=float(P[i].item()),
                    wind_speed=ws_for_calc,
                    sky_longwave=float(esky.item() if isinstance(esky, torch.Tensor) else esky)
                    * 5.67051e-8
                    * (float(Ta[i].item()) + 273.15) ** 4,
                    shortwave_irradiance=facet_shortwave,
                    precipitation_rate=rain_rate,
                    rain_capture_fraction=coupled_bridge.rain_capture_fraction,
                )
            )
            coupled_body_shortwave.append(body_shortwave)
        # Create matrices for meteorological parameters for the current time step
        # Spatially-varying air temperature
        if _use_target and eb_bridge.last_tsfc_ground is not None:
            # TARGET-inspired surface/free-stream conductance blend.
            Tsfc_ground = eb_bridge.last_tsfc_ground  # (rows, cols) in Kelvin

            # Use the raster wind diagnostic if available, otherwise met wind.
            if ws_field_2d is not None:
                U_ref_field = ws_field_2d.clamp(min=0.5)
            else:
                U_ref_field = max(Ws[i].item(), 0.5)

            U_canyon = (U_ref_field * torch.exp(-0.386 * _target_H_W)).clamp(min=0.3)
            cs = (11.8 + 4.2 * U_canyon) / _target_rho_cp

            if ws_field_2d is not None:
                # Per-pixel ca using spatial wind
                ca = (_target_kappa ** 2 * U_ref_field) / (_target_ln_term ** 2)
            else:
                ca = (_target_kappa ** 2 * max(Ws[i].item(), 0.5)) / (_target_ln_term ** 2)

            Ta_K = Ta[i] + 273.15
            Ta_spatial = (torch.nan_to_num(Tsfc_ground, nan=Ta_K.item()) * cs + Ta_K * ca) / (cs + ca)
            Ta_spatial = (Ta_spatial - 273.15).clamp(Ta[i].item() - 5.0, Ta[i].item() + 10.0)
            Ta_spatial[~valid_mask] = Ta[i].item()

            # Gaussian smoothing (~30m) for horizontal air mixing
            t = Ta_spatial.unsqueeze(0).unsqueeze(0)
            t = F.pad(t, (_gk, _gk, 0, 0), mode='replicate')
            t = F.conv2d(t, _gkernel_h)
            t = F.pad(t, (0, 0, _gk, _gk), mode='replicate')
            t = F.conv2d(t, _gkernel_v)
            Ta_mat = t.squeeze(0).squeeze(0)
        elif tair_method == 'advdiff' and eb_bridge is not None:
            # Prognostic advection-diffusion of the canopy-layer air field.
            # Runs every timestep to maintain the persistent field state (memory),
            # advected by the URock wind (or uniform met wind if no wind model).
            if ws_field_2d is not None and wd_field_2d is not None:
                _sp, _di = ws_field_2d, wd_field_2d
            else:
                _sp, _di = float(Ws[i].item()), float(Wd[i].item()) if Wd is not None else 0.0
            tair_ad = eb_bridge.compute_tair_advdiff(float(Ta[i].item()), _sp, _di, scale)
            Ta_mat = tair_ad - 273.15
            Ta_mat = torch.where(valid_mask, Ta_mat, torch.zeros_like(Ta_mat) + Ta[i])
        elif tair_method == 'log-profile' and eb_bridge is not None and save_tair:
            tair_2m = eb_bridge.compute_tair_2m(float(Ta[i].item()))
            Ta_mat = tair_2m - 273.15
            Ta_mat = torch.where(valid_mask, Ta_mat, torch.zeros_like(Ta_mat) + Ta[i])
        else:
            Ta_mat = torch.zeros((rows, cols), device=device) + Ta[i]
        RH_mat = torch.zeros((rows, cols), device=device) + RH[i]
        Tmrt_mat = torch.zeros((rows, cols), device=device) + Tmrt
        # UTCI requires wind measured at 10 m. The diagnostic urban wind field
        # is evaluated at pedestrian height (1.5 m) and must not be substituted.
        va10m_mat = torch.zeros((rows, cols), device=device) + Ws[i]
        UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt_mat, va10m_mat)
        UTCI = torch.full(UTCI_mat.shape, float('nan'), device=device)
        UTCI[valid_mask] = UTCI_mat[valid_mask]
        # Advance state on every spin-up and output timestep.
        if eb_bridge is not None:
            if use_canyon_air_temp and _qh_np is not None:
                eb_bridge.update_canyon_air_temp(_qh_np, Ta[i], ws_for_calc)

        # Spin-up records update persistent state but are never written.
        if i >= output_start_index:
            UTCI_all.append(UTCI.cpu().numpy())
            TMRT_all.append(Tmrt.cpu().numpy())
            Kup_all.append(Kup.cpu().numpy())
            Kdown_all.append(Kdown.cpu().numpy())
            Lup_all.append(Lup.cpu().numpy())
            Ldown_all.append(Ldown.cpu().numpy())
            Shadow_all.append(shadow.cpu().numpy())
            if _qh_np is not None:
                QH_all.append(
                    _qh_np.cpu().numpy()
                    if isinstance(_qh_np, torch.Tensor) else _qh_np
                )
                QE_all.append(
                    _qe_np.cpu().numpy()
                    if isinstance(_qe_np, torch.Tensor) else _qe_np
                )

            # Collect roof/canopy diagnostics from the requested day only.
            if eb_bridge is not None:
                if save_tsfc and eb_bridge.last_tsfc_ground is not None:
                    Tsfc_all.append(eb_bridge.last_tsfc_ground.cpu().numpy() - 273.15)
                if save_tsfc_roof and eb_bridge.last_tsfc_roof is not None:
                    TsfcRoof_all.append(eb_bridge.last_tsfc_roof.cpu().numpy() - 273.15)
                if save_t_leaf and eb_bridge.last_t_leaf is not None:
                    TLeaf_all.append(eb_bridge.last_t_leaf.cpu().numpy() - 273.15)
                if save_canopy_qh and eb_bridge.last_canopy_qh is not None:
                    CanopyQH_all.append(eb_bridge.last_canopy_qh.cpu().numpy())
                if save_canopy_qe and eb_bridge.last_canopy_qe is not None:
                    CanopyQE_all.append(eb_bridge.last_canopy_qe.cpu().numpy())
                if save_tair_canyon and eb_bridge.tcan_prev is not None:
                    tcan_field = torch.full((rows, cols), float('nan'), device=device)
                    ground_mask = buildings > 0.5
                    tcan_field[ground_mask] = eb_bridge.tcan_prev - 273.15
                    TairCanyon_all.append(tcan_field.cpu().numpy())
            if save_tair:
                Tair_out = torch.full_like(Ta_mat, float('nan'))
                Tair_out[valid_mask] = Ta_mat[valid_mask]
                Tair_all.append(Tair_out.cpu().numpy())

            if save_wind_speed and ws_field_2d is not None:
                WindSpeed_all.append(ws_field_2d.cpu().numpy())
            elif save_wind_speed:
                WindSpeed_all.append(
                    (torch.zeros((rows, cols), device=device) + Ws[i]).cpu().numpy()
                )

            if save_wind_direction and wd_field_2d is not None:
                WindDir_all.append(wd_field_2d.cpu().numpy())
            elif save_wind_direction and Wd is not None:
                WindDir_all.append(
                    (torch.zeros((rows, cols), device=device) + Wd[i]).cpu().numpy()
                )

    if coupled_bridge is not None:
        # Repeating target-day rainfall during every periodic spin-up cycle is
        # not observed antecedent weather.  The public coupled path therefore
        # uses a declared dry-antecedent spin-up, while retaining precipitation
        # in the requested output cycle.
        spinup_forcings = _dry_coupled_spinup_forcings(coupled_forcings)
        coupled_results, coupled_radiation = coupled_bridge.solve_cycle(
            coupled_forcings,
            coupled_body_shortwave,
            spinup_forcings=spinup_forcings,
        )
        spinup = coupled_bridge.last_spinup
        print(
            f"[Coupled EB] spin-up cycles={spinup.cycles}, "
            f"converged={spinup.converged}, "
            f"max_temperature_drift={spinup.maximum_temperature_drift:.4g} K, "
            "dry-antecedent spin-up"
        )
        UTCI_all = []
        TMRT_all = []
        Lup_all = []
        QH_all = []
        QE_all = []
        Tsfc_all = []
        TsfcRoof_all = []
        TLeaf_all = []
        CanopyQH_all = []
        CanopyQE_all = []
        Tair_all = []
        TairCanyon_all = []
        WallNorth_all = []
        WallEast_all = []
        WallSouth_all = []
        WallWest_all = []
        facet_area = coupled_bridge.bundle.geometry.area

        def _facet_output(value, facet):
            return torch.where(
                facet_area[facet] > 0.0,
                value[facet],
                torch.full_like(value[facet], float('nan')),
            )

        for index, (result, radiation_result) in enumerate(
            zip(coupled_results, coupled_radiation)
        ):
            canyon_air_c = result.state.canyon_air_temperature - 273.15
            rh_canyon = radiation_result.relative_humidity_percent
            tmrt = radiation_result.tmrt_celsius
            va10m = torch.zeros((rows, cols), device=device) + Ws[index]
            utci_matrix = utci_calculator(canyon_air_c, rh_canyon, tmrt, va10m)
            utci = torch.full_like(utci_matrix, float('nan'))
            utci[valid_mask] = utci_matrix[valid_mask]
            UTCI_all.append(utci.cpu().numpy())
            TMRT_all.append(tmrt.cpu().numpy())
            Lup_all.append(radiation_result.upward_longwave.cpu().numpy())
            QH_all.append(_facet_output(result.sensible_heat, GROUND).cpu().numpy())
            QE_all.append(_facet_output(result.latent_heat, GROUND).cpu().numpy())
            Tsfc_all.append(
                (_facet_output(result.state.surface_temperature, GROUND) - 273.15).cpu().numpy()
            )
            TsfcRoof_all.append(
                (_facet_output(result.state.surface_temperature, ROOF) - 273.15).cpu().numpy()
            )
            TLeaf_all.append(
                (_facet_output(result.state.surface_temperature, CANOPY) - 273.15).cpu().numpy()
            )
            CanopyQH_all.append(
                _facet_output(result.sensible_heat, CANOPY).cpu().numpy()
            )
            CanopyQE_all.append(
                _facet_output(result.latent_heat, CANOPY).cpu().numpy()
            )
            tair_field = torch.where(
                valid_mask,
                canyon_air_c,
                torch.full_like(canyon_air_c, float('nan')),
            )
            Tair_all.append(tair_field.cpu().numpy())
            TairCanyon_all.append(tair_field.cpu().numpy())
            WallNorth_all.append(
                (_facet_output(result.state.surface_temperature, WALL_NORTH) - 273.15).cpu().numpy()
            )
            WallEast_all.append(
                (_facet_output(result.state.surface_temperature, WALL_EAST) - 273.15).cpu().numpy()
            )
            WallSouth_all.append(
                (_facet_output(result.state.surface_temperature, WALL_SOUTH) - 273.15).cpu().numpy()
            )
            WallWest_all.append(
                (_facet_output(result.state.surface_temperature, WALL_WEST) - 273.15).cpu().numpy()
            )

    # Convert the lists to numpy arrays with shape (time_steps, rows, cols)
    UTCI_all  = np.array(UTCI_all)
    TMRT_all  = np.array(TMRT_all)
    Kup_all   = np.array(Kup_all)
    Kdown_all = np.array(Kdown_all)
    Lup_all   = np.array(Lup_all)
    Ldown_all = np.array(Ldown_all)
    Shadow_all= np.array(Shadow_all)

    # Helper to reconstruct timestamp from met file year + DOY (supports multi-day)
    def _make_timestamp(band):
        source_band = output_start_index + band
        h = int(hours[source_band].cpu().item())
        m = int(minu[source_band].cpu().item())
        y = int(met_file[source_band, 0])
        d = int(DOY[source_band].cpu().item())
        dt = datetime.datetime(y, 1, 1) + datetime.timedelta(days=d - 1)
        return dt.replace(hour=h, minute=m).isoformat()

    if QH_all:
        QH_all = np.array(QH_all)
        QE_all = np.array(QE_all)
    # Write a multi-band GeoTIFF for UTCI (each band corresponds to one time step)
    driver = gdal.GetDriverByName('GTiff')
    out_file_path = os.path.join(output_path, f'UTCI_{number}.tif')
    num_bands = UTCI_all.shape[0]
    out_dataset = driver.Create(out_file_path, cols, rows, num_bands, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    for band in range(num_bands):
        out_band = out_dataset.GetRasterBand(band + 1)
        out_band.WriteArray(UTCI_all[band])
        out_band.FlushCache()
        timestamp = _make_timestamp(band)
        out_band.SetMetadata({'Time': timestamp})
    out_dataset = None
    # Optionally, you can similarly write TMRT to a single multi-band file:
    if save_tmrt:
        out_file_path_op = os.path.join(output_path, f'TMRT_{number}.tif')
        num_bands_op = TMRT_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(TMRT_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_svf:
        out_file_path_op = os.path.join(output_path, f'SVF_{number}.tif')
        SVF = svftotal.cpu().numpy()
        SVF = np.array(SVF)
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, 1, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        out_band = out_dataset_op.GetRasterBand(1)
        out_band.WriteArray(SVF)
        out_band.FlushCache()
        out_dataset_op = None
    if save_kup:
        out_file_path_op = os.path.join(output_path, f'Kup_{number}.tif')
        num_bands_op = Kup_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Kup_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_kdown:
        out_file_path_op = os.path.join(output_path, f'Kdown_{number}.tif')
        num_bands_op = Kdown_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Kdown_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_lup:
        out_file_path_op = os.path.join(output_path, f'Lup_{number}.tif')
        num_bands_op = Lup_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Lup_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_ldown:
        out_file_path_op = os.path.join(output_path, f'Ldown_{number}.tif')
        num_bands_op = Ldown_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Ldown_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_shadow:
        out_file_path_op = os.path.join(output_path, f'Shadow_{number}.tif')
        num_bands_op = Shadow_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Shadow_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    if save_qh and isinstance(QH_all, np.ndarray) and QH_all.size > 0:
        out_file_path_op = os.path.join(output_path, f'QH_{number}.tif')
        num_bands_op = QH_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(QH_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_qe and isinstance(QE_all, np.ndarray) and QE_all.size > 0:
        out_file_path_op = os.path.join(output_path, f'QE_{number}.tif')
        num_bands_op = QE_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(QE_all[band])
            out_band.FlushCache()
            timestamp = _make_timestamp(band)
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    # Write roof/canopy diagnostic TIFs
    for var_name, var_list, save_flag in [
        ('Tsfc', Tsfc_all, save_tsfc),
        ('TsfcRoof', TsfcRoof_all, save_tsfc_roof),
        ('TLeaf', TLeaf_all, save_t_leaf),
        ('CanopyQH', CanopyQH_all, save_canopy_qh),
        ('CanopyQE', CanopyQE_all, save_canopy_qe),
        ('Tair', Tair_all, save_tair),
        ('TairCanyon', TairCanyon_all, save_tair_canyon),
        ('TsfcWallNorth', WallNorth_all, save_wall_temperature),
        ('TsfcWallEast', WallEast_all, save_wall_temperature),
        ('TsfcWallSouth', WallSouth_all, save_wall_temperature),
        ('TsfcWallWest', WallWest_all, save_wall_temperature),
        ('WindSpeed', WindSpeed_all, save_wind_speed),
        ('WindDir', WindDir_all, save_wind_direction),
    ]:
        if save_flag and len(var_list) > 0:
            var_arr = np.array(var_list)
            out_file_path_op = os.path.join(output_path, f'{var_name}_{number}.tif')
            num_bands_op = var_arr.shape[0]
            out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
            out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
            out_dataset_op.SetProjection(dataset.GetProjection())
            for band in range(num_bands_op):
                out_band = out_dataset_op.GetRasterBand(band + 1)
                out_band.WriteArray(var_arr[band])
                out_band.FlushCache()
                timestamp = _make_timestamp(band)
                out_band.SetMetadata({'Time': timestamp})
            out_dataset_op = None

    dataset = None
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to execute tile {number}: {time_taken:.2f} seconds")
