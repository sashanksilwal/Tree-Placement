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

import glob
import datetime
import math
import os
import re
import shutil

import pytz
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from netCDF4 import Dataset, date2num
import datetime as dt
from datetime import timedelta
from osgeo import gdal, osr
from shapely.geometry import box, Polygon
from matplotlib.path import Path
from timezonefinder import TimezoneFinder
from scipy.spatial import cKDTree

gdal.UseExceptions()

WRF_PATTERNS = [
    re.compile(r'^wrfout_d0([1-9])_(\d{4}-\d{2}-\d{2})_(\d{2}_\d{2}_\d{2})$'),  # HH_MM_SS
    re.compile(r'^wrfout_d0([1-9])_(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})$'),  # HH:MM:SS
    re.compile(r'^wrfout_d0([1-9])_(\d{4}-\d{2}-\d{2})_(\d{2})$'),              # HH
]

def _match_wrfout(base):
    """
    Match WRF output filename against known patterns.
    
    Args:
        base (str): Base filename to match
    
    Returns:
        tuple: (pattern_index, match_object) or (None, None) if no match
    """
    for i, rx in enumerate(WRF_PATTERNS):
        m = rx.match(base)
        if m:
            return i, m
    return None, None

def extract_datetime_strict(filename):
    """
    Return (datetime, domain_int) for strictly valid wrfout names.
    Raises ValueError for any non-matching filename.
    """
    base = os.path.basename(filename)
    idx, m = _match_wrfout(base)
    if m is None:
        raise ValueError(f"Unsupported wrfout filename: {base}")

    dom = int(m.group(1))         
    date = m.group(2)             
    t    = m.group(3)              

    if idx == 0:  # HH_MM_SS
        hh, mm, ss = map(int, t.split('_'))
        dt = datetime.datetime(
            int(date[0:4]), int(date[5:7]), int(date[8:10]), hh, mm, ss
        )
    elif idx == 1:  # HH:MM:SS
        dt = datetime.datetime.strptime(f"{date}_{t}", "%Y-%m-%d_%H:%M:%S")
    else:  # idx == 2, HH only
        dt = datetime.datetime.strptime(f"{date}_{t}", "%Y-%m-%d_%H")

    return dt, dom

# =============================================================================
# Function to check that all raster files have matching dimensions, pixel size, and CRS.
# =============================================================================
def check_rasters(files):
    """
    Check that all provided raster files have matching dimensions, pixel size, and CRS.

    Parameters:
        files (list): List of raster file paths.

    Returns:
        bool: True if all checks pass, raises ValueError/FileNotFoundError otherwise.
    """
    if not files:
        raise ValueError("No raster files provided.")

    ref_file = files[0]
    ds = gdal.Open(ref_file)
    if ds is None:
        raise FileNotFoundError(f"Could not open {ref_file}")
    ref_width = ds.RasterXSize
    ref_height = ds.RasterYSize
    ref_gt = ds.GetGeoTransform()  # (originX, pixelWidth, rot, originY, rot, pixelHeight)
    ref_crs = ds.GetProjection()
    if not ref_crs:
        raise ValueError(f"Raster has no CRS: {ref_file}")
    ds = None

    for f in files[1:]:
        ds = gdal.Open(f)
        if ds is None:
            raise FileNotFoundError(f"Could not open {f}")
        if ds.RasterXSize != ref_width or ds.RasterYSize != ref_height:
            raise ValueError(f"Raster dimensions do not match: {f}")
        gt = ds.GetGeoTransform()
        if gt != ref_gt:
            raise ValueError(f"Raster transform does not match: {f}")
        if ds.GetProjection() != ref_crs:
            raise ValueError(f"Raster CRS does not match: {f}")
        ds = None

    return True

# =============================================================================
# Function to tile a raster file into smaller chunks.
# =============================================================================
def create_tiles(infile, tilesize, overlap, tile_type, preprocess_dir):
    """
    Tile a raster file into smaller chunks.

    Parameters:
        infile (str): Path to input raster.
        tilesize (int): Size of each tile in pixels.
        overlap (int): Number of pixels to overlap between tiles.
        tile_type (str): Label to use for naming output tiles.
        preprocess_dir (str): Directory to save tiles in the pre_processing_outputs folder.

    Raises:
        FileNotFoundError: If the input file is not found.
        ValueError: If the overlap is not within the valid range.
    """
    ds = gdal.Open(infile)

    if overlap < 0 or overlap >= tilesize:
        raise ValueError("overlap must be 0 ≤ overlap < tilesize")
    
    if ds is None:
        raise FileNotFoundError(f"Could not open {infile}")

    width = ds.RasterXSize
    height = ds.RasterYSize

    out_folder = os.path.join(preprocess_dir, tile_type)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        shutil.rmtree(out_folder)
        os.makedirs(out_folder)

    if tilesize >= width and tilesize >= height:
        outfile = os.path.join(out_folder, f"{tile_type}_0_0.tif")
        options = gdal.TranslateOptions(format='GTiff', srcWin=[0, 0, width, height])
        gdal.Translate(outfile, ds, options=options)
        print(f"Created single tile (original file): {outfile}")
        ds = None
        return

    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            tile_width = min(tilesize + overlap, width - i)
            tile_height = min(tilesize + overlap, height - j)
            outfile = os.path.join(out_folder, f"{tile_type}_{i}_{j}.tif")
            options = gdal.TranslateOptions(format='GTiff', srcWin=[i, j, tile_width, tile_height])
            gdal.Translate(outfile, ds, options=options)
            print(f"Created tile: {outfile}")
   
    ds = None

def _normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy with a proper 'time' coord from 'valid_time' or 'time'+'step'."""
    ds = ds.copy()

    # Case 1: valid_time -> time
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})

    # Case 2: time + step
    has_time = ("time" in ds.dims) or ("time" in ds.coords)
    has_step = ("step" in ds.dims) or ("step" in ds.coords)
    if has_time and has_step:
        base = ds["time"]
        step = ds["step"]
        # ensure timedelta dtype
        if not np.issubdtype(step.dtype, np.timedelta64):
            try:
                step_td = xr.DataArray(pd.to_timedelta(step.values),
                                       dims=step.dims, coords=step.coords)
            except Exception:
                step_td = xr.DataArray(pd.to_timedelta(step.values, unit="h"),
                                       dims=step.dims, coords=step.coords)
        else:
            step_td = step

        vt = (base + step_td).rename("time")
        if vt.ndim == 2:
            vt_flat = vt.stack(time_flat=("time", "step")).rename("time")
            ds = ds.stack(time_flat=("time", "step")).rename_dims({"time_flat": "time"}).drop_vars(["time", "step"])
            ds = ds.assign_coords(time=vt_flat.values)
        else:
            if "step" in vt.dims and "time" not in vt.dims:
                ds = ds.swap_dims({"step": "time"}).drop_vars(["step"])
            ds = ds.assign_coords(time=vt.values)
        return ds

    # Case 3: plain time
    if has_time:
        try:
            ds = xr.decode_cf(ds)
        except Exception:
            pass
        return ds

    raise KeyError(
        "No usable time coordinate. Expected 'valid_time' or 'time' (+ optional 'step'). "
        f"Found dims={list(ds.dims)}, coords={list(ds.coords)}"
    )

def process_era5_data(start_time, end_time, folder_path, output_file="Outfile.nc"):
    """
    Read timestamps from the files and slice to [start_time, end_time] UTC. Variables written:
      T2 (t2m, Kelvin), PSFC (sp, Pa), RH2 (%), WIND (m/s), SWDOWN (W/m^2 = ssrd/3600).
    """
    # Parse UTC times
    start_dt = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt   = dt.datetime.strptime(end_time,   "%Y-%m-%d %H:%M:%S")

    # Open and normalize time coordinates
    instant_file = os.path.join(folder_path, 'data_stream-oper_stepType-instant.nc')
    accum_file   = os.path.join(folder_path, 'data_stream-oper_stepType-accum.nc')

    ds_instant = _normalize_time_coord(xr.open_dataset(instant_file, decode_times=True))
    ds_accum   = _normalize_time_coord(xr.open_dataset(accum_file,   decode_times=True))

    # Slice both to requested window present in files
    ds_instant = ds_instant.sel(time=slice(start_dt, end_dt))
    ds_accum   = ds_accum.sel(time=slice(start_dt, end_dt))

    if ds_instant.time.size == 0:
        raise ValueError("No instantaneous timestamps within requested window.")
    if ds_accum.time.size == 0:
        raise ValueError("No accumulated timestamps within requested window.")

    # Align to exactly overlapping timestamps (hourly)
    common_t = np.intersect1d(ds_instant.time.values, ds_accum.time.values)
    if common_t.size == 0:
        raise ValueError("Instantaneous and accumulated streams have no overlapping timestamps.")
    ds_instant = ds_instant.sel(time=common_t)
    ds_accum   = ds_accum.sel(time=common_t)

    def saturation_vapor_pressure(Tc):
        """
        Calculate saturation vapor pressure using Magnus formula.
        
        Args:
            Tc (float or np.ndarray): Temperature in Celsius
        
        Returns:
            float or np.ndarray: Saturation vapor pressure in hPa
        """
        # Tc in °C → hPa
        return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

    temperatures = ds_instant['t2m'].values              # K (kept as-is)
    dew_points   = ds_instant['d2m'].values              # K
    u10 = ds_instant['u10'].values
    v10 = ds_instant['v10'].values
    wind_speeds = np.sqrt(u10**2 + v10**2)

    # Hourly totals (J m^-2 per hour) → W m^-2 by /3600
    shortwave_radiation = (ds_accum['ssrd'].values / 3600.0).astype(np.float32)
    # longwave_radiation  = (ds_accum['strd'].values / 3600.0).astype(np.float32)  # not written

    # RH using Kelvin inputs converted to °C for the formula
    e_temp      = saturation_vapor_pressure(temperatures - 273.15)
    e_dew_point = saturation_vapor_pressure(dew_points   - 273.15)
    relative_humidities = 100.0 * (e_dew_point / e_temp)
    relative_humidities = np.clip(relative_humidities, 0, 100)

    # Lat/Lon grids (same behavior)
    latitudes  = ds_instant['latitude'].values
    longitudes = ds_instant['longitude'].values
    if latitudes.ndim == 1 and longitudes.ndim == 1:
        lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    else:
        lat2d = latitudes
        lon2d = longitudes

    # Use dataset times (no synthetic list)
    times = ds_instant.time.values

    # Write the intermediate NetCDF.
    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', times.shape[0])
        nc.createDimension('lat',  lat2d.shape[0])
        nc.createDimension('lon',  lon2d.shape[1])

        time_var = nc.createVariable('time', 'f8', ('time',))
        lat_var  = nc.createVariable('lat',  'f4', ('lat', 'lon'))
        lon_var  = nc.createVariable('lon',  'f4', ('lat', 'lon'))

        t2_var     = nc.createVariable('T2',     'f4', ('time', 'lat', 'lon'), zlib=True)
        psfc_var   = nc.createVariable('PSFC',   'f4', ('time', 'lat', 'lon'), zlib=True)
        rh2_var    = nc.createVariable('RH2',    'f4', ('time', 'lat', 'lon'), zlib=True)
        wind_var   = nc.createVariable('WIND',   'f4', ('time', 'lat', 'lon'), zlib=True)
        swdown_var = nc.createVariable('SWDOWN', 'f4', ('time', 'lat', 'lon'), zlib=True)
        # glw_var  = nc.createVariable('GLW',    'f4', ('time', 'lat', 'lon'), zlib=True)

        time_var.units = "hours since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        t2_var.units = "K"
        psfc_var.units = "Pa"
        rh2_var.units = "%"
        wind_var.units = "m/s"
        swdown_var.units = "W/m^2"
        # glw_var.units = "W/m^2"

        # Encode time from dataset timestamps
        py_times = [np.datetime64(t).astype("datetime64[s]").astype(object) for t in times]
        time_var[:] = date2num(py_times, units=time_var.units, calendar=time_var.calendar)

        lat_var[:, :] = lat2d.astype('float32')
        lon_var[:, :] = lon2d.astype('float32')

        # Ensure (time, lat, lon) ordering
        # xarray variables are typically (time, latitude, longitude)
        t2 = ds_instant['t2m'].transpose('time','latitude','longitude').values.astype('float32')
        sp = ds_instant['sp'].transpose('time','latitude','longitude').values.astype('float32')
        rh = xr.DataArray(relative_humidities,
                          dims=('time','latitude','longitude'),
                          coords={'time': times,
                                  'latitude': ds_instant['latitude'],
                                  'longitude': ds_instant['longitude']}
                         ).transpose('time','latitude','longitude').values.astype('float32')
        wind = xr.DataArray(wind_speeds,
                            dims=('time','latitude','longitude'),
                            coords={'time': times,
                                    'latitude': ds_instant['latitude'],
                                    'longitude': ds_instant['longitude']}
                           ).transpose('time','latitude','longitude').values.astype('float32')
        swd = xr.DataArray(shortwave_radiation,
                           dims=('time','latitude','longitude'),
                           coords={'time': times,
                                   'latitude': ds_instant['latitude'],
                                   'longitude': ds_instant['longitude']}
                          ).transpose('time','latitude','longitude').values.astype('float32')

        t2_var[:, :, :]     = t2
        psfc_var[:, :, :]   = sp
        rh2_var[:, :, :]    = rh
        wind_var[:, :, :]   = wind
        swdown_var[:, :, :] = swd
        # glw_var[:, :, :]  = ...

    print(f"ERA5 forcing file created: {output_file}  ({len(times)} steps from {py_times[0]} to {py_times[-1]} UTC)")

    
# =============================================================================
#    The function will:
#        - Populate the list of available WRF output files (names starting with 'wrfout')
#          and sort them based on the datetime string embedded in the filename.
#        - Loop over the sorted files and extract variables:
#            - 2-meter temperature (T2)
#            - Mixing ratio at 2 m (Q2)
#            - Surface pressure (PSFC)
#            - Land surface temperature (TSK)
#            - Downwelling shortwave radiation (SWDOWN)
#            - Downwelling longwave radiation (GLW)
#            - U and V wind components (U10, V10) to compute wind speed
#        - Calculate relative humidity using a helper function.
#        - Generate an hourly time array between start_time and end_time.
#        - Combine the data from all files along the time axis and save to a new NetCDF file.
# =============================================================================

def _select_wrf_files(folder_path, start_time, end_time):
    """Return one WRF domain with exactly the requested hourly timestamps."""
    records = []
    for filename in os.listdir(folder_path):
        try:
            timestamp, domain = extract_datetime_strict(filename)
        except ValueError:
            continue
        if start_time <= timestamp <= end_time:
            records.append((timestamp, domain, filename))

    if not records:
        raise FileNotFoundError(
            "No wrfout files matching the requested interval and supported filename patterns"
        )
    domains = {record[1] for record in records}
    if len(domains) != 1:
        raise ValueError(
            "WRF forcing must come from one domain; found domains "
            + ", ".join(str(domain) for domain in sorted(domains))
        )

    records.sort()
    timestamps = [record[0] for record in records]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("WRF forcing contains duplicate timestamps")
    total_hours = int((end_time - start_time).total_seconds() // 3600) + 1
    expected = [start_time + timedelta(hours=index) for index in range(total_hours)]
    if timestamps != expected:
        missing = [timestamp.isoformat() for timestamp in expected if timestamp not in timestamps]
        extra = [timestamp.isoformat() for timestamp in timestamps if timestamp not in expected]
        raise ValueError(
            f"WRF forcing timestamps do not match the requested hourly sequence; "
            f"missing={missing}, extra={extra}"
        )
    return [record[2] for record in records], timestamps


def process_wrfout_data(start_time, end_time, folder_path, output_file="Outfile.nc"):
    """
    Process WRF output files to create meteorological forcing data.

    Parameters:
        start_time (str): Start datetime string in format "%Y-%m-%d %H:%M:%S".
        end_time (str): End datetime string in format "%Y-%m-%d %H:%M:%S".
        folder_path (str): Directory containing wrfout files.
        output_file (str): Output NetCDF file name.
    """
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time")
    if any((start_time.minute, start_time.second, end_time.minute, end_time.second)):
        raise ValueError("WRF start_time and end_time must fall on whole hours")

    def calculate_rh(t2, q2, psfc):
        """
        Calculate relative humidity from temperature, mixing ratio, and pressure.
        
        Args:
            t2 (np.ndarray): 2-meter temperature in Kelvin
            q2 (np.ndarray): Water vapor mixing ratio (kg/kg)
            psfc (np.ndarray): Surface pressure in Pascals
        
        Returns:
            np.ndarray: Relative humidity in percent [0-100]
        """
        # Compute saturation vapor pressure (in hPa) using temperature converted to Celsius.
        e_s = 6.112 * np.exp((17.67 * (t2 - 273.15)) / ((t2 - 273.15) + 243.5))
        e_s = e_s * 100  # convert hPa to Pa
        # Calculate actual vapor pressure using mixing ratio.
        Rd = 287.05  # Gas constant for dry air (J/kg/K)
        Rv = 461.5   # Gas constant for water vapor (J/kg/K)
        eps = Rd / Rv
        e = q2 * psfc / (eps + q2)
        rh = (e / e_s) * 100
        return np.clip(rh, 0, 100)
    
    wrf_files_sorted, time_array = _select_wrf_files(
        folder_path, start_time, end_time
    )
    
    t2_list, wind_list, rh2_list, tsk_list = [], [], [], []
    swdown_list, psfc_list = [], []
    lat, lon = None, None
    
    for file in wrf_files_sorted:
        file_path = os.path.join(folder_path, file)
        with xr.open_dataset(file_path) as ds:
            # Extract variables.
            t2 = ds['T2'].values           # 2-meter temperature (K)
            q2 = ds['Q2'].values           # Mixing ratio at 2 m (kg/kg)
            psfc = ds['PSFC'].values       # Surface pressure (Pa)
            
            t2_list.append(t2)
            tsk_list.append(ds['TSK'].values)       # Land surface temperature (K)
            swdown_list.append(ds['SWDOWN'].values)    # Downwelling shortwave radiation (W/m^2)
            psfc_list.append(psfc)
            
            # Calculate wind speed from U10 and V10 components at 10 m.
            u10 = ds['U10'].values
            v10 = ds['V10'].values
            wind_speed = np.sqrt(u10**2 + v10**2)
            wind_list.append(wind_speed)
            
            # Calculate relative humidity using the helper function.
            rh2 = calculate_rh(t2, q2, psfc)
            rh2_list.append(rh2)
            
            # Extract latitude and longitude (assumed same for all files).
            if lat is None or lon is None:
                lat = ds['XLAT'].values[0, :, :]
                lon = ds['XLONG'].values[0, :, :]
    
    t2_array      = np.concatenate(t2_list, axis=0)
    wind_array    = np.concatenate(wind_list, axis=0)
    rh2_array     = np.concatenate(rh2_list, axis=0)
    tsk_array     = np.concatenate(tsk_list, axis=0)
    swdown_array  = np.concatenate(swdown_list, axis=0)
    psfc_array    = np.concatenate(psfc_list, axis=0)

    forcing_arrays = {
        "T2": t2_array,
        "WIND": wind_array,
        "RH2": rh2_array,
        "TSK": tsk_array,
        "SWDOWN": swdown_array,
        "PSFC": psfc_array,
    }
    expected_shape = (len(time_array), *lat.shape)
    for name, values in forcing_arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"WRF variable {name} has shape {values.shape}; expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"WRF variable {name} contains non-finite values")
    
    # Create a new NetCDF file and write the combined data.
    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', len(time_array))
        nc.createDimension('lat', lat.shape[0])
        nc.createDimension('lon', lon.shape[1])
        
        time_var = nc.createVariable('time', 'f8', ('time',))
        lat_var = nc.createVariable('lat', 'f4', ('lat', 'lon'))
        lon_var = nc.createVariable('lon', 'f4', ('lat', 'lon'))
        
        t2_var    = nc.createVariable('T2', 'f4', ('time', 'lat', 'lon'), zlib=True)
        wind_var  = nc.createVariable('WIND', 'f4', ('time', 'lat', 'lon'), zlib=True)
        rh2_var   = nc.createVariable('RH2', 'f4', ('time', 'lat', 'lon'), zlib=True)
        tsk_var   = nc.createVariable('TSK', 'f4', ('time', 'lat', 'lon'), zlib=True)
        swdown_var= nc.createVariable('SWDOWN', 'f4', ('time', 'lat', 'lon'), zlib=True)
     #   glw_var   = nc.createVariable('GLW', 'f4', ('time', 'lat', 'lon'), zlib=True)
        psfc_var  = nc.createVariable('PSFC', 'f4', ('time', 'lat', 'lon'), zlib=True)
        
        time_var.units = "hours since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        
        t2_var.units = "K"
        wind_var.units = "m/s"
        rh2_var.units = "%"
        tsk_var.units = "K"
        swdown_var.units = "W/m^2"
       # glw_var.units = "W/m^2"
        psfc_var.units = "Pa"
        
        time_var[:] = date2num(time_array, units=time_var.units, calendar=time_var.calendar)
        lat_var[:, :] = lat
        lon_var[:, :] = lon
        
        t2_var[:, :, :]    = t2_array
        wind_var[:, :, :]  = wind_array
        rh2_var[:, :, :]   = rh2_array
        tsk_var[:, :, :]   = tsk_array
        swdown_var[:, :, :] = swdown_array
        #glw_var[:, :, :]    = glw_array
        psfc_var[:, :, :]   = psfc_array
    
    print(f"New NetCDF file created: {output_file}")
    
# =============================================================================
# Functions to process the NetCDF file and create metfiles based on a set of raster tiles.
# =============================================================================
def _haversine_m(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1 (float): Latitude of first point (degrees)
        lon1 (float): Longitude of first point (degrees)
        lat2 (float): Latitude of second point (degrees)
        lon2 (float): Longitude of second point (degrees)
    
    Returns:
        float: Distance in meters
    """
    # distance in meters
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _local_cell_size_m(lon2d, lat2d, cx, cy, tree):
    """
    Estimate local grid cell size in meters at a given coordinate.
    
    Args:
        lon2d (np.ndarray): 2D array of longitudes
        lat2d (np.ndarray): 2D array of latitudes
        cx (float): Center longitude
        cy (float): Center latitude
        tree (cKDTree): KDTree for nearest neighbor lookup
    
    Returns:
        tuple: (cell_width_m, cell_height_m) in meters
    """
    ny, nx = lat2d.shape
    _, idx = tree.query([cx, cy], k=1)
    i, j = np.unravel_index(idx, (ny, nx))
    ew, ns = [], []
    def dist(i1, j1, i2, j2):
        """Calculate haversine distance between two grid cells."""
        return _haversine_m(lat2d[i1,j1], lon2d[i1,j1], lat2d[i2,j2], lon2d[i2,j2])
    if j-1 >= 0: ew.append(dist(i,j, i, j-1))
    if j+1 < nx: ew.append(dist(i,j, i, j+1))
    if i-1 >= 0: ns.append(dist(i,j, i-1, j))
    if i+1 < ny: ns.append(dist(i,j, i+1, j))
    anyd = ew + ns
    if not anyd:
        return 1e30, 1e30
    cell_w = np.median(ew) if ew else np.median(anyd)
    cell_h = np.median(ns) if ns else np.median(anyd)
    return cell_w, cell_h

def _tile_size_m(poly):
    """
    Calculate tile size in meters from polygon bounds.
    
    Args:
        poly (Polygon): Shapely polygon representing tile extent
    
    Returns:
        tuple: (width_m, height_m) tile dimensions in meters
    """
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx+maxx)/2.0, (miny+maxy)/2.0
    w = _haversine_m(cy, minx, cy, maxx) 
    h = _haversine_m(miny, cx, maxy, cx) 
    return w, h

def process_metfiles(netcdf_file, raster_folder, base_path, selected_date_str, preprocess_dir):
    """Sample gridded forcing over each raster tile and write UMEP met files."""
    metfiles_folder = os.path.join(preprocess_dir, "metfiles")
    os.makedirs(metfiles_folder, exist_ok=True)
    
    tf = TimezoneFinder()
    dataset = nc.Dataset(netcdf_file, "r")
    
    tif_files = glob.glob(os.path.join(raster_folder, "*.tif"))
    if not tif_files:
        dataset.close()
        raise FileNotFoundError(f"no GeoTIFF tiles found in {raster_folder}")

    var_map = {
        "Wind": "WIND",     
        "RH": "RH2",
        "Td": "T2",         # K -> °C
        "press": "PSFC",    # Pa -> kPa
        "Kdn": "SWDOWN",       
    }
    fixed_values = {
        "Q*": -999, "QH": -999, "QE": -999, "Qs": -999, "Qf": -999,
        "snow": -999, "ldown": -999, "fcld": -999, "wuh": -999, "xsmd": -999, "lai_hr": -999,
        "Kdiff": -999, "Kdir": -999, "Wd": -999,
        "rain": 0
    }

    required_variables = {"time", "lat", "lon", *var_map.values()}
    missing_variables = sorted(required_variables.difference(dataset.variables))
    if missing_variables:
        dataset.close()
        raise ValueError(
            f"forcing file {netcdf_file} is missing variables: {', '.join(missing_variables)}"
        )
    
    time_var = dataset.variables["time"][:]
    time_units = dataset.variables["time"].units
    time_base_date = nc.num2date(time_var, units=time_units, only_use_cftime_datetimes=False)
    selected_local_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()

    lat2d = np.array(dataset.variables["lat"][:], dtype=float)  
    lon2d = np.array(dataset.variables["lon"][:], dtype=float)  
    if lat2d.ndim != 2 or lon2d.shape != lat2d.shape or min(lat2d.shape) < 1:
        dataset.close()
        raise ValueError("forcing latitude and longitude must be matching two-dimensional grids")
    if not np.isfinite(lat2d).all() or not np.isfinite(lon2d).all():
        dataset.close()
        raise ValueError("forcing latitude and longitude contain non-finite values")
    ny, nx = lat2d.shape
    pts_flat = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    tree = cKDTree(pts_flat)
    columns = [
        'iy', 'id', 'it', 'imin',
        'Q*', 'QH', 'QE', 'Qs', 'Qf',
        'Wind', 'RH', 'Td', 'press',
        'Kdn','rain', 'snow', 'ldown',
        'fcld', 'wuh', 'xsmd', 'lai_hr',
        'Kdiff', 'Kdir', 'Wd'
    ]
    columns_out = [
        "iy", "id", "it", "imin",
        "Q*", "QH", "QE", "Qs", "Qf",
        "Wind", "RH", "Td", "press",
        "rain",
        "Kdn",
        "snow",
        "ldown",
        "fcld",
        "wuh",
        "xsmd",
        "lai_hr",
        "Kdiff",
        "Kdir",
        "Wd"
    ]
    
    for tif_file in tif_files:
        ds_tif = gdal.Open(tif_file)
        if ds_tif is None:
            dataset.close()
            raise FileNotFoundError(f"could not open {tif_file}")

        gt_tif = ds_tif.GetGeoTransform()
        xsize = ds_tif.RasterXSize
        ysize = ds_tif.RasterYSize

        proj_tif = ds_tif.GetProjection()
        if not proj_tif:
            ds_tif = None
            dataset.close()
            raise ValueError(f"raster has no CRS: {tif_file}")
        srs_tif = osr.SpatialReference()
        if srs_tif.ImportFromWkt(proj_tif) != 0:
            ds_tif = None
            dataset.close()
            raise ValueError(f"could not read raster CRS: {tif_file}")
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)
        srs_tif.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(srs_tif, target_srs)

        corners = [
            gdal.ApplyGeoTransform(gt_tif, pixel_x, pixel_y)
            for pixel_x, pixel_y in ((0, 0), (xsize, 0), (xsize, ysize), (0, ysize))
        ]
        try:
            lonlat_corners = [transform.TransformPoint(x, y) for x, y in corners]
            lons = [pt[0] for pt in lonlat_corners]
            lats = [pt[1] for pt in lonlat_corners]
        except Exception as exc:
            ds_tif = None
            dataset.close()
            raise RuntimeError(f"could not transform {tif_file} to EPSG:4326") from exc
        if not np.isfinite(lons).all() or not np.isfinite(lats).all():
            ds_tif = None
            dataset.close()
            raise ValueError(f"transformed raster bounds are non-finite: {tif_file}")

        min_lon_tif, max_lon_tif = min(lons), max(lons)
        min_lat_tif, max_lat_tif = min(lats), max(lats)
        shape = box(min_lon_tif, min_lat_tif, max_lon_tif, max_lat_tif)
        shape = Polygon([(x, y) for (x, y) in shape.exterior.coords])  # explicitly polygon

        shape_name = os.path.splitext(os.path.basename(tif_file))[0]
        shape_name_clean = re.sub(r'\W+', '_', shape_name).replace("DEM", "metfile", 1)
        output_text_file = os.path.join(metfiles_folder, f"{shape_name_clean}_{selected_date_str}.txt")
        
        lat_center, lon_center = shape.centroid.y, shape.centroid.x
        timezone_name = tf.timezone_at(lng=lon_center, lat=lat_center) or "UTC"
        local_tz = pytz.timezone(timezone_name)
        
        local_start = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(0, 0)))
        local_end = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(23, 59)))
        utc_start = local_start.astimezone(pytz.utc)
        utc_end = local_end.astimezone(pytz.utc)
        
        time_indices = [
            idx for idx, dt in enumerate(time_base_date)
            if utc_start <= dt.replace(tzinfo=pytz.utc) <= utc_end
        ]
        if not time_indices:
            ds_tif = None
            dataset.close()
            raise ValueError(
                f"forcing file has no data for local date {selected_date_str} at {tif_file}"
            )
        print(f"Processing {len(time_indices)} time steps for {shape_name_clean}")

        tile_w_m, tile_h_m = _tile_size_m(shape)
        cell_w_m, cell_h_m = _local_cell_size_m(lon2d, lat2d, lon_center, lat_center, tree)
        use_nn = (cell_w_m > tile_w_m) and (cell_h_m > tile_h_m)

        inside_mask = None
        if not use_nn:
            path = Path(np.asarray(shape.exterior.coords)[:, :2])
            inside_mask = path.contains_points(np.column_stack([lon2d.ravel(), lat2d.ravel()])).reshape(lat2d.shape)
            if not np.any(inside_mask):
                use_nn = True
                
        met_new = []
        for t in time_indices:
            utc_time = time_base_date[t].replace(tzinfo=pytz.utc)
            local_time = utc_time.astimezone(local_tz)
            year = local_time.year
            doy = local_time.timetuple().tm_yday
            hour = local_time.hour
            minute = local_time.minute

            row = [year, doy, hour, minute]
            row.extend([fixed_values[key] for key in ["Q*", "QH", "QE", "Qs", "Qf"]])
            
            for key in ["Wind", "RH", "Td", "press", "Kdn"]:
                var_name = var_map[key]
                try:
                    data_array = dataset.variables[var_name][t, :, :]
                    data_array = np.asanyarray(data_array)
                    if data_array.shape != (ny, nx):
                        raise ValueError(
                            f"{var_name} grid {data_array.shape} does not match coordinates {(ny, nx)}"
                        )
                    if np.ma.isMaskedArray(data_array):
                        data_array = np.where(data_array.mask, np.nan, data_array.data)

                    if use_nn:
                        _, idx_nn = tree.query([lon_center, lat_center], k=1)
                        ii, jj = np.unravel_index(idx_nn, (ny, nx))
                        mean_value = float(data_array[ii, jj])
                    else:
                        masked_data = np.where(inside_mask, data_array, np.nan)
                        mean_value = float(np.nanmean(masked_data)) if np.any(~np.isnan(masked_data)) else np.nan
                        if not np.isfinite(mean_value):
                            _, idx_nn = tree.query([lon_center, lat_center], k=1)
                            ii, jj = np.unravel_index(idx_nn, (ny, nx))
                            mean_value = float(data_array[ii, jj])

                    if key == "Td":
                        mean_value -= 273.15
                    if key == "press":
                        mean_value /= 1000.0
                    if not np.isfinite(mean_value):
                        raise ValueError(f"sampled {var_name} is not finite")
                    row.append(mean_value)

                except Exception as exc:
                    ds_tif = None
                    dataset.close()
                    raise RuntimeError(
                        f"could not sample {var_name} at forcing time index {t} for {tif_file}"
                    ) from exc
            
            row.append(fixed_values["rain"])
            row.extend([fixed_values[key] for key in ["snow", "ldown", "fcld", "wuh", "xsmd", "lai_hr", "Kdiff", "Kdir", "Wd"]])
            met_new.append(row)

        df = pd.DataFrame(met_new, columns=columns)
        df = df[columns_out]
        with open(output_text_file, "w") as f:
            f.write(" ".join(df.columns) + "\n")
            for _, row in df.iterrows():
                f.write('{:d} {:d} {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.5f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {: .2f} {: .2f}\n'.format(
                    int(row["iy"]), int(row["id"]), int(row["it"]), int(row["imin"]),
                    row["Q*"], row["QH"], row["QE"], row["Qs"], row["Qf"],
                    row["Wind"], row["RH"], row["Td"], row["press"], row["rain"],
                    row["Kdn"], row["snow"], row["ldown"], row["fcld"], row["wuh"],
                    row["xsmd"], row["lai_hr"], row["Kdiff"], row["Kdir"], row["Wd"]
                ))
        print(f"Metfile saved: {output_text_file}")
        ds_tif = None

    dataset.close()
    print(f"All raster extents processed and metfiles saved in {metfiles_folder}")
    
# =============================================================================
# Function to process own met file: copies the source met file into new files
# renaming each copy based on the numeric suffix extracted from .tif files.
# =============================================================================
def create_met_files(base_path, source_met_file, preprocess_dir):
    """
    Copy a given met file to multiple outputs based on the raster tile filenames.

    Parameters:
        base_path (str): Base directory containing input rasters.
        source_met_file (str): Path to user-provided met file.
        preprocess_dir (str): Directory for preprocessing outputs (pre_processing_outputs).
    """
    raster_folder = os.path.join(preprocess_dir, 'Building_DSM')
    target_folder = os.path.join(preprocess_dir, 'metfiles')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    else:
        shutil.rmtree(target_folder)
        os.makedirs(target_folder)
    
    for file in os.listdir(raster_folder):
        if file.lower().endswith('.tif'):
            name_without_ext = os.path.splitext(file)[0]
            prefix = 'Building_DSM_'
            if name_without_ext.startswith(prefix):
                digits = name_without_ext[len(prefix):]
                new_filename = f'metfile_{digits}.txt'
                target_met_file = os.path.join(target_folder, new_filename)
                shutil.copy(source_met_file, target_met_file)
                print(f"Copied to {target_met_file}")

# =============================================================================
# Main function: checks rasters, creates tiles, and creates metfiles using either a
# user-supplied met file or a netCDF file. Only the parameters required for the chosen
# method need to be provided.
# =============================================================================
def ppr(base_path, building_dsm_filename, dem_filename, trees_filename, landcover_filename,
         tile_size, overlap, selected_date_str, use_own_met,start_time=None, end_time=None, data_source_type=None, data_folder=None,
         own_met_file=None, preprocess_dir=None):
    """
    Preprocessing routine to validate raster files, generate tiles, and prepare metfiles for SOLWEIG.

    Parameters:
        base_path (str): Base working directory containing input rasters.
        building_dsm_filename (str): Filename of building DSM raster.
        dem_filename (str): Filename of DEM raster.
        trees_filename (str): Filename of trees raster.
        landcover_filename (str): Filename of landcover raster or None.
        tile_size (int): Tile size in pixels.
        overlap (int): Overlap between tiles in pixels.
        selected_date_str (str): Selected date (YYYY-MM-DD).
        use_own_met (bool): Whether to use a user-provided met file.
        start_time (str): Start datetime (required if not using own met file).
        end_time (str): End datetime (required if not using own met file).
        data_source_type (str): Either 'ERA5' or 'wrfout'.
        data_folder (str): Folder containing input NetCDF files.
        own_met_file (str): Path to user-provided met file (used if use_own_met is True).
        preprocess_dir (str): Directory for preprocessing outputs (pre_processing_outputs folder).
    """
    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)
             
    building_dsm_path = os.path.join(base_path, building_dsm_filename)
    dem_path = os.path.join(base_path, dem_filename)
    trees_path = os.path.join(base_path, trees_filename)
    if landcover_filename is not None:
        landcover_path = os.path.join(base_path, landcover_filename)

    # Check that all rasters have matching dimensions, transform, and CRS.
    if landcover_filename is not None:
        check_rasters([building_dsm_path, dem_path, trees_path, landcover_path])
    else:
        check_rasters([building_dsm_path, dem_path, trees_path])

    if landcover_filename is not None:
        rasters = {
            "Building_DSM": building_dsm_path,
            "DEM": dem_path,
            "Trees": trees_path,
            "Landcover": landcover_path
        }
    else: 
        rasters = {
            "Building_DSM": building_dsm_path,
            "DEM": dem_path,
            "Trees": trees_path   
        }  
        
    for tile_type, raster in rasters.items():
        print(f"Creating tiles for {tile_type}...")
        create_tiles(raster, tile_size, overlap, tile_type, preprocess_dir)
    
    # For metfiles processing, we use the DEM tiles folder.
    dem_tiles_folder = os.path.join(preprocess_dir, "DEM")
    
    # Choose between own met file or processed NetCDF file.
    if use_own_met:
        if own_met_file is None:
            raise ValueError("own_met_file is required when use_own_met=True")
        create_met_files(base_path, own_met_file, preprocess_dir)
    else:
        # Ensure all additional required parameters are provided.
        if data_folder is None or data_source_type is None or start_time is None or end_time is None:
            raise ValueError(
                "data_folder, data_source_type, start_time and end_time are "
                "required when use_own_met=False"
            )
            
        # Define the name (and path) for the processed NetCDF output.
        processed_nc_file = os.path.join(preprocess_dir, "Outfile.nc")
        
        if data_source_type.lower() == "era5":
            process_era5_data(start_time, end_time, data_folder, output_file=processed_nc_file)
        elif data_source_type.lower() == "wrfout":
            process_wrfout_data(start_time, end_time, data_folder, output_file=processed_nc_file)
        else:
            raise ValueError("data_source_type must be either 'ERA5' or 'wrfout'")
        
        # Process the generated NetCDF file to create metfiles.
        process_metfiles(processed_nc_file, dem_tiles_folder, base_path, selected_date_str, preprocess_dir)








