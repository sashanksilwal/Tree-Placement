#!/usr/bin/env python3
"""
download_era5_cities.py — Download ERA5 reanalysis data for Nature Cities experiment.

For each city in cities.json:
  1. Determine local timezone from coordinates (TimezoneFinder)
  2. Download ERA5 single-level hourly data for a peak summer day + 1 buffer day
  3. Convert to UMEP-format met file with correct local time

ERA5 variables: T2m, D2m, SP, U10, V10, SSRD, STRD, TP
Output: <output-dir>/<city_id>/met.txt

IMPORTANT: SOLWEIG expects met file hours in LOCAL TIME.
The solar position code does: UT = hour - UTC_offset, so met file hours
must be in the city's local timezone.

Usage:
  python download_era5_cities.py                          # all cities
  python download_era5_cities.py --city 42                # single city
  python download_era5_cities.py --skip-download          # convert existing NetCDF only
  python download_era5_cities.py --date 2023-07-20        # override simulation date
  python download_era5_cities.py --validate-only          # check existing met files
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.environ.get("UTHERM_DATA_DIR", "data/300_cities_data")
DEFAULT_SIM_DATE = "2023-07-15"  # peak-summer clear-sky day
DEFAULT_MONTH = "2023-07"

# ERA5 variables
INSTANT_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
ACCUM_VARS = [
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
]
ALL_VARS = INSTANT_VARS + ACCUM_VARS

# UMEP met file header (24 columns)
MET_HEADER = (
    "iy id it imin Q* QH QE Qs Qf Wind RH Td press rain "
    "Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd"
)


def met_matches_date(met_path, sim_date_str):
    """Return whether an existing UMEP met file belongs to the requested day."""
    try:
        target = datetime.strptime(sim_date_str, "%Y-%m-%d").date()
        data = np.loadtxt(met_path, skiprows=1, ndmin=2)
    except (OSError, TypeError, ValueError):
        return False
    if data.shape[0] == 0 or data.shape[1] < 23:
        return False
    expected_doy = target.timetuple().tm_yday
    return bool(
        np.isfinite(data[:, :4]).all()
        and np.equal(data[:, :4], np.floor(data[:, :4])).all()
        and ((data[:, 0] == target.year) &
             (data[:, 1] == expected_doy)).all()
    )


# -- Solar geometry -----------------------------------------------------------


def solar_declination(doy):
    """Solar declination angle in radians."""
    if not isinstance(doy, (int, np.integer)) or isinstance(doy, (bool, np.bool_)) or not 1 <= doy <= 366:
        raise ValueError("doy must be an integer between 1 and 366")
    return 23.45 * math.pi / 180 * math.sin(2 * math.pi * (284 + doy) / 365)


def hour_angle(solar_hour):
    """Hour angle in radians. solar_hour in decimal hours (12 = solar noon)."""
    if not math.isfinite(float(solar_hour)):
        raise ValueError("solar_hour must be finite")
    return (solar_hour - 12.0) * 15.0 * math.pi / 180.0


def solar_zenith(lat_deg, lon_deg, doy, utc_hour):
    """Solar zenith angle in radians with equation of time correction."""
    if not math.isfinite(float(lat_deg)) or not -90.0 <= lat_deg <= 90.0:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not math.isfinite(float(lon_deg)) or not -180.0 <= lon_deg <= 180.0:
        raise ValueError("longitude must be between -180 and 180 degrees")
    if not math.isfinite(float(utc_hour)):
        raise ValueError("utc_hour must be finite")
    lat = math.radians(lat_deg)
    decl = solar_declination(doy)

    B = 2 * math.pi * (doy - 1) / 365.0
    eot = 229.18 * (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089 * math.sin(2 * B)
    )

    solar_time = utc_hour + lon_deg / 15.0 + eot / 60.0
    ha = hour_angle(solar_time)

    cos_zen = (math.sin(lat) * math.sin(decl) +
               math.cos(lat) * math.cos(decl) * math.cos(ha))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    return math.acos(cos_zen)


def extraterrestrial_radiation(doy):
    """Extraterrestrial solar radiation (W/m2) on horizontal surface."""
    if not isinstance(doy, (int, np.integer)) or isinstance(doy, (bool, np.bool_)) or not 1 <= doy <= 366:
        raise ValueError("doy must be an integer between 1 and 366")
    return 1361.0 * (1 + 0.033 * math.cos(2 * math.pi * doy / 365.0))


# -- Erbs decomposition -------------------------------------------------------


def erbs_decomposition(ghi, lat_deg, lon_deg, doy, utc_hour):
    """Split GHI into DHI and DNI using Erbs et al. (1982) model.

    Returns (dhi, dni) in W/m2.
    """
    if not math.isfinite(float(ghi)) or ghi < 0.0:
        raise ValueError("ghi must be finite and nonnegative")
    zen = solar_zenith(lat_deg, lon_deg, doy, utc_hour)
    cos_zen = math.cos(zen)

    if cos_zen <= 0.0 or ghi <= 0.0:
        return 0.0, 0.0

    I_ext = extraterrestrial_radiation(doy)
    kt = min(ghi / (I_ext * cos_zen), 1.0)

    if kt <= 0.22:
        kd = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        kd = (0.9511 - 0.1604 * kt + 4.388 * kt**2
              - 16.638 * kt**3 + 12.336 * kt**4)
    else:
        kd = 0.165

    kd = min(max(kd, 0.0), 1.0)
    dhi = kd * ghi
    dni = min(max((ghi - dhi) / cos_zen, 0.0), 1361.0)
    return dhi, dni


# -- Unit conversions ----------------------------------------------------------


def relative_humidity(t_celsius, td_celsius):
    """RH (%) from air temperature and dew point via Magnus formula."""
    if not math.isfinite(float(t_celsius)) or not math.isfinite(float(td_celsius)):
        raise ValueError("air and dew-point temperatures must be finite")
    if t_celsius <= -243.04 or td_celsius <= -243.04:
        raise ValueError("temperature is outside the Magnus formula domain")
    a, b = 17.625, 243.04
    gamma_t = (a * t_celsius) / (b + t_celsius)
    gamma_td = (a * td_celsius) / (b + td_celsius)
    rh = 100.0 * math.exp(gamma_td - gamma_t)
    return min(max(rh, 0.0), 100.0)


def wind_speed_direction(u, v):
    """Wind speed (m/s) and meteorological direction (deg) from u,v components."""
    if not math.isfinite(float(u)) or not math.isfinite(float(v)):
        raise ValueError("wind components must be finite")
    speed = math.sqrt(u**2 + v**2)
    direction = math.degrees(math.atan2(-u, -v)) % 360.0
    return speed, direction


# -- Timezone ------------------------------------------------------------------


def get_timezone(lon, lat):
    """Get IANA timezone string for a lon/lat coordinate."""
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    tz_str = tf.timezone_at(lng=lon, lat=lat)
    if tz_str is None:
        offset_hours = round(lon / 15)
        tz_str = f"Etc/GMT{-offset_hours:+d}" if offset_hours != 0 else "Etc/GMT"
        log.warning(f"  TimezoneFinder returned None for ({lon}, {lat}), using {tz_str}")
    return tz_str


# -- ERA5 download -------------------------------------------------------------


def download_era5_for_cities(cities, output_dir, sim_date_str, skip_download=False):
    """Download ERA5 data covering all cities for the simulation date.

    Downloads a single ERA5 request covering the bounding box of all cities.
    Needs sim_date + 1 buffer day for UTC -> local time conversion.
    """
    if skip_download:
        log.info("Skipping ERA5 download (--skip-download)")
        return

    import cdsapi

    target_date = datetime.strptime(sim_date_str, "%Y-%m-%d").date()
    buffer_date = target_date + timedelta(days=1)
    days_needed = sorted(set([
        target_date.strftime("%Y-%m-%d"),
        buffer_date.strftime("%Y-%m-%d"),
    ]))

    by_month = defaultdict(set)
    for d in days_needed:
        ym = d[:7]
        by_month[ym].add(d.split("-")[2])

    # Bounding box covering all cities
    lons = [c["lon"] for c in cities]
    lats = [c["lat"] for c in cities]
    margin = 0.5
    area = [max(lats) + margin, min(lons) - margin,
            min(lats) - margin, max(lons) + margin]

    raw_dir = output_dir / "era5_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for ym, day_set in sorted(by_month.items()):
        yr, mo = ym.split("-")
        days = sorted(day_set)

        out_file = raw_dir / f"era5_{ym}.nc"
        has_parts = list(raw_dir.glob(f"era5_{ym}_part*.nc"))
        if (out_file.exists() and out_file.stat().st_size > 0) or has_parts:
            log.info(f"  {out_file.name}: already exists, skipping")
            continue

        log.info(f"  Requesting ERA5 {ym} (days: {days}), area: {area}")
        request = {
            "product_type": ["reanalysis"],
            "variable": ALL_VARS,
            "year": [yr],
            "month": [mo],
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "area": area,
        }

        zip_file = raw_dir / f"era5_{ym}.zip"
        client.retrieve("reanalysis-era5-single-levels", request, str(zip_file))

        import zipfile
        if zipfile.is_zipfile(str(zip_file)):
            with zipfile.ZipFile(str(zip_file), "r") as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if len(nc_names) == 1:
                    zf.extract(nc_names[0], str(raw_dir))
                    (raw_dir / nc_names[0]).rename(out_file)
                else:
                    for i, nc_name in enumerate(sorted(nc_names)):
                        zf.extract(nc_name, str(raw_dir))
                        (raw_dir / nc_name).rename(raw_dir / f"era5_{ym}_part{i}.nc")
                    out_file.touch()
            zip_file.unlink()
        else:
            zip_file.rename(out_file)

        log.info(f"  Saved: {out_file}")


def download_era5_full_month(cities, output_dir, month_str, skip_download=False):
    """Download ERA5 data for a full month covering all cities.

    Downloads all days in the specified month + 1 buffer day (first day of next month)
    for UTC -> local time conversion at month end.

    Args:
        cities: list of city dicts with 'lon', 'lat'
        output_dir: Path to output directory
        month_str: month string 'YYYY-MM' (e.g. '2023-07')
        skip_download: if True, skip CDS API call
    """
    if skip_download:
        log.info("Skipping ERA5 download (--skip-download)")
        return

    import calendar
    import cdsapi

    yr, mo = month_str.split("-")
    yr_int, mo_int = int(yr), int(mo)
    n_days = calendar.monthrange(yr_int, mo_int)[1]
    all_days = [f"{d:02d}" for d in range(1, n_days + 1)]

    # Buffer: first day of next month
    if mo_int == 12:
        buffer_ym = f"{yr_int + 1}-01"
    else:
        buffer_ym = f"{yr_int}-{mo_int + 1:02d}"
    buffer_day = "01"

    # Bounding box covering all cities
    lons = [c["lon"] for c in cities]
    lats = [c["lat"] for c in cities]
    margin = 0.5
    area = [max(lats) + margin, min(lons) - margin,
            min(lats) - margin, max(lons) + margin]

    raw_dir = output_dir / "era5_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    # Download main month (all days)
    requests = [(month_str, yr, mo, all_days)]
    # Buffer month (just day 1)
    buf_yr, buf_mo = buffer_ym.split("-")
    requests.append((buffer_ym, buf_yr, buf_mo, [buffer_day]))

    for ym, r_yr, r_mo, days in requests:
        out_file = raw_dir / f"era5_{ym}.nc"
        has_parts = list(raw_dir.glob(f"era5_{ym}_part*.nc"))
        if (out_file.exists() and out_file.stat().st_size > 0) or has_parts:
            log.info(f"  {out_file.name}: already exists, skipping")
            continue

        log.info(f"  Requesting ERA5 {ym} ({len(days)} days), area: {area}")
        request = {
            "product_type": ["reanalysis"],
            "variable": ALL_VARS,
            "year": [r_yr],
            "month": [r_mo],
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "area": area,
        }

        zip_file = raw_dir / f"era5_{ym}.zip"
        client.retrieve("reanalysis-era5-single-levels", request, str(zip_file))

        import zipfile
        if zipfile.is_zipfile(str(zip_file)):
            with zipfile.ZipFile(str(zip_file), "r") as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if len(nc_names) == 1:
                    zf.extract(nc_names[0], str(raw_dir))
                    (raw_dir / nc_names[0]).rename(out_file)
                else:
                    for i, nc_name in enumerate(sorted(nc_names)):
                        zf.extract(nc_name, str(raw_dir))
                        (raw_dir / nc_name).rename(raw_dir / f"era5_{ym}_part{i}.nc")
                    out_file.touch()
            zip_file.unlink()
        else:
            zip_file.rename(out_file)

        log.info(f"  Saved: {out_file}")


def select_clear_sky_day(city, raw_dir, month_str="2023-07"):
    """Find the day in the month with highest total GHI for this city.

    GHI (SSRD) is the best clear-sky proxy -- maximized on cloud-free days.

    Args:
        city: dict with 'id', 'lon', 'lat'
        raw_dir: Path to era5_raw directory containing NetCDF files
        month_str: 'YYYY-MM' month to search

    Returns:
        (best_date_str, daily_ghi_max) or (None, 0) if no valid data.
        best_date_str is 'YYYY-MM-DD'.
    """
    times, lats_arr, lons_arr, var_map, close_fn = open_era5_nc(raw_dir, month_str)
    if times is None:
        log.error(f"  City {city['id']}: No ERA5 data for {month_str}")
        return None, 0

    lat_idx = find_nearest_idx(lats_arr, city["lat"])
    lon_idx = find_nearest_idx(lons_arr, city["lon"])

    # Extract SSRD for all timesteps
    ssrd_ds = var_map.get("ssrd")
    if ssrd_ds is None:
        close_fn()
        log.error(f"  City {city['id']}: SSRD variable not found")
        return None, 0

    ssrd_var = ssrd_ds.variables["ssrd"]
    if ssrd_var.ndim == 4:
        ssrd_all = ssrd_var[:, 0, lat_idx, lon_idx]
    elif ssrd_var.ndim == 3:
        ssrd_all = ssrd_var[:, lat_idx, lon_idx]
    else:
        close_fn()
        log.error(f"  City {city['id']}: Unexpected SSRD dims: {ssrd_var.ndim}")
        return None, 0

    ssrd_all = np.array(ssrd_all)
    # Convert from J/m2 accumulated to W/m2 instantaneous
    ghi_all = np.maximum(ssrd_all / 3600.0, 0.0)

    import pytz

    local_tz = pytz.timezone(get_timezone(city["lon"], city["lat"]))
    daily_ghi = defaultdict(float)
    daily_max_hourly = defaultdict(float)
    for t_idx, t in enumerate(times):
        utc_dt = datetime(t.year, t.month, t.day, t.hour, t.minute, tzinfo=pytz.utc)
        local_dt = utc_dt.astimezone(local_tz)
        day_str = local_dt.strftime("%Y-%m-%d")
        if not day_str.startswith(month_str):
            continue
        ghi_val = float(ghi_all[t_idx])
        daily_ghi[day_str] += ghi_val
        daily_max_hourly[day_str] = max(daily_max_hourly[day_str], ghi_val)

    close_fn()

    if not daily_ghi:
        log.error(f"  City {city['id']}: No daily GHI data for {month_str}")
        return None, 0

    # Filter: exclude days where max hourly GHI < 400 W/m2 (overcast)
    valid_days = {d: g for d, g in daily_ghi.items()
                  if daily_max_hourly[d] >= 400.0}

    if not valid_days:
        log.warning(f"  City {city['id']}: No days with max hourly GHI >= 400, "
                     "using best available")
        valid_days = daily_ghi

    best_day = max(valid_days, key=valid_days.get)
    best_ghi = valid_days[best_day]

    log.info(f"  City {city['id']} ({city.get('name', '')}): "
             f"clear-sky day = {best_day}, daily GHI = {best_ghi:.0f} W/m2")

    return best_day, float(best_ghi)


# -- NetCDF reading ------------------------------------------------------------


def find_nearest_idx(arr, val):
    """Index of nearest value in array."""
    return int(np.argmin(np.abs(np.array(arr) - val)))


def open_era5_nc(raw_dir, ym):
    """Open ERA5 NetCDF files for a year-month.

    Returns (times, lats, lons, var_map, close_func) or all None if missing.
    """
    import netCDF4 as nc

    nc_files = []
    single = raw_dir / f"era5_{ym}.nc"
    if single.exists() and single.stat().st_size > 0:
        nc_files.append(single)
    for p in sorted(raw_dir.glob(f"era5_{ym}_part*.nc")):
        nc_files.append(p)

    if not nc_files:
        return None, None, None, None, None

    datasets = [nc.Dataset(str(f)) for f in nc_files]

    var_map = {}
    for ds in datasets:
        for vname in ds.variables:
            if vname not in ("number", "valid_time", "latitude", "longitude",
                             "expver", "time"):
                var_map[vname] = ds

    ref_ds = datasets[0]
    time_var = ref_ds.variables["valid_time"]
    times = nc.num2date(time_var[:], time_var.units)
    lats = np.array(ref_ds.variables["latitude"][:])
    lons = np.array(ref_ds.variables["longitude"][:])

    def close_all():
        for ds in datasets:
            ds.close()

    return times, lats, lons, var_map, close_all


# -- Convert to UMEP met format -----------------------------------------------


def convert_era5_to_met(city, output_dir, sim_date_str):
    """Convert ERA5 NetCDF to UMEP-format met.txt for one city.

    Extracts the nearest grid point and converts UTC -> local time.
    """
    import pytz

    cid = city["id"]
    lon, lat = city["lon"], city["lat"]
    city_dir = output_dir / str(cid)
    met_path = city_dir / "met.txt"

    if met_path.exists() and met_matches_date(met_path, sim_date_str):
        log.info(f"  City {cid}: met.txt already matches {sim_date_str}, skipping")
        return met_path
    if met_path.exists():
        log.info(f"  City {cid}: replacing met.txt for requested date {sim_date_str}")

    tz_str = get_timezone(lon, lat)
    local_tz = pytz.timezone(tz_str)
    target_date = datetime.strptime(sim_date_str, "%Y-%m-%d").date()
    target_date_str = target_date.strftime("%Y-%m-%d")

    raw_dir = output_dir / "era5_raw"
    all_rows = []

    # Process main month and buffer month (if different)
    buffer_date = target_date + timedelta(days=1)
    yms_to_check = sorted(set([sim_date_str[:7], buffer_date.strftime("%Y-%m")]))

    for ym in yms_to_check:
        times, lats_arr, lons_arr, var_map, close_fn = open_era5_nc(raw_dir, ym)
        if times is None:
            continue

        lat_idx = find_nearest_idx(lats_arr, lat)
        lon_idx = find_nearest_idx(lons_arr, lon)

        def get_val(varname, t_idx):
            ds = var_map[varname]
            var = ds.variables[varname]
            if var.ndim == 4:
                return float(var[t_idx, 0, lat_idx, lon_idx])
            elif var.ndim == 3:
                return float(var[t_idx, lat_idx, lon_idx])
            raise ValueError(f"Unexpected dims for {varname}: {var.ndim}")

        for t_idx, t in enumerate(times):
            utc_dt = datetime(t.year, t.month, t.day, t.hour, t.minute,
                              tzinfo=pytz.utc)
            local_dt = utc_dt.astimezone(local_tz)

            if local_dt.strftime("%Y-%m-%d") != target_date_str:
                continue

            local_hour = local_dt.hour
            local_minute = local_dt.minute
            local_year = local_dt.year
            local_doy = (local_dt.date() - date(local_year, 1, 1)).days + 1

            try:
                t2m = get_val("t2m", t_idx)
                d2m = get_val("d2m", t_idx)
                sp = get_val("sp", t_idx)
                u10 = get_val("u10", t_idx)
                v10 = get_val("v10", t_idx)
                ssrd = get_val("ssrd", t_idx)
                strd = get_val("strd", t_idx)
                try:
                    tp = get_val("tp", t_idx)
                except KeyError:
                    tp = 0.0
            except (KeyError, IndexError) as e:
                log.warning(f"    Skipping timestep {t}: {e}")
                continue

            ta = t2m - 273.15
            td = d2m - 273.15
            rh = relative_humidity(ta, td)
            pressure = sp / 1000.0
            wind, wind_dir = wind_speed_direction(u10, v10)
            ghi = max(ssrd / 3600.0, 0.0)
            ldown = strd / 3600.0
            rain = max(tp * 1000.0, 0.0)
            dhi, dni = erbs_decomposition(ghi, lat, lon, local_doy, utc_dt.hour)

            row = [
                local_year, local_doy, local_hour, local_minute,
                -999, -999, -999, -999, -999,
                wind, rh, ta, pressure, rain,
                ghi, -999, ldown,
                -999, -999, -999, -999,
                dhi, dni, wind_dir,
            ]
            all_rows.append((local_doy, local_hour, local_minute, row))

        close_fn()

    if not all_rows:
        log.error(f"  City {cid}: No ERA5 data for {sim_date_str}")
        return None

    # Sort and deduplicate
    all_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    seen = set()
    unique_rows = []
    for doy_val, hour, minute, row in all_rows:
        key = (doy_val, hour, minute)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    city_dir.mkdir(parents=True, exist_ok=True)
    with open(met_path, "w") as f:
        f.write(MET_HEADER + "\n")
        for row in unique_rows:
            f.write(
                " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
                + "\n"
            )

    log.info(f"  City {cid} ({city['name']}): {len(unique_rows)}h, tz={tz_str}")
    return met_path


# -- Validation ----------------------------------------------------------------


def validate_met(met_path):
    """Validate a met file: check ranges and completeness."""
    try:
        data = np.loadtxt(met_path, skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        log.warning(f"    WARNING: cannot read met file: {exc}")
        return False
    if data.shape[0] == 0 or data.shape[1] < 23:
        log.warning("    WARNING: met file must have at least one row and 23 columns")
        return False
    n_hours = data.shape[0]
    ta = data[:, 11]
    rh = data[:, 10]
    wind = data[:, 9]
    ghi = data[:, 14]

    log.info(f"    {n_hours}h | Ta [{ta.min():.1f}, {ta.max():.1f}]C | "
             f"RH [{rh.min():.0f}, {rh.max():.0f}]% | "
             f"Wind [{wind.min():.1f}, {wind.max():.1f}]m/s | GHI max={ghi.max():.0f}")

    issues = []
    if not np.isfinite(data[:, (0, 1, 2, 3, 9, 10, 11, 12, 14, 21, 22)]).all():
        issues.append("non-finite values in required columns")
    if ta.max() > 55 or ta.min() < -50:
        issues.append(f"Ta out of range [{ta.min():.1f}, {ta.max():.1f}]")
    if rh.max() > 101 or rh.min() < 0:
        issues.append(f"RH out of range [{rh.min():.1f}, {rh.max():.1f}]")
    if n_hours < 20:
        issues.append(f"Only {n_hours} hours (expected ~24)")
    if ghi.max() > 1400:
        issues.append(f"GHI suspiciously high: {ghi.max():.0f}")
    if wind.min() < 0:
        issues.append(f"negative wind speed: {wind.min():.1f}")
    if data[:, 12].min() < 80 or data[:, 12].max() > 110:
        issues.append("pressure outside 80-110 kPa")

    for issue in issues:
        log.warning(f"    WARNING: {issue}")
    return len(issues) == 0


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 and convert to UMEP met format for Nature Cities"
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cities-json", type=Path, default=None,
                        help="Path to cities.json (default: <output-dir>/cities.json)")
    parser.add_argument("--city", type=int, default=None, help="Single city by ID")
    parser.add_argument("--date", type=str, default=DEFAULT_SIM_DATE,
                        help=f"Simulation date YYYY-MM-DD (default: {DEFAULT_SIM_DATE})")
    parser.add_argument("--auto-select-day", action="store_true",
                        help="Auto-select clearest day per city by max daily GHI")
    parser.add_argument("--month", type=str, default=DEFAULT_MONTH,
                        help=f"Month to search for clear-sky day YYYY-MM (default: {DEFAULT_MONTH})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip CDS API download, only convert existing NetCDF")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing met files")
    args = parser.parse_args()

    json_path = args.cities_json or (args.output_dir / "cities.json")
    if not json_path.exists():
        log.error(f"cities.json not found at {json_path}. "
                   "Run prepare_nature_cities.py --step select first.")
        sys.exit(1)

    with open(json_path) as f:
        cities = json.load(f)

    if args.city is not None:
        cities = [c for c in cities if c["id"] == args.city]
        if not cities:
            log.error(f"City ID {args.city} not found")
            sys.exit(1)

    log.info(f"Processing {len(cities)} cities")

    if args.validate_only:
        ok = 0
        for city in cities:
            met_path = args.output_dir / str(city["id"]) / "met.txt"
            if met_path.exists():
                ok += validate_met(met_path)
            else:
                log.warning(f"  City {city['id']} ({city['name']}): missing")
        log.info(f"Validated {ok}/{len(cities)} OK")
        if ok != len(cities):
            raise SystemExit(1)
        return

    if args.auto_select_day:
        # Download full month + select clearest day per city
        log.info(f"[1] Downloading full ERA5 month {args.month}...")
        download_era5_full_month(cities, args.output_dir, args.month,
                                 skip_download=args.skip_download)

        log.info(f"\n[2] Selecting clear-sky day per city from {args.month}...")
        raw_dir = args.output_dir / "era5_raw"
        clear_sky_days = {}
        for city in cities:
            best_date, daily_ghi = select_clear_sky_day(city, raw_dir, args.month)
            if best_date:
                clear_sky_days[str(city["id"])] = {
                    "date": best_date,
                    "daily_ghi": round(daily_ghi, 1),
                }

        # Save clear_sky_days.json
        cs_path = args.output_dir / "clear_sky_days.json"
        with open(cs_path, "w") as f:
            json.dump(clear_sky_days, f, indent=2)
        log.info(f"  Saved clear-sky selections to {cs_path}")

        # Log distribution of selected days
        day_counts = defaultdict(int)
        for info in clear_sky_days.values():
            day_counts[info["date"]] += 1
        log.info(f"  Day distribution: {dict(sorted(day_counts.items()))}")

        log.info(f"\n[3] Converting ERA5 -> UMEP met for {len(cities)} cities...")
        success = 0
        for i, city in enumerate(cities):
            cid_str = str(city["id"])
            if cid_str not in clear_sky_days:
                log.warning(f"  City {city['id']}: no clear-sky day selected, skipping")
                continue
            sim_date = clear_sky_days[cid_str]["date"]
            if (i + 1) % 50 == 0:
                log.info(f"  Progress: {i+1}/{len(cities)}")
            met_path = convert_era5_to_met(city, args.output_dir, sim_date)
            if met_path is not None and validate_met(met_path):
                success += 1

        log.info(f"\nDone: {success}/{len(cities)} cities converted")
        if success != len(cities):
            raise SystemExit(1)

    else:
        # Original mode: download specific date for all cities
        log.info(f"[1] Downloading ERA5 data for {args.date}...")
        download_era5_for_cities(cities, args.output_dir, args.date,
                                 skip_download=args.skip_download)

        log.info(f"\n[2] Converting ERA5 -> UMEP met for {len(cities)} cities...")
        success = 0
        for i, city in enumerate(cities):
            if (i + 1) % 50 == 0:
                log.info(f"  Progress: {i+1}/{len(cities)}")
            met_path = convert_era5_to_met(city, args.output_dir, args.date)
            if met_path is not None and validate_met(met_path):
                success += 1

        log.info(f"\nDone: {success}/{len(cities)} cities converted")
        if success != len(cities):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
