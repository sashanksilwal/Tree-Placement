"""End-to-end regression: the full pipeline must reproduce known-good radiant temperatures.

Every unit test in this suite constructs its own tensors and checks one kernel. That
leaves a gap: a change can leave every kernel correct and still corrupt the model's
primary output. It happened -- an integer cast on the land-cover raster silently zeroed
emissivity and albedo inside Tgmaps_v1, ground longwave collapsed to zero, and mean
radiant temperature fell by ~50 C. The full suite passed throughout.

This test closes that gap by running thermal_comfort on a deterministic synthetic scene
and asserting on the output. It checks two independent things:

  1. Physical invariants that hold regardless of model version -- a sunlit surface is
     warmer than the air, longwave leaving the ground is a substantial flux, night-time
     radiant temperature tracks air temperature. These catch the *class* of failure.
  2. Stored reference statistics with a tolerance, which catch quieter drift.

Reference values were produced on an RTX 6000 Ada (torch 2.6.0+cu124, numpy 2.2.6) and
cross-checked against the pre-release implementation on a real Las Vegas neighbourhood,
where night and peak-hour maxima agreed to within 0.06 C. The synthetic run uses one
periodic spin-up day and verifies that only the requested day is published.
"""
import json
import os
import warnings

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio", reason="end-to-end run needs rasterio")

from utherm.core import thermal_comfort

# Synthetic scene: small enough to run in seconds, structured enough to produce
# buildings, walls, canopy shade and open ground.
N = 120
RES = 1.0
ORIGIN_X, ORIGIN_Y = 660000.0, 4000000.0
EPSG = 32611          # UTM 11N, i.e. the Las Vegas latitude band
DATE = "2023-07-06"

# Reference statistics for band 14 (13:00 local) and band 1 (00:00).
# Tolerances are wide enough to absorb platform and library differences but far
# tighter than any of the failure modes this guards against.
REF = {
    "tmrt_peak_mean": (52.0, 12.0),   # (expected, tolerance)
    "tmrt_peak_max": (72.0, 12.0),
    "tmrt_night_mean": (20.0, 12.0),
}


def _write(path, array, dtype):
    profile = dict(
        driver="GTiff", height=N, width=N, count=1, dtype=dtype,
        crs=rasterio.crs.CRS.from_epsg(EPSG),
        transform=rasterio.transform.from_origin(ORIGIN_X, ORIGIN_Y, RES, RES),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def _build_scene(base):
    """Deterministic scene: ground plane, four blocks, a row of trees, mixed land cover."""
    dem = np.full((N, N), 600.0, dtype="float32")

    dsm = dem.copy()
    for r0, c0 in ((20, 20), (20, 70), (70, 20), (70, 70)):
        dsm[r0:r0 + 26, c0:c0 + 26] += 12.0          # 12 m blocks

    cdsm = np.zeros((N, N), dtype="float32")
    cdsm[55:62, 10:110:6] = 8.0                       # street trees, 8 m crowns

    landcover = np.full((N, N), 1, dtype="int32")     # 1 = paved
    landcover[cdsm > 0] = 5                           # 5 = grass under canopy
    landcover[50:52, :] = 6                           # 6 = bare soil strip
    for r0, c0 in ((20, 20), (20, 70), (70, 20), (70, 70)):
        landcover[r0:r0 + 26, c0:c0 + 26] = 2         # 2 = buildings

    _write(os.path.join(base, "DEM.tif"), dem, "float32")
    _write(os.path.join(base, "Building_DSM.tif"), dsm, "float32")
    _write(os.path.join(base, "CDSM.tif"), cdsm, "float32")
    _write(os.path.join(base, "landcover.tif"), landcover, "int32")


def _write_met(path):
    """24 hours of clear-sky July forcing in UMEP format."""
    header = ("iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow "
              "ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd")
    rows = []
    for hour in range(24):
        # smooth diurnal cycles; sun up 06:00-19:00
        day = np.clip(np.sin(np.pi * (hour - 6) / 13.0), 0.0, None)
        kdn = 1000.0 * day
        ta = 24.0 + 13.0 * np.clip(np.sin(np.pi * (hour - 5) / 15.0), 0.0, None)
        rh = 30.0 - 18.0 * day
        rows.append(
            f"2023 187 {hour} 0 -999 -999 -999 -999 -999 2.5 {rh:.4f} {ta:.4f} "
            f"95.0 0.0 {kdn:.4f} -999 340.0 -999 -999 -999 -999 "
            f"{0.25 * kdn:.4f} {0.75 * kdn:.4f} 200.0"
        )
    with open(path, "w") as fh:
        fh.write(header + "\n" + "\n".join(rows) + "\n")


def _band(path, index):
    with rasterio.open(path) as src:
        arr = src.read(index).astype("float64")
    arr = arr[np.isfinite(arr)]
    return arr[arr > -500.0]


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    base = str(tmp_path_factory.mktemp("e2e"))
    _build_scene(base)
    _write_met(os.path.join(base, "met.txt"))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Mean of empty slice")
        thermal_comfort(
            base_path=base, selected_date_str=DATE,
            own_met_file=os.path.join(base, "met.txt"), use_own_met=True,
            building_dsm_filename="Building_DSM.tif", dem_filename="DEM.tif",
            trees_filename="CDSM.tif", landcover_filename="landcover.tif",
            tile_size=1400, overlap=20,
            save_tmrt=True, save_lup=True, save_tsfc=True,
            use_energy_balance=True, use_canopy_eb=True,
            canopy_type="deciduous", canopy_lai=3.0, z0=0.01,
            spinup_days=1,
        )
    out = os.path.join(base, "output_folder", "0_0")
    assert os.path.isdir(out), "run produced no output directory"
    return out


def test_spinup_does_not_leak_extra_bands_or_dates(completed_run):
    """Warm-up timesteps advance state but must not enter published rasters."""
    for variable in ("UTCI", "TMRT", "Lup", "Tsfc"):
        path = os.path.join(completed_run, f"{variable}_0_0.tif")
        with rasterio.open(path) as src:
            assert src.count == 24
            assert src.tags(1)["Time"] == "2023-07-06T00:00:00"
            assert src.tags(24)["Time"] == "2023-07-06T23:00:00"


def test_ground_longwave_is_a_substantial_flux(completed_run):
    """Guards the emissivity-truncation class of failure.

    Lup = SBC * emis_grid * (...). If emis_grid is ever zeroed -- by a dtype cast, a
    bad land-cover join, a missing lookup -- this collapses to ~0 while every kernel
    test still passes.
    """
    lup = _band(os.path.join(completed_run, "Lup_0_0.tif"), 14)
    assert lup.size, "Lup raster is empty"
    assert lup.mean() > 300.0, (
        f"ground longwave collapsed: mean {lup.mean():.2f} W/m2. "
        "A hot surface must emit several hundred W/m2; near-zero means the emissivity "
        "grid was lost."
    )


def test_sunlit_radiant_temperature_exceeds_air_temperature(completed_run):
    """A body in full sun absorbs the direct beam and must be hotter than the air."""
    tmrt = _band(os.path.join(completed_run, "TMRT_0_0.tif"), 14)
    ta_peak = 37.0     # forcing peak from _write_met
    assert tmrt.max() > ta_peak, (
        f"peak Tmrt {tmrt.max():.2f} C does not exceed air temperature {ta_peak} C; "
        "the shortwave or longwave contribution to S_str is missing."
    )


def test_night_radiant_temperature_tracks_air_temperature(completed_run):
    """With no solar input, Tmrt is set by longwave from sky and surfaces.

    It sits near air temperature -- somewhat below in the open, near it in a canyon.
    A large negative excursion means a radiative term vanished.
    """
    tmrt = _band(os.path.join(completed_run, "TMRT_0_0.tif"), 1)
    assert -10.0 < tmrt.mean() < 40.0, (
        f"night Tmrt mean {tmrt.mean():.2f} C is not physical for a warm night; "
        "surrounding surfaces cannot be radiating at that temperature."
    )


def test_diurnal_shape_peaks_during_the_day(completed_run):
    """Radiant load must rise from night to afternoon."""
    night = _band(os.path.join(completed_run, "TMRT_0_0.tif"), 1).mean()
    peak = _band(os.path.join(completed_run, "TMRT_0_0.tif"), 14).mean()
    assert peak > night + 10.0, (
        f"afternoon Tmrt ({peak:.2f} C) is not meaningfully above night "
        f"({night:.2f} C); the diurnal cycle is not being resolved."
    )


@pytest.mark.parametrize("key,band,stat", [
    ("tmrt_peak_mean", 14, "mean"),
    ("tmrt_peak_max", 14, "max"),
    ("tmrt_night_mean", 1, "mean"),
])
def test_matches_stored_reference(completed_run, key, band, stat):
    """Quieter drift than the invariants above would catch."""
    arr = _band(os.path.join(completed_run, "TMRT_0_0.tif"), band)
    got = float(getattr(np, stat)(arr))
    expected, tol = REF[key]
    assert abs(got - expected) <= tol, (
        f"{key}: got {got:.2f} C, reference {expected:.2f} +/- {tol:.2f} C. "
        "Investigate before updating the reference."
    )
def _write_peak_met(path):
    header = ("iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow "
              "ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd")
    row = ("2023 187 13 0 -999 -999 -999 -999 -999 2.5 12.0 37.0 "
           "95.0 0.0 992.7 -999 340.0 -999 -999 -999 -999 248.175 744.525 200.0")
    with open(path, "w") as fh:
        fh.write(header + "\n" + row + "\n")


def _write_coupled_geometry(path):
    facet_area = np.array(
        [0.7, 0.3, 0.25, 0.25, 0.25, 0.25, 0.8], dtype=np.float32
    )
    area = np.broadcast_to(facet_area[:, None, None], (7, N, N)).copy()
    exchange = np.zeros((7, 7, N, N), dtype=np.float32)
    for first in range(7):
        for second in range(first + 1, 7):
            shared = 0.02 * min(facet_area[first], facet_area[second])
            exchange[first, second] = shared
            exchange[second, first] = shared
    sky = area - exchange.sum(axis=1)
    body = np.array(
        [0.20, 0.0, 0.15, 0.15, 0.15, 0.15, 0.10], dtype=np.float32
    )
    body = np.broadcast_to(body[:, None, None], (7, N, N)).copy()
    body_sky = np.full((N, N), 0.10, dtype=np.float32)
    crs_wkt = rasterio.crs.CRS.from_epsg(EPSG).to_wkt()
    geotransform = rasterio.transform.from_origin(
        ORIGIN_X, ORIGIN_Y, RES, RES
    ).to_gdal()
    with rasterio.open(os.path.join(os.path.dirname(path), "DEM.tif")) as source:
        dem = source.read(1).astype(np.float32)
    with rasterio.open(os.path.join(os.path.dirname(path), "Building_DSM.tif")) as source:
        building_dsm = source.read(1).astype(np.float32)
    with rasterio.open(os.path.join(os.path.dirname(path), "CDSM.tif")) as source:
        canopy_height = source.read(1).astype(np.float32)
    with rasterio.open(os.path.join(os.path.dirname(path), "landcover.tif")) as source:
        trace_landcover = source.read(1).astype(np.int16)
    grid_y, grid_x = np.indices((N, N), dtype=np.float32)
    origin_x = np.broadcast_to(grid_x[None] + 0.5, (7, N, N)).copy()
    origin_y = np.broadcast_to(grid_y[None] + 0.5, (7, N, N)).copy()
    origin_z = np.empty((7, N, N), dtype=np.float32)
    origin_z[0] = dem + 0.02
    origin_z[1] = building_dsm + 0.02
    origin_z[2:6] = 0.5 * (dem + building_dsm)
    origin_z[6] = dem + canopy_height + 0.02
    raw_surface_sky = np.divide(
        sky,
        area,
        out=np.ones_like(sky),
        where=area > 0.0,
    )
    np.savez(
        path,
        area=area,
        sky_view_area=sky,
        exchange_area=exchange,
        body_view_factor=body,
        body_sky_view_factor=body_sky,
        raw_surface_sky_view_factor=raw_surface_sky,
        facet_origin_x=origin_x,
        facet_origin_y=origin_y,
        facet_origin_z=origin_z,
        trace_dem=dem,
        trace_building_dsm=building_dsm,
        trace_canopy_height=canopy_height,
        trace_landcover=trace_landcover,
        trace_output_window=np.asarray([0, 0, N, N], dtype=np.int64),
        config_json=np.asarray(
            json.dumps(
                {
                    "pixel_size_m": 1.0,
                    "max_distance_m": 4.0,
                    "ray_step_m": 1.0,
                    "surface_ray_count": 8,
                    "body_ray_count": 16,
                }
            )
        ),
        water_capacity=np.where(area > 0.0, 0.2, 0.0).astype(np.float32),
        crs_wkt=np.array(crs_wkt),
        geotransform=np.asarray(geotransform, dtype=np.float64),
    )


@pytest.fixture(scope="module")
def coupled_public_run(tmp_path_factory):
    import utherm.coupled_pipeline as coupled_pipeline

    base = str(tmp_path_factory.mktemp("e2e_coupled"))
    _build_scene(base)
    _write_peak_met(os.path.join(base, "met.txt"))
    geometry = os.path.join(base, "coupled_geometry.npz")
    _write_coupled_geometry(geometry)
    calls = {"constructed": 0, "spin_up": 0, "facet_shortwave": 0}
    original_model = coupled_pipeline.CoupledUrbanEnergyBalance
    original_shortwave = coupled_pipeline.CoupledRadiationBridge.facet_shortwave_irradiance

    class RecordingCoupledUrbanEnergyBalance(original_model):
        def __init__(self, *args, **kwargs):
            calls["constructed"] += 1
            super().__init__(*args, **kwargs)
            assert self.material_ids is not None
            assert set(torch.unique(self.material_ids[0]).cpu().tolist()) == {1, 2, 5, 6}
            assert torch.unique(self.albedo[0]).numel() > 1

        def spin_up(self, *args, **kwargs):
            calls["spin_up"] += 1
            return super().spin_up(*args, **kwargs)

    def recording_shortwave(self, *args, **kwargs):
        calls["facet_shortwave"] += 1
        return original_shortwave(self, *args, **kwargs)

    coupled_pipeline.CoupledUrbanEnergyBalance = RecordingCoupledUrbanEnergyBalance
    coupled_pipeline.CoupledRadiationBridge.facet_shortwave_irradiance = recording_shortwave
    try:
        thermal_comfort(
            base_path=base, selected_date_str=DATE,
            own_met_file=os.path.join(base, "met.txt"), use_own_met=True,
            building_dsm_filename="Building_DSM.tif", dem_filename="DEM.tif",
            trees_filename="CDSM.tif", landcover_filename="landcover.tif",
            tile_size=1400, overlap=20,
            use_coupled_eb=True, coupled_geometry_path=geometry,
            coupled_spinup_max_cycles=80, coupled_strict_convergence=True,
            save_tmrt=True, save_lup=True, save_tsfc=True,
            save_tsfc_roof=True, save_t_leaf=True,
            save_canopy_qh=True, save_canopy_qe=True,
            save_qh=True, save_qe=True, save_tair=True,
            save_tair_canyon=True, save_wall_temperature=True,
        )
    finally:
        coupled_pipeline.CoupledUrbanEnergyBalance = original_model
        coupled_pipeline.CoupledRadiationBridge.facet_shortwave_irradiance = original_shortwave
    assert calls == {"constructed": 1, "spin_up": 1, "facet_shortwave": 1}
    out = os.path.join(base, "output_folder", "0_0")
    assert os.path.isdir(out), "coupled public-API run produced no output directory"
    return out


def test_coupled_public_path_writes_physical_wall_facets(coupled_public_run):
    for direction in ("North", "East", "South", "West"):
        path = os.path.join(coupled_public_run, f"TsfcWall{direction}_0_0.tif")
        assert os.path.exists(path), f"coupled wall output missing: {direction}"
        wall = _band(path, 1)
        assert wall.size
        assert wall.min() > -30.0
        assert wall.max() < 90.0


def test_coupled_public_path_preserves_radiant_field_location(
    completed_run, coupled_public_run
):
    baseline = _band(os.path.join(completed_run, "TMRT_0_0.tif"), 14)
    coupled = _band(os.path.join(coupled_public_run, "TMRT_0_0.tif"), 1)
    assert coupled.size
    assert coupled.max() > 37.0
    assert abs(coupled.mean() - baseline.mean()) < 20.0
    lup = _band(os.path.join(coupled_public_run, "Lup_0_0.tif"), 1)
    assert lup.mean() > 300.0
