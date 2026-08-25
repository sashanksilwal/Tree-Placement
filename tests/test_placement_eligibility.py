"""End-to-end checks that placement respects the ground it is planting on.

These exercise the path ``run_city`` uses -- OSM masks feeding ``place_trees``
-- on a small scene containing the surfaces a tree must never be planted on,
and assert the two properties the study depends on:

* a trunk never lands on a road, water body, building, parking lot, pool, or
  sidewalk, and a crown never overlaps a building;
* what gets placed is a *tree* -- a trunk carrying a full crown of a plausible
  height, spaced from its neighbours -- rather than loose canopy pixels.
"""

import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform as project_points  # noqa: E402

from study_code.geospatial_preprocessing.osm_eligibility import (  # noqa: E402
    REQUIRED_LAYERS,
    build_osm_placement_masks,
)
from study_code.tree_placement import (  # noqa: E402
    PlacementConfig,
    crown_offsets,
    place_trees,
)

GRID = 200
CRS = "EPSG:3857"


def _to_wgs84(geometry):
    def point(coordinate):
        longitude, latitude = project_points(
            CRS, "EPSG:4326", [coordinate[0]], [coordinate[1]]
        )
        return [longitude[0], latitude[0]]

    if geometry["type"] == "LineString":
        coordinates = [point(c) for c in geometry["coordinates"]]
    else:
        coordinates = [[point(c) for c in ring] for ring in geometry["coordinates"]]
    return {"type": geometry["type"], "coordinates": coordinates}


def _feature(geometry, **properties):
    return {
        "type": "Feature",
        "geometry": _to_wgs84(geometry),
        "properties": properties,
    }


def _line(coordinates):
    return {"type": "LineString", "coordinates": coordinates}


def _box(x0, y0, x1, y1):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


@pytest.fixture(scope="module")
def city_block(tmp_path_factory):
    """A block with a primary road, a side street, water, buildings, parking, a pool."""
    root = tmp_path_factory.mktemp("eligibility")
    osm_dir = root / "osm"
    osm_dir.mkdir()
    layers = {
        "highway": [
            _feature(_line([[0, 100], [200, 100]]), highway="primary"),
            _feature(_line([[50, 0], [50, 200]]), highway="residential"),
        ],
        "sidewalk": [
            _feature(
                _line([[60, 0], [60, 200]]), highway="footway", footway="sidewalk"
            )
        ],
        "building": [_feature(_box(120, 120, 160, 160), building="yes")],
        "parking": [_feature(_box(20, 150, 45, 180), amenity="parking")],
        "residential": [_feature(_box(0, 0, 200, 200), landuse="residential")],
        "water": [_feature(_box(160, 20, 190, 80), natural="water")],
        "pool": [_feature(_box(100, 30, 115, 45), leisure="swimming_pool")],
    }
    for layer in REQUIRED_LAYERS:
        (osm_dir / f"{layer}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": layers[layer]})
        )

    transform = from_origin(0.0, float(GRID), 1.0, 1.0)
    profile = dict(
        driver="GTiff", height=GRID, width=GRID, count=1, dtype="float32",
        crs=CRS, transform=transform,
    )
    cdsm = np.zeros((GRID, GRID), dtype=np.float32)
    # Two canopy patches of different height, so height assignment has to choose.
    cdsm[10:20, 10:20] = 6.0
    cdsm[170:185, 100:120] = 14.0
    dem = np.zeros((GRID, GRID), dtype=np.float32)
    dsm = np.zeros((GRID, GRID), dtype=np.float32)
    dsm[120:160, 120:160] = 15.0   # mapped in OSM too
    dsm[70:90, 20:40] = 11.0       # DSM-only building OSM never mapped
    for name, array in (("CDSM", cdsm), ("DEM", dem), ("Building_DSM", dsm)):
        with rasterio.open(root / f"{name}.tif", "w", **profile) as destination:
            destination.write(array, 1)

    rng = np.random.default_rng(0)
    svf = np.clip(rng.normal(0.7, 0.1, (GRID, GRID)), 0.01, 1.0).astype(np.float32)
    tsfc = np.zeros((16, GRID, GRID), dtype=np.float32)
    base = 300 + 12 * rng.random((GRID, GRID))
    for hour in range(16):
        tsfc[hour] = base + hour * 0.4
    return root, cdsm, dem, dsm, svf, tsfc


def _place(city_block, domain, strategy):
    root, cdsm, dem, dsm, svf, tsfc = city_block
    masks = build_osm_placement_masks(root / "osm", root / "CDSM.tif", domain, "all")
    config = PlacementConfig(
        dose_pp=3.0, analysis_buffer_px=10, crown_radius_px=4,
        placement_domain=domain, road_scope="all",
    )
    modified, centres, summary = place_trees(
        cdsm, dsm, dem, tsfc, svf, strategy, config,
        centre_eligibility=masks.centre_eligible,
        additional_buildings=masks.osm_buildings,
        landcover=np.ones((GRID, GRID), dtype=np.float32),
    )
    assert len(centres), f"{strategy} placed nothing, so nothing is being verified"
    return masks, config, modified, centres, summary


@pytest.mark.parametrize(
    "domain,strategy",
    [("everywhere", "hotspot"), ("street-verge", "street_hotspot")],
)
def test_no_tree_is_planted_on_an_unplantable_surface(city_block, domain, strategy):
    _, cdsm, dem, dsm, _, _ = city_block
    masks, config, _, centres, _ = _place(city_block, domain, strategy)
    rows = centres[:, 0].astype(int)
    cols = centres[:, 1].astype(int)

    unplantable = {
        "road surface": masks.layer_masks["road_surface"],
        "water": masks.layer_masks["water"],
        "parking": masks.layer_masks["parking"],
        "swimming pool": masks.layer_masks["pool"],
        "OSM building": masks.layer_masks["building"],
    }
    if domain == "everywhere":
        unplantable["sidewalk"] = masks.layer_masks["sidewalk"]
    for label, mask in unplantable.items():
        assert not mask[rows, cols].any(), f"a trunk landed on {label}"

    buildings = np.isfinite(dsm) & (
        (dsm - dem) > config.building_height_threshold_m
    )
    assert not buildings[rows, cols].any(), "a trunk landed on a building"
    assert not (cdsm[rows, cols] > 0).any(), "a trunk landed inside existing canopy"

    # The crown, not only the trunk, has to clear buildings and the grid edge.
    dr, dc = crown_offsets(config.crown_radius_px)
    crown_rows = (rows[:, None] + dr[None, :]).ravel()
    crown_cols = (cols[:, None] + dc[None, :]).ravel()
    assert crown_rows.min() >= 0 and crown_rows.max() < GRID
    assert crown_cols.min() >= 0 and crown_cols.max() < GRID
    assert not buildings[crown_rows, crown_cols].any(), "a crown overlapped a building"


@pytest.mark.parametrize(
    "domain,strategy",
    [("everywhere", "hotspot"), ("street-verge", "street_hotspot")],
)
def test_placement_adds_trees_rather_than_scattered_canopy_pixels(
    city_block, domain, strategy
):
    _, config, modified, centres, summary = _place(city_block, domain, strategy)
    rows = centres[:, 0].astype(int)
    cols = centres[:, 1].astype(int)
    heights = centres[:, 2]
    dr, dc = crown_offsets(config.crown_radius_px)

    footprint = len(dr)
    assert footprint > 1, "a crown must be a disc, not a single pixel"

    crown_rows = (rows[:, None] + dr[None, :]).ravel()
    crown_cols = (cols[:, None] + dc[None, :]).ravel()
    assert (modified[crown_rows, crown_cols] > 0).all(), (
        "every pixel of every crown must carry canopy height"
    )

    assert ((heights >= 2.0) & (heights <= 50.0)).all(), "implausible tree height"
    assert len(heights) == len(centres), "every tree needs its own height"

    # Trees are counted separately from the pixels they cover, and the pixel
    # count has to be consistent with whole crowns rather than loose pixels.
    assert summary["tree_centres"] == len(centres)
    assert summary["crown_footprint_pixels"] == footprint
    assert summary["pixels_added"] >= len(centres) * (footprint // 2)

    # Distinct trunks, not one blob: spacing floor is respected.
    if len(centres) > 1:
        squared = (rows[:, None] - rows[None, :]) ** 2 + (
            cols[:, None] - cols[None, :]
        ) ** 2
        np.fill_diagonal(squared, 10**9)
        assert np.sqrt(squared.min()) >= summary["minimum_centre_distance_px"]


def test_street_verge_trees_sit_beside_the_carriageway_not_on_it(city_block):
    masks, _, _, centres, _ = _place(city_block, "street-verge", "street_hotspot")
    rows = centres[:, 0].astype(int)
    cols = centres[:, 1].astype(int)
    assert not masks.layer_masks["street_carriageway"][rows, cols].any()
    in_ring = masks.layer_masks["street_verge"] | masks.layer_masks["sidewalk"]
    assert in_ring[rows, cols].all(), "a verge tree fell outside every planting ring"
