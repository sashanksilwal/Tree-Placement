import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")
from rasterio.transform import from_origin, rowcol
from rasterio.warp import transform as project_points

from study_code.geospatial_preprocessing.osm_eligibility import (
    REQUIRED_LAYERS,
    build_osm_placement_masks,
    osm_to_geojson,
)


def _wgs84_geometry(geometry):
    def point(coordinate):
        longitude, latitude = project_points(
            "EPSG:3857", "EPSG:4326", [coordinate[0]], [coordinate[1]]
        )
        return [longitude[0], latitude[0]]

    if geometry["type"] == "LineString":
        coordinates = [point(value) for value in geometry["coordinates"]]
    elif geometry["type"] == "Polygon":
        coordinates = [
            [point(value) for value in ring] for ring in geometry["coordinates"]
        ]
    else:
        raise AssertionError("test helper supports lines and polygons only")
    return {"type": geometry["type"], "coordinates": coordinates}


def _feature(geometry, **properties):
    return {
        "type": "Feature",
        "geometry": _wgs84_geometry(geometry),
        "properties": properties,
    }


def _polygon(x0, y0, x1, y1):
    return {
        "type": "Polygon",
        "coordinates": [
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        ],
    }


def _write_fixture(tmp_path):
    reference = tmp_path / "CDSM.tif"
    grid_transform = from_origin(0.0, 80.0, 1.0, 1.0)
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        height=80,
        width=80,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=grid_transform,
    ) as destination:
        destination.write(np.zeros((1, 80, 80), dtype=np.float32))

    osm_dir = tmp_path / "osm"
    osm_dir.mkdir()
    layers = {
        "highway": [
            _feature(
                {"type": "LineString", "coordinates": [[40, 0], [40, 80]]},
                highway="residential",
            )
        ],
        "sidewalk": [
            _feature(
                {"type": "LineString", "coordinates": [[45, 0], [45, 80]]},
                highway="footway",
                footway="sidewalk",
            )
        ],
        "building": [_feature(_polygon(10, 10, 20, 20), building="yes")],
        "parking": [_feature(_polygon(60, 10, 70, 20), amenity="parking")],
        "residential": [
            _feature(_polygon(30, 10, 55, 70), landuse="residential")
        ],
        "water": [
            _feature(
                {"type": "LineString", "coordinates": [[25, 0], [25, 80]]},
                waterway="stream",
            )
        ],
        "pool": [
            _feature(_polygon(10, 60, 20, 70), leisure="swimming_pool")
        ],
    }
    for layer in REQUIRED_LAYERS:
        payload = {"type": "FeatureCollection", "features": layers[layer]}
        (osm_dir / f"{layer}.geojson").write_text(json.dumps(payload))
    return osm_dir, reference, grid_transform


def _pixel(grid_transform, x, y):
    return rowcol(grid_transform, x, y)


def test_osm_domains_exclude_mapped_features_and_define_verge(tmp_path):
    osm_dir, reference, grid_transform = _write_fixture(tmp_path)
    everywhere = build_osm_placement_masks(
        osm_dir, reference, "everywhere", "local"
    )
    verge = build_osm_placement_masks(
        osm_dir, reference, "street-verge", "local"
    )

    assert not everywhere.centre_eligible[_pixel(grid_transform, 15, 15)]
    assert not everywhere.centre_eligible[_pixel(grid_transform, 65, 15)]
    assert not everywhere.centre_eligible[_pixel(grid_transform, 15, 65)]
    assert not everywhere.centre_eligible[_pixel(grid_transform, 25, 40)]
    assert not everywhere.centre_eligible[_pixel(grid_transform, 40, 40)]
    assert not everywhere.centre_eligible[_pixel(grid_transform, 45, 40)]
    assert everywhere.centre_eligible[_pixel(grid_transform, 50, 40)]

    assert verge.centre_eligible.any()
    assert verge.centre_eligible[_pixel(grid_transform, 44, 40)]
    assert not verge.centre_eligible[_pixel(grid_transform, 48, 40)]
    assert verge.audit["verge_rings_m"]["residential"] == pytest.approx([3.5, 7.0])
    assert verge.audit["verge_default_ring_m"] == pytest.approx([4.0, 8.0])
    assert verge.audit["verge_reference"] == "road-centreline"
    assert verge.audit["sidewalks_supplement_verge"]
    assert "residential" not in verge.audit["applied_exclusion_layers"]
    assert "sidewalk" in everywhere.audit["applied_exclusion_layers"]
    assert verge.audit["applied_exclusion_union_pixels"] > 0
    assert verge.street_analysis.any()
    assert verge.layer_masks["residential"].any()
    assert verge.audit["osm_sources"]["highway"]["sha256"]
    assert verge.audit["reference_raster"]["sha256"]


def test_osm_layers_are_required_instead_of_silently_skipped(tmp_path):
    _, reference, _ = _write_fixture(tmp_path)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "highway.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        build_osm_placement_masks(incomplete, reference, "everywhere")


def test_overpass_way_conversion_preserves_tags_and_geometry():
    payload = json.dumps(
        {
            "elements": [
                {
                    "type": "way",
                    "id": 7,
                    "tags": {"highway": "residential"},
                    "geometry": [
                        {"lon": -86.0, "lat": 40.0},
                        {"lon": -85.999, "lat": 40.001},
                    ],
                }
            ]
        }
    ).encode()
    result = osm_to_geojson(payload)
    assert result["features"][0]["geometry"]["type"] == "LineString"
    assert result["features"][0]["properties"]["highway"] == "residential"


def test_overpass_relation_conversion_retains_inner_holes():
    from shapely.geometry import shape

    def member(role, coordinates):
        return {
            "type": "way",
            "role": role,
            "geometry": [
                {"lon": longitude, "lat": latitude}
                for longitude, latitude in coordinates
            ],
        }

    outer = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    inner = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
    payload = json.dumps(
        {
            "elements": [
                {
                    "type": "relation",
                    "id": 8,
                    "tags": {"building": "yes", "type": "multipolygon"},
                    "members": [member("outer", outer), member("inner", inner)],
                }
            ]
        }
    ).encode()
    result = osm_to_geojson(payload)
    geometry = shape(result["features"][0]["geometry"])
    assert geometry.area == pytest.approx(12.0)
    assert not geometry.contains(shape({"type": "Point", "coordinates": [2, 2]}))


def _write_single_road_fixture(tmp_path, highway):
    """One straight highway of the given class down the middle of an 80 m grid."""
    reference = tmp_path / "CDSM.tif"
    grid_transform = from_origin(0.0, 80.0, 1.0, 1.0)
    with rasterio.open(
        reference, "w", driver="GTiff", height=80, width=80, count=1,
        dtype="float32", crs="EPSG:3857", transform=grid_transform,
    ) as destination:
        destination.write(np.zeros((1, 80, 80), dtype=np.float32))
    osm_dir = tmp_path / "osm"
    osm_dir.mkdir()
    empty = {"type": "FeatureCollection", "features": []}
    for layer in REQUIRED_LAYERS:
        (osm_dir / f"{layer}.geojson").write_text(json.dumps(empty))
    (osm_dir / "highway.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature(
                        {"type": "LineString", "coordinates": [[40, 0], [40, 80]]},
                        highway=highway,
                    )
                ],
            }
        )
    )
    return osm_dir, reference, grid_transform


@pytest.mark.parametrize(
    "highway,inner,outer",
    [("primary", 8.0, 14.0), ("secondary", 6.0, 11.0), ("motorway", 15.0, 25.0)],
)
def test_verge_never_lands_on_the_carriageway_of_wide_roads(
    tmp_path, highway, inner, outer
):
    """A verge centre must sit beside the pavement, never on it.

    The class ring starts at the carriageway half-width, so every eligible
    centre is strictly outside it. A fixed 3--6 m band would place trees in the
    traffic lanes of any road wider than 6 m.
    """
    osm_dir, reference, grid_transform = _write_single_road_fixture(tmp_path, highway)
    masks = build_osm_placement_masks(osm_dir, reference, "street-verge", "all")

    assert masks.centre_eligible.any()
    # Anywhere inside the carriageway is ineligible ...
    for offset in (0.0, inner - 1.5):
        for side in (-1, 1):
            pixel = _pixel(grid_transform, 40 + side * offset, 40)
            assert not masks.centre_eligible[pixel], (highway, offset, side)
    # ... while the ring beside it is eligible ...
    for side in (-1, 1):
        pixel = _pixel(grid_transform, 40 + side * (inner + 1.5), 40)
        assert masks.centre_eligible[pixel], (highway, side)
    # ... and beyond the planting strip it stops again.
    for side in (-1, 1):
        pixel = _pixel(grid_transform, 40 + side * (outer + 3.0), 40)
        assert not masks.centre_eligible[pixel], (highway, side)


def test_area_mapped_road_surface_is_excluded_across_its_interior(tmp_path):
    """A plaza mapped as a polygon is paved all the way across, not just at its edge."""
    osm_dir, reference, grid_transform = _write_single_road_fixture(tmp_path, "service")
    (osm_dir / "highway.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature(_polygon(20, 20, 60, 60), highway="pedestrian", area="yes")
                ],
            }
        )
    )
    masks = build_osm_placement_masks(osm_dir, reference, "everywhere", "all")
    # Dead centre of the plaza, far from every edge.
    assert not masks.centre_eligible[_pixel(grid_transform, 40, 40)]
    assert not masks.layer_masks["general_eligible"][_pixel(grid_transform, 40, 40)]
