import numpy as np
import pytest

from study_code.tree_placement import (
    AOI_STRATEGIES,
    STREET_STRATEGIES,
    PlacementConfig,
    _base_output_pair,
    _read_raster,
    calculate_eqcap_trees,
    crown_offsets,
    non_overlapping_centre_distance,
    place_trees,
    canonical_strategy,
)


def scene():
    shape = (80, 80)
    dem = np.zeros(shape, dtype=np.float32)
    building = dem.copy()
    building[34:42, 34:42] = 10.0
    cdsm = np.zeros(shape, dtype=np.float32)
    cdsm[12:19, 12:19] = 8.0
    tsfc = np.full((24, *shape), 300.0, dtype=np.float32)
    tsfc[:, 20:60, 20:60] = 304.0
    tsfc[:, 48:66, 48:66] = 308.0
    svf = np.full(shape, 0.3, dtype=np.float32)
    svf[10:34, 42:70] = 0.8
    return cdsm, building, dem, tsfc, svf


def test_crown_geometry_has_49_pixels_at_four_metre_radius():
    rows, _ = crown_offsets(4)
    assert len(rows) == 49


def test_unknown_strategy_is_rejected_with_available_choices():
    inputs = scene()
    config = PlacementConfig(dose_pp=1.0, analysis_buffer_px=8, crown_radius_px=3)
    try:
        place_trees(*inputs, "unsupported", config)
    except ValueError as error:
        assert "adaptive" in str(error)
    else:
        raise AssertionError("unsupported placement strategy was accepted")


def test_adaptive_uses_open_sky_and_nonoverlapping_crowns():
    inputs = scene()
    config = PlacementConfig(dose_pp=1.0, analysis_buffer_px=8, crown_radius_px=3)
    _, centres, summary = place_trees(*inputs, "adaptive", config)
    assert summary["adaptive_svf_threshold"] == 0.6
    assert np.all(inputs[-1][centres[:, 0].astype(int), centres[:, 1].astype(int)] > 0.6)
    pairwise = centres[:, None, :2] - centres[None, :, :2]
    distances = np.sqrt(np.sum(pairwise**2, axis=2))
    distances[distances == 0] = np.inf
    assert distances.min() >= 7


def test_legacy_analysis_names_canonicalize_to_public_names():
    assert canonical_strategy("adaptive") == "adaptive"
    assert canonical_strategy("svf_gate") == "adaptive"
    assert canonical_strategy("street_svf_gate") == "street_adaptive"
    assert canonical_strategy("hotspot_add") == "hotspot"


def test_both_strategies_use_the_same_12_to_15_hour_score():
    cdsm, building, dem, tsfc, svf = scene()
    tsfc[:] = 300.0
    # Pixel A wins the four-hour mean; pixel B wins hour 14 alone.
    tsfc[12:16, 24, 24] = (330.0, 330.0, 300.0, 330.0)
    tsfc[12:16, 55, 55] = (300.0, 300.0, 340.0, 300.0)
    svf[:] = 0.8
    config = PlacementConfig(
        dose_pp=0.02,
        analysis_buffer_px=8,
        crown_radius_px=1,
        fixed_tree_height_m=8.0,
    )
    for strategy in ("adaptive", "hotspot"):
        _, centres, _ = place_trees(
            cdsm, building, dem, tsfc, svf, strategy, config
        )
        assert tuple(centres[0, :2].astype(int)) == (24, 24)


def test_placement_rejects_missing_peak_hours():
    cdsm, building, dem, _, svf = scene()
    incomplete = np.full((15, *cdsm.shape), 300.0, dtype=np.float32)
    with pytest.raises(ValueError, match="hours 12--15"):
        place_trees(
            cdsm,
            building,
            dem,
            incomplete,
            svf,
            "hotspot",
            PlacementConfig(dose_pp=1.0, analysis_buffer_px=8),
        )


def test_external_centre_eligibility_restricts_both_strategies():
    cdsm, building, dem, tsfc, svf = scene()
    eligibility = np.zeros(cdsm.shape, dtype=bool)
    eligibility[24, 24] = True
    svf[24, 24] = 0.8
    tsfc[12:16, 24, 24] = 350.0
    config = PlacementConfig(
        dose_pp=0.02,
        analysis_buffer_px=8,
        crown_radius_px=1,
        fixed_tree_height_m=8.0,
    )
    for strategy in ("adaptive", "hotspot"):
        _, centres, summary = place_trees(
            cdsm,
            building,
            dem,
            tsfc,
            svf,
            strategy,
            config,
            centre_eligibility=eligibility,
        )
        assert tuple(centres[0, :2].astype(int)) == (24, 24)
        assert summary["ranking_hours_local"] == [12, 13, 14, 15]


@pytest.mark.parametrize("strategy", ["hotspot", "adaptive", "high_svf"])
def test_ranked_strategies_never_select_nonfinite_required_values(strategy):
    cdsm, building, dem, tsfc, svf = scene()
    eligibility = np.zeros(cdsm.shape, dtype=bool)
    eligibility[24, 24] = True
    eligibility[25, 25] = True
    tsfc[12:16, 24, 24] = np.nan
    tsfc[12:16, 25, 25] = 310.0
    svf[24, 24] = np.nan
    svf[25, 25] = 0.8
    _, centres, summary = place_trees(
        cdsm,
        building,
        dem,
        tsfc,
        svf,
        strategy,
        PlacementConfig(
            dose_value=1,
            dose_mode="eqcap",
            eqcap_trees=1,
            analysis_buffer_px=8,
            placement_geometry="pixel",
            fixed_tree_height_m=8.0,
        ),
        centre_eligibility=eligibility,
    )
    assert tuple(centres[0, :2].astype(int)) == (25, 25)
    assert summary["rankable_candidate_centres"] == 1


def test_additional_building_mask_removes_osm_building_centres():
    cdsm, building, dem, tsfc, svf = scene()
    eligibility = np.zeros(cdsm.shape, dtype=bool)
    eligibility[24, 24] = True
    osm_building = np.zeros(cdsm.shape, dtype=bool)
    osm_building[24, 24] = True
    config = PlacementConfig(dose_pp=0.02, analysis_buffer_px=8, crown_radius_px=1)
    kwargs = {
        "centre_eligibility": eligibility,
        "additional_buildings": osm_building,
    }
    # The OSM building removes the only eligible centre, so nothing can be
    # placed.  That is a failed experiment, not an empty scenario, so a normal
    # run refuses rather than writing an empty placement.
    with pytest.raises(RuntimeError, match="placed no tree"):
        place_trees(cdsm, building, dem, tsfc, svf, "hotspot", config, **kwargs)

    # Capacity probing is the one caller that needs the empty answer back.
    _, centres, summary = place_trees(
        cdsm, building, dem, tsfc, svf, "hotspot", config, allow_empty=True, **kwargs
    )
    assert not len(centres)
    assert summary["constrained"]
    assert not summary["dose"]["target_met"]


def test_height_assignment_mode_is_recorded():
    inputs = scene()
    fixed = PlacementConfig(
        dose_pp=1.0, analysis_buffer_px=8, crown_radius_px=3,
        fixed_tree_height_m=9.0,
    )
    _, _, fixed_summary = place_trees(*inputs, "adaptive", fixed)
    assert fixed_summary["height_assignment"] == "fixed"
    assert fixed_summary["height_mean_m"] == pytest.approx(9.0)

    nearby = PlacementConfig(dose_pp=1.0, analysis_buffer_px=8, crown_radius_px=3)
    _, _, nearby_summary = place_trees(*inputs, "adaptive", nearby)
    assert nearby_summary["height_assignment"] == "nearby_canopy"


def test_placement_reaches_requested_dose_with_complete_crowns():
    inputs = scene()
    config = PlacementConfig(dose_pp=1.0, analysis_buffer_px=8, crown_radius_px=3)
    for strategy in ("adaptive", "hotspot"):
        _, _, summary = place_trees(*inputs, strategy, config)
        assert not summary["constrained"]
        assert summary["pixels_added"] >= summary["target_pixels"]
        assert summary["pixels_added"] - summary["target_pixels"] < summary["crown_footprint_pixels"]


@pytest.mark.parametrize("dose", [0.0, -1.0, 101.0, np.nan])
def test_invalid_dose_is_rejected(dose):
    with pytest.raises(ValueError, match="dose_pp"):
        PlacementConfig(dose_pp=dose)


def test_dose_that_rounds_to_zero_is_rejected():
    inputs = scene()
    config = PlacementConfig(dose_pp=0.001, analysis_buffer_px=8, crown_radius_px=3)
    with pytest.raises(ValueError, match="fewer than one canopy pixel"):
        place_trees(*inputs, "adaptive", config)


def test_mismatched_input_shapes_are_rejected():
    cdsm, building, dem, tsfc, svf = scene()
    with pytest.raises(ValueError, match="input shapes differ"):
        place_trees(
            cdsm,
            building[:-1],
            dem,
            tsfc,
            svf,
            "hotspot",
            PlacementConfig(dose_pp=1.0, analysis_buffer_px=8),
        )


def test_crown_radius_must_be_positive():
    with pytest.raises(ValueError, match="crown_radius_px"):
        PlacementConfig(crown_radius_px=0)


def test_base_output_pair_requires_one_matching_tile(tmp_path):
    root = tmp_path / "output" / "base" / "output_folder" / "7_3"
    root.mkdir(parents=True)
    tsfc = root / "Tsfc_7_3.tif"
    svf = root / "SVF_7_3.tif"
    tsfc.touch()
    svf.touch()
    assert _base_output_pair(tmp_path) == (tsfc, svf)


def test_base_output_pair_rejects_multiple_tiles(tmp_path):
    for key in ("0_0", "100_0"):
        root = tmp_path / "output" / "base" / "output_folder" / key
        root.mkdir(parents=True)
        (root / f"Tsfc_{key}.tif").touch()
        (root / f"SVF_{key}.tif").touch()
    with pytest.raises(RuntimeError, match="requires one base-model tile"):
        _base_output_pair(tmp_path)


def test_read_raster_masks_finite_nodata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "masked.tif"
    data = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32616",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as destination:
        destination.write(data, 1)
    values, _ = _read_raster(path)
    assert np.isnan(values[0, 0, 1])
    assert values[0, 1, 1] == pytest.approx(3.0)


@pytest.mark.parametrize("domain", ["road", "all-ground", "street_verge"])
def test_unknown_placement_domain_is_rejected(domain):
    with pytest.raises(ValueError, match="placement_domain"):
        PlacementConfig(placement_domain=domain)


def _crown_pixels(radius, dr=0, dc=0):
    import numpy as np
    axis = np.arange(-radius, radius + 1)
    rows, cols = np.meshgrid(axis, axis, indexing="ij")
    keep = rows * rows + cols * cols <= radius * radius
    return set(zip((rows[keep] + dr).tolist(), (cols[keep] + dc).tolist()))


def test_non_overlapping_distance_is_the_true_disjointness_threshold():
    """d must be the SMALLEST separation that guarantees zero shared pixels.

    Sufficiency: for every integer offset at Euclidean distance >= d the crowns
    are disjoint. Minimality: at d - 1 (straight along an axis) they still touch,
    so d cannot be reduced.
    """
    for ra in (2, 3, 4, 6):
        for rb in (2, 3, 4, 6):
            d = non_overlapping_centre_distance(ra, rb)
            a = _crown_pixels(ra)
            span = d + 2
            checked = 0
            for dr in range(-span, span + 1):
                for dc in range(-span, span + 1):
                    if dr * dr + dc * dc >= d * d:
                        assert not (a & _crown_pixels(rb, dr, dc)), (ra, rb, dr, dc)
                        checked += 1
            assert checked > 0
            # one pixel closer along an axis they must still share ground
            assert a & _crown_pixels(rb, 0, d - 1)


def test_non_overlapping_distance_matches_the_uniform_crown_rule():
    """Must reproduce the 2r+1 separation the published runs used."""
    for r in (1, 2, 4, 8):
        assert non_overlapping_centre_distance(r, r) == 2 * r + 1


def test_non_overlapping_distance_rejects_bad_radii():
    import pytest
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError):
            non_overlapping_centre_distance(bad, 4)


def test_pixel_and_crown_geometry_have_distinct_audited_footprints():
    inputs = scene()
    pixel = PlacementConfig(
        dose_pp=0.1,
        analysis_buffer_px=8,
        placement_geometry="pixel",
        fixed_tree_height_m=8.0,
    )
    crown = PlacementConfig(
        dose_pp=0.1,
        analysis_buffer_px=8,
        placement_geometry="crown",
        crown_radius_px=4,
        fixed_tree_height_m=8.0,
    )
    _, _, pixel_summary = place_trees(*inputs, "random", pixel)
    _, _, crown_summary = place_trees(*inputs, "random", crown)
    assert pixel_summary["placement_geometry"] == "pixel"
    assert pixel_summary["crown_footprint_pixels"] == 1
    assert crown_summary["placement_geometry"] == "crown"
    assert crown_summary["crown_footprint_pixels"] == 49


def test_pixel_hotspot_spread_grid_thins_while_hotspot_does_not():
    cdsm, building, dem, tsfc, svf = scene()
    cdsm[:] = 0
    building[:] = 0
    tsfc[:] = 300.0
    tsfc[12:16, 20:30, 20:30] = 350.0
    eligibility = np.zeros(cdsm.shape, dtype=bool)
    eligibility[10:70, 10:70] = True
    config = PlacementConfig(
        dose_mode="eqcap",
        eqcap_trees=12,
        analysis_buffer_px=8,
        placement_geometry="pixel",
        fixed_tree_height_m=8.0,
    )
    _, hotspot, hotspot_summary = place_trees(
        cdsm, building, dem, tsfc, svf, "hotspot", config,
        centre_eligibility=eligibility,
    )
    _, spread, spread_summary = place_trees(
        cdsm, building, dem, tsfc, svf, "hotspot_spread", config,
        centre_eligibility=eligibility,
    )
    assert hotspot_summary["spacing_mode"] == "none"
    assert spread_summary["spacing_mode"] == "grid-thin"
    assert not np.array_equal(hotspot[:, :2], spread[:, :2])
    hotspot_extent = np.ptp(hotspot[:, :2], axis=0).sum()
    spread_extent = np.ptp(spread[:, :2], axis=0).sum()
    assert spread_extent > hotspot_extent


def test_absolute_relative_and_eqcap_doses_are_distinct():
    inputs = scene()
    common = dict(analysis_buffer_px=8, placement_geometry="pixel", fixed_tree_height_m=8.0)
    _, _, absolute = place_trees(
        *inputs, "random", PlacementConfig(dose_value=1.0, dose_mode="absolute-pp", **common)
    )
    _, _, relative = place_trees(
        *inputs,
        "random",
        PlacementConfig(dose_value=10.0, dose_mode="relative-canopy", **common),
    )
    _, _, eqcap = place_trees(
        *inputs,
        "random",
        PlacementConfig(dose_mode="eqcap", eqcap_trees=7, **common),
    )
    assert absolute["target_pixels"] == round((80 - 16) ** 2 * 0.01)
    assert relative["target_pixels"] == round(49 * 0.10)
    assert eqcap["dose"]["requested_trees"] == 7
    assert eqcap["tree_centres"] == 7


def test_all_priority_aoi_strategies_execute_and_report_capacity():
    cdsm, building, dem, tsfc, svf = scene()
    landcover = np.full(cdsm.shape, 5, dtype=np.int16)
    landcover[:, 40:] = 1
    config = PlacementConfig(
        dose_pp=0.1,
        analysis_buffer_px=8,
        placement_geometry="pixel",
        fixed_tree_height_m=8.0,
    )
    for strategy in AOI_STRATEGIES:
        _, centres, summary = place_trees(
            cdsm,
            building,
            dem,
            tsfc,
            svf,
            strategy,
            config,
            landcover=landcover,
        )
        assert len(centres)
        assert summary["capacity"]["candidate_centres"] > 0


def test_street_strategy_requires_street_verge_domain():
    with pytest.raises(ValueError, match="street-verge"):
        place_trees(
            *scene(),
            "street_random",
            PlacementConfig(dose_pp=1.0, analysis_buffer_px=8),
        )


def test_all_street_strategies_execute_inside_verge_and_support_eqcap():
    cdsm, building, dem, tsfc, svf = scene()
    eligibility = np.zeros(cdsm.shape, dtype=bool)
    eligibility[10:70, 20:30] = True
    svf[eligibility] = 0.8
    landcover = np.ones(cdsm.shape, dtype=np.int16)
    tmrt = tsfc + 5.0
    config = PlacementConfig(
        dose_mode="eqcap",
        eqcap_trees=5,
        analysis_buffer_px=8,
        placement_geometry="pixel",
        placement_domain="street-verge",
        fixed_tree_height_m=8.0,
    )
    for strategy in STREET_STRATEGIES:
        _, centres, summary = place_trees(
            cdsm,
            building,
            dem,
            tsfc,
            svf,
            strategy,
            config,
            centre_eligibility=eligibility,
            landcover=landcover,
            tmrt=tmrt,
        )
        assert len(centres) == 5
        rr, cc = centres[:, :2].astype(int).T
        assert eligibility[rr, cc].all()
        assert summary["dose"]["target_met"]


def test_eqcap_uses_the_smallest_strategy_capacity():
    cdsm, building, dem, tsfc, svf = scene()
    eligibility = np.zeros(cdsm.shape, dtype=bool)
    eligibility[20:30, 20:30] = True
    svf[eligibility] = 0.8
    config = PlacementConfig(
        dose_mode="eqcap",
        eqcap_trees=1,
        analysis_buffer_px=8,
        placement_geometry="pixel",
        fixed_tree_height_m=8.0,
    )
    budget, capacities = calculate_eqcap_trees(
        cdsm,
        building,
        dem,
        tsfc,
        svf,
        ("random", "hotspot", "adaptive"),
        config,
        centre_eligibility=eligibility,
    )
    assert budget == min(capacities.values())
    assert budget > 0
