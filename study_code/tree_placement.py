"""Versioned AOI-wide and street-verge tree-placement strategy engine."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, uniform_filter


PEAK_HOURS_LOCAL = (12, 13, 14, 15)
AFTERNOON_HOURS_LOCAL = (15, 16, 17)
PLACEMENT_GEOMETRIES = ("pixel", "crown")
DOSE_MODES = ("absolute-pp", "relative-canopy", "eqcap")
SPACING_MODES = ("strategy", "none", "trunk-floor", "nonoverlap", "grid-thin")
AOI_STRATEGIES = (
    "random",
    "high_svf",
    "near_buildings",
    "impervious",
    "expand",
    "hotspot",
    "hotspot_spread",
    "adaptive",
)
STREET_STRATEGIES = (
    "street_random",
    "street_hotspot",
    "street_hotspot_spread",
    "street_adaptive",
    "street_canopy_dilate",
    "street_unshaded_hot",
    "street_shade_greedy",
    "street_cluster_connect",
    "street_building_complement",
)
STRATEGIES = AOI_STRATEGIES + STREET_STRATEGIES
STRATEGY_ALIASES = {
    "svf_gate": "adaptive",
    "street_svf_gate": "street_adaptive",
    "hotspot_add": "hotspot",
    "random_add": "random",
    "high_svf_add": "high_svf",
}


@dataclass(frozen=True)
class PlacementConfig:
    dose_pp: float = 10.0
    analysis_buffer_px: int = 200
    crown_radius_px: int = 4
    building_height_threshold_m: float = 2.0
    fixed_tree_height_m: float | None = None
    seed: int = 42
    placement_domain: str = "everywhere"
    road_scope: str = "all"
    placement_geometry: str = "crown"
    spacing_mode: str = "strategy"
    dose_mode: str = "absolute-pp"
    dose_value: float | None = None
    eqcap_trees: int | None = None
    model_configuration_sha256: str | None = None

    def __post_init__(self) -> None:
        requested = self.requested_dose
        if not np.isfinite(requested) or requested <= 0.0:
            label = "dose_pp" if self.dose_value is None else "dose_value"
            raise ValueError(f"{label} must be finite and greater than zero")
        if self.dose_mode in {"absolute-pp", "relative-canopy"} and requested > 100.0:
            label = "dose_pp" if self.dose_value is None else "dose_value"
            raise ValueError(f"{label} must be no more than 100")
        if (
            not isinstance(self.analysis_buffer_px, int)
            or isinstance(self.analysis_buffer_px, bool)
            or self.analysis_buffer_px < 0
        ):
            raise ValueError("analysis_buffer_px must be a nonnegative integer")
        if (
            not isinstance(self.crown_radius_px, int)
            or isinstance(self.crown_radius_px, bool)
            or self.crown_radius_px < 1
        ):
            raise ValueError("crown_radius_px must be a positive integer")
        if (
            not np.isfinite(self.building_height_threshold_m)
            or self.building_height_threshold_m < 0
        ):
            raise ValueError("building_height_threshold_m must be finite and nonnegative")
        if self.fixed_tree_height_m is not None and (
            not np.isfinite(self.fixed_tree_height_m)
            or not 2.0 <= self.fixed_tree_height_m <= 50.0
        ):
            raise ValueError("fixed_tree_height_m must be between 2 and 50 m")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.placement_domain not in {"everywhere", "street-verge"}:
            raise ValueError(
                "placement_domain must be 'everywhere' or 'street-verge'"
            )
        if self.road_scope not in {"local", "all"}:
            raise ValueError("road_scope must be 'local' or 'all'")
        if self.placement_geometry not in PLACEMENT_GEOMETRIES:
            raise ValueError(f"placement_geometry must be one of {PLACEMENT_GEOMETRIES}")
        if self.spacing_mode not in SPACING_MODES:
            raise ValueError(f"spacing_mode must be one of {SPACING_MODES}")
        if self.dose_mode not in DOSE_MODES:
            raise ValueError(f"dose_mode must be one of {DOSE_MODES}")
        if self.dose_mode == "eqcap" and (
            not isinstance(self.eqcap_trees, int)
            or isinstance(self.eqcap_trees, bool)
            or self.eqcap_trees < 1
        ):
            raise ValueError("eqcap mode requires a positive eqcap_trees value")

    @property
    def requested_dose(self) -> float:
        return float(self.dose_pp if self.dose_value is None else self.dose_value)


def crown_offsets(radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(radius_px, int) or isinstance(radius_px, bool) or radius_px < 1:
        raise ValueError("radius_px must be a positive integer")
    axis = np.arange(-radius_px, radius_px + 1, dtype=np.int16)
    rows, cols = np.meshgrid(axis, axis, indexing="ij")
    keep = rows * rows + cols * cols <= radius_px * radius_px
    return rows[keep], cols[keep]


def non_overlapping_centre_distance(radius_a_px: int, radius_b_px: int) -> int:
    """Smallest integer centre distance at which two rasterised crowns share no pixel.

    A crown is the rasterised disk ``row^2 + col^2 <= radius^2``. Two such disks
    still share at least one pixel when their centres are ``radius_a + radius_b``
    apart, and are disjoint from ``radius_a + radius_b + 1`` onward, including the
    diagonal worst case. Deriving the separation from the radii keeps the rule
    correct if crowns of differing size are ever mixed; for the uniform reference
    crown it reduces to ``2 * radius + 1``.
    """
    for name, value in (("radius_a_px", radius_a_px), ("radius_b_px", radius_b_px)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return radius_a_px + radius_b_px + 1


def analysis_mask(shape: tuple[int, int], buffer_px: int) -> np.ndarray:
    if buffer_px < 0 or 2 * buffer_px >= min(shape):
        raise ValueError("analysis buffer leaves no interior domain")
    mask = np.zeros(shape, dtype=bool)
    if buffer_px == 0:
        mask[:] = True
    else:
        mask[buffer_px:-buffer_px, buffer_px:-buffer_px] = True
    return mask


def _read_raster(path: Path) -> tuple[np.ndarray, dict]:
    import rasterio

    with rasterio.open(path) as src:
        # float32 already holds every value these rasters carry and keeps a
        # 24-band Tsfc cube at ~190 MB instead of ~380 MB; NaN needs a float
        # dtype, so integer layers (landcover) widen rather than truncate.
        data = src.read(masked=True).astype(np.float32).filled(np.nan)
        return data, src.profile.copy()


def _base_output_pair(city_dir: Path) -> tuple[Path, Path]:
    root = city_dir / "output" / "base" / "output_folder"
    tsfc_paths = sorted(root.rglob("Tsfc_*.tif"))
    svf_paths = sorted(root.rglob("SVF_*.tif"))
    if len(tsfc_paths) != 1 or len(svf_paths) != 1:
        raise RuntimeError(
            "tree placement requires one base-model tile; "
            f"found {len(tsfc_paths)} Tsfc and {len(svf_paths)} SVF rasters"
        )
    tsfc_key = tsfc_paths[0].stem.removeprefix("Tsfc_")
    svf_key = svf_paths[0].stem.removeprefix("SVF_")
    if tsfc_key != svf_key:
        raise RuntimeError(
            f"base Tsfc and SVF tile keys do not match: {tsfc_key!r}, {svf_key!r}"
        )
    return tsfc_paths[0], svf_paths[0]


def _check_grids(named: dict[str, tuple[np.ndarray, dict]]) -> None:
    shapes = {name: value[0].shape[-2:] for name, value in named.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"input raster shapes differ: {shapes}")
    transforms = {name: value[1]["transform"] for name, value in named.items()}
    if len(set(transforms.values())) != 1:
        raise ValueError("input raster transforms differ")
    missing_crs = [name for name, value in named.items() if value[1].get("crs") is None]
    if missing_crs:
        raise ValueError(f"input rasters have no coordinate system: {missing_crs}")
    crs = {name: str(value[1]["crs"]) for name, value in named.items()}
    if len(set(crs.values())) != 1:
        raise ValueError(f"input raster coordinate systems differ: {crs}")
    transform = next(iter(transforms.values()))
    if not np.isclose(abs(float(transform.a)), 1.0) or not np.isclose(
        abs(float(transform.e)), 1.0
    ):
        raise ValueError("placement requires a 1 m raster grid")


def _valid_centres(
    cdsm: np.ndarray,
    building_dsm: np.ndarray,
    dem: np.ndarray,
    config: PlacementConfig,
    centre_eligibility: np.ndarray | None = None,
    additional_buildings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    interior = analysis_mask(cdsm.shape, config.analysis_buffer_px)
    finite = np.isfinite(cdsm) & np.isfinite(building_dsm) & np.isfinite(dem)
    buildings = finite & ((building_dsm - dem) > config.building_height_threshold_m)
    if additional_buildings is not None:
        if additional_buildings.shape != cdsm.shape:
            raise ValueError("additional building mask shape differs from placement grid")
        buildings |= additional_buildings.astype(bool)
    plantable = interior & finite & (cdsm <= 0) & ~buildings
    if config.placement_geometry == "crown":
        dr, dc = crown_offsets(config.crown_radius_px)
        size = 2 * config.crown_radius_px + 1
        footprint = np.zeros((size, size), dtype=bool)
        footprint[dr + config.crown_radius_px, dc + config.crown_radius_px] = True
        candidates = binary_erosion(plantable, structure=footprint, border_value=0)
    else:
        candidates = plantable.copy()
    if centre_eligibility is not None:
        if centre_eligibility.shape != cdsm.shape:
            raise ValueError("centre eligibility mask shape differs from placement grid")
        candidates &= centre_eligibility.astype(bool)
    return candidates, interior


def _peak_surface_temperature(tsfc: np.ndarray) -> np.ndarray:
    """Return the common 12:00--15:00 local placement score.

    Array index equals model hour, so these are zero-based indices 12--15 and
    GeoTIFF bands 13--16.  Publication runs must provide the complete window;
    silently substituting another hour would change the placement algorithm.
    """
    if tsfc.shape[0] <= max(PEAK_HOURS_LOCAL):
        raise ValueError(
            "Tsfc must contain model hours 12--15 "
            f"(at least 16 bands); found {tsfc.shape[0]}"
        )
    return np.mean(tsfc[list(PEAK_HOURS_LOCAL)], axis=0)


def canonical_strategy(strategy: str) -> str:
    value = strategy.strip().lower().replace("-", "_")
    value = STRATEGY_ALIASES.get(value, value)
    if value not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    return value


def _strategy_base(strategy: str) -> str:
    return strategy.removeprefix("street_")


def _finite_rank_values(
    rows: np.ndarray,
    cols: np.ndarray,
    *values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Remove candidates lacking any quantity required by a strategy."""
    sampled = tuple(value[rows, cols] for value in values)
    keep = np.ones(len(rows), dtype=bool)
    for value in sampled:
        keep &= np.isfinite(value)
    return rows[keep], cols[keep], tuple(value[keep] for value in sampled)


def _rank_strategy(
    strategy: str,
    candidates: np.ndarray,
    cdsm: np.ndarray,
    building_dsm: np.ndarray,
    dem: np.ndarray,
    tsfc: np.ndarray,
    svf: np.ndarray,
    config: PlacementConfig,
    *,
    landcover: np.ndarray | None,
    tmrt: np.ndarray | None,
    minimum_centres: int,
    additional_buildings: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    base = _strategy_base(strategy)
    rows, cols = np.where(candidates)
    if not len(rows):
        return np.empty((0, 2), dtype=int), {}
    rng = np.random.default_rng(config.seed)
    jitter = rng.uniform(0.0, 1.0e-6, len(rows))
    peak = _peak_surface_temperature(tsfc)
    metadata: dict = {}

    if base == "random":
        order = rng.permutation(len(rows))
    elif base in {"hotspot", "hotspot_spread", "cluster_connect"}:
        rows, cols, (score,) = _finite_rank_values(rows, cols, peak)
        jitter = rng.uniform(0.0, 1.0e-6, len(rows))
        if base == "cluster_connect":
            seed_count = max(1, min(6, minimum_centres // 400 or 1))
            hot = np.argsort(-(score + jitter))[:seed_count]
            seed_mask = np.zeros(candidates.shape, dtype=bool)
            seed_mask[rows[hot], cols[hot]] = True
            distance = distance_transform_edt(~seed_mask)
            order = np.lexsort((-(score + jitter), distance[rows, cols]))
        else:
            order = np.argsort(-(score + jitter))
    elif base == "adaptive":
        threshold = 0.6
        valid = candidates & np.isfinite(svf) & np.isfinite(peak)
        gated = valid & (svf > threshold)
        if int(gated.sum()) < minimum_centres:
            threshold = 0.4
            gated = valid & (svf > threshold)
        rows, cols = np.where(gated)
        score = peak[rows, cols]
        order = np.argsort(-score)
        metadata["svf_threshold_used"] = threshold
        metadata["gated_candidate_centres"] = int(gated.sum())
    elif base == "high_svf":
        rows, cols, (score,) = _finite_rank_values(rows, cols, svf)
        jitter = rng.uniform(0.0, 1.0e-6, len(rows))
        order = np.argsort(-(score + jitter))
    elif base == "near_buildings":
        buildings = np.isfinite(building_dsm) & (
            (building_dsm - dem) > config.building_height_threshold_m
        )
        if additional_buildings is not None:
            # Same building definition the candidate screen used, so "near a
            # building" cannot mean "near a DSM blob that OSM calls something
            # else" -- or miss an OSM building the DSM never resolved.
            buildings = buildings | additional_buildings.astype(bool)
        if buildings.any():
            score = distance_transform_edt(~buildings)[rows, cols]
            order = np.argsort(score + jitter)
        else:
            order = rng.permutation(len(rows))
            metadata["fallback"] = "random_no_buildings"
    elif base == "impervious":
        if landcover is None:
            raise ValueError("impervious strategy requires landcover")
        if landcover.shape != candidates.shape:
            raise ValueError("landcover shape differs from placement grid")
        rows, cols, (landcover_values,) = _finite_rank_values(rows, cols, landcover)
        jitter = rng.uniform(0.0, 1.0e-6, len(rows))
        asphalt = landcover_values == 1
        order = np.argsort(-(asphalt.astype(float) + jitter))
        metadata["impervious_candidate_centres"] = int(asphalt.sum())
    elif base in {"expand", "canopy_dilate"}:
        existing = (cdsm > 0) & np.isfinite(cdsm)
        if existing.any():
            score = distance_transform_edt(~existing)[rows, cols]
            order = np.argsort(score + jitter)
        else:
            order = rng.permutation(len(rows))
            metadata["fallback"] = "random_no_existing_canopy"
    elif base == "unshaded_hot":
        rows, cols, (peak_values,) = _finite_rank_values(rows, cols, peak)
        jitter = rng.uniform(0.0, 1.0e-6, len(rows))
        local_canopy = uniform_filter((cdsm > 0).astype(np.float32), size=15)
        score = peak_values * (
            1.0 - local_canopy[rows, cols]
        )
        order = np.argsort(-(score + jitter))
    elif base == "shade_greedy":
        rows, cols, (peak_values,) = _finite_rank_values(rows, cols, peak)
        jitter = rng.uniform(0.0, 1.0e-6, len(rows))
        ground = (cdsm <= 0) & np.isfinite(cdsm)
        finite_peak = peak[np.isfinite(peak)]
        if not finite_peak.size:
            return np.empty((0, 2), dtype=int), metadata
        shifted = np.where(
            ground & np.isfinite(peak), np.maximum(peak - finite_peak.min(), 0.0), 0.0
        )
        score_grid = uniform_filter(shifted.astype(np.float32), size=13)
        order = np.argsort(-(score_grid[rows, cols] + jitter))
    elif base == "building_complement":
        if tmrt is None or tmrt.ndim != 3 or tmrt.shape[0] <= max(AFTERNOON_HOURS_LOCAL):
            raise ValueError("street_building_complement requires TMRT hours 15--17")
        score_grid = np.mean(tmrt[list(AFTERNOON_HOURS_LOCAL)], axis=0)
        rows, cols, (score,) = _finite_rank_values(rows, cols, score_grid)
        jitter = rng.uniform(0.0, 1.0e-6, len(rows))
        order = np.argsort(-(score + jitter))
        metadata["ranking_hours_local"] = list(AFTERNOON_HOURS_LOCAL)
    else:  # guarded by canonical_strategy
        raise AssertionError(base)
    ranked = np.column_stack((rows, cols))[order]
    return ranked, metadata


def _strategy_is_dispersed(strategy: str) -> bool:
    base = _strategy_base(strategy)
    return base in {
        "hotspot_spread",
        "adaptive",
        "unshaded_hot",
        "shade_greedy",
        "building_complement",
    } or (base == "random" and not strategy.startswith("street_"))


def _resolved_spacing_mode(strategy: str, config: PlacementConfig) -> str:
    if config.spacing_mode != "strategy":
        return config.spacing_mode
    if _strategy_is_dispersed(strategy):
        return "grid-thin" if config.placement_geometry == "pixel" else "nonoverlap"
    return "none" if config.placement_geometry == "pixel" else "trunk-floor"


def _grid_thin_ranked(ranked: np.ndarray, count: int) -> np.ndarray:
    """Reproduce the archived coarsest-grid, one-candidate-per-cell thinning."""
    if count <= 0 or len(ranked) <= count:
        return ranked
    rows, cols = ranked[:, 0], ranked[:, 1]
    row0, col0 = int(rows.min()), int(cols.min())
    low = 1.0
    high = float(max(rows.max() - rows.min(), cols.max() - cols.min(), 1))
    best: list[int] | None = None
    for _ in range(28):
        cell = 0.5 * (low + high)
        inverse = 1.0 / cell
        seen: set[tuple[int, int]] = set()
        accepted: list[int] = []
        for index, (row, col) in enumerate(ranked):
            key = (int((row - row0) * inverse), int((col - col0) * inverse))
            if key in seen:
                continue
            seen.add(key)
            accepted.append(index)
            if len(accepted) > count:
                break
        if len(accepted) >= count:
            best = accepted[:count]
            low = cell
        else:
            high = cell
    if best is None:
        return ranked[:count]
    return ranked[np.asarray(best, dtype=int)]


def _assign_heights(
    cdsm: np.ndarray,
    centres: list[tuple[int, int]],
    fixed_height_m: float | None,
) -> np.ndarray:
    if not centres:
        return np.empty(0, dtype=np.float32)
    if fixed_height_m is not None:
        if not 2.0 <= fixed_height_m <= 50.0:
            raise ValueError("fixed tree height must be between 2 and 50 m")
        return np.full(len(centres), fixed_height_m, dtype=np.float32)
    canopy = cdsm[(cdsm > 0) & np.isfinite(cdsm)]
    fallback = float(np.median(canopy)) if canopy.size else 8.0
    heights = []
    for row, col in centres:
        value = None
        for radius in (5, 10, 20):
            patch = cdsm[
                max(0, row - radius) : row + radius + 1,
                max(0, col - radius) : col + radius + 1,
            ]
            nearby = patch[(patch > 0) & np.isfinite(patch)]
            if nearby.size:
                value = float(nearby.mean())
                break
        heights.append(fallback if value is None else value)
    return np.clip(np.asarray(heights, dtype=np.float32), 2.0, 50.0)


def place_trees(
    cdsm: np.ndarray,
    building_dsm: np.ndarray,
    dem: np.ndarray,
    tsfc: np.ndarray,
    svf: np.ndarray,
    strategy: str,
    config: PlacementConfig,
    *,
    centre_eligibility: np.ndarray | None = None,
    additional_buildings: np.ndarray | None = None,
    eligibility_audit: dict | None = None,
    landcover: np.ndarray | None = None,
    tmrt: np.ndarray | None = None,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a modified CDSM, centre table, and audit summary.

    ``allow_empty`` is for capacity probing only.  A normal run that places no
    tree is a failed experiment, not an empty scenario, so it raises.
    """
    arrays_2d = {
        "CDSM": cdsm,
        "building DSM": building_dsm,
        "DEM": dem,
        "SVF": svf,
    }
    invalid_dimensions = [name for name, value in arrays_2d.items() if value.ndim != 2]
    if invalid_dimensions:
        raise ValueError(f"placement inputs must be two-dimensional: {invalid_dimensions}")
    if tsfc.ndim != 3 or tsfc.shape[0] < 1:
        raise ValueError("Tsfc must contain at least one two-dimensional band")
    shapes = {name: value.shape for name, value in arrays_2d.items()}
    shapes["Tsfc"] = tsfc.shape[-2:]
    if len(set(shapes.values())) != 1:
        raise ValueError(f"placement input shapes differ: {shapes}")
    requested_strategy = strategy
    strategy = canonical_strategy(strategy)
    if strategy.startswith("street_") and config.placement_domain != "street-verge":
        raise ValueError(f"{strategy} requires placement_domain='street-verge'")
    candidates, interior = _valid_centres(
        cdsm,
        building_dsm,
        dem,
        config,
        centre_eligibility=centre_eligibility,
        additional_buildings=additional_buildings,
    )
    baseline_canopy_pixels = int(((cdsm > 0) & interior).sum())
    if config.placement_geometry == "crown":
        dr, dc = crown_offsets(config.crown_radius_px)
    else:
        dr = np.asarray([0], dtype=np.int16)
        dc = np.asarray([0], dtype=np.int16)
    requested_dose = config.requested_dose
    target_trees = config.eqcap_trees if config.dose_mode == "eqcap" else None
    if config.dose_mode == "absolute-pp":
        target_pixels = int(round(interior.sum() * requested_dose / 100.0))
    elif config.dose_mode == "relative-canopy":
        target_pixels = int(round(baseline_canopy_pixels * requested_dose / 100.0))
    else:
        target_pixels = int(target_trees * len(dr))
    if target_pixels < 1:
        raise ValueError("requested dose rounds to fewer than one canopy pixel")
    minimum_centres = target_trees or int(np.ceil(target_pixels / len(dr)))
    ranked, ranking_metadata = _rank_strategy(
        strategy,
        candidates,
        cdsm,
        building_dsm,
        dem,
        tsfc,
        svf,
        config,
        landcover=landcover,
        tmrt=tmrt,
        minimum_centres=minimum_centres,
        additional_buildings=additional_buildings,
    )
    rankable_candidate_count = len(ranked)
    spacing_mode = _resolved_spacing_mode(strategy, config)
    if spacing_mode == "grid-thin":
        if config.placement_geometry != "pixel":
            raise ValueError("grid-thin spacing is defined only for pixel geometry")
        selection_count = int(target_trees) if config.dose_mode == "eqcap" else target_pixels
        ranked = _grid_thin_ranked(ranked, selection_count)
        minimum_distance = 1
    elif spacing_mode == "none":
        minimum_distance = 1
    elif spacing_mode == "nonoverlap":
        minimum_distance = non_overlapping_centre_distance(
            config.crown_radius_px, config.crown_radius_px
        )
    elif spacing_mode == "trunk-floor":
        minimum_distance = config.crown_radius_px
    else:
        raise AssertionError(spacing_mode)

    added = np.zeros(cdsm.shape, dtype=bool)
    accepted: list[tuple[int, int]] = []
    minimum_distance_sq = minimum_distance * minimum_distance
    bucket_size = max(float(minimum_distance), 1.0)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def centre_is_spaced(row: int, col: int) -> bool:
        key = (int(row // bucket_size), int(col // bucket_size))
        for bucket_row in range(key[0] - 1, key[0] + 2):
            for bucket_col in range(key[1] - 1, key[1] + 2):
                for old_row, old_col in buckets.get((bucket_row, bucket_col), []):
                    if (row - old_row) ** 2 + (col - old_col) ** 2 < minimum_distance_sq:
                        return False
        return True

    for row_value, col_value in ranked:
        row, col = int(row_value), int(col_value)
        if not centre_is_spaced(row, col):
            continue
        rr, cc = row + dr, col + dc
        accepted.append((row, col))
        key = (int(row // bucket_size), int(col // bucket_size))
        buckets.setdefault(key, []).append((row, col))
        added[rr, cc] = True
        if config.dose_mode == "eqcap":
            reached = len(accepted) >= int(target_trees)
        else:
            reached = int(added.sum()) >= target_pixels
        if reached:
            break

    if not accepted and not allow_empty:
        raise RuntimeError(
            f"{strategy} placed no tree: {int(candidates.sum())} physical candidate "
            f"centres, {rankable_candidate_count} rankable after scoring. "
            "Check the placement domain and eligibility masks."
        )

    heights = _assign_heights(cdsm, accepted, config.fixed_tree_height_m)
    modified = cdsm.copy()
    for (row, col), height in zip(accepted, heights):
        rr, cc = row + dr, col + dc
        modified[rr, cc] = np.maximum(modified[rr, cc], height)
    centres = np.asarray(
        [(row, col, float(height)) for (row, col), height in zip(accepted, heights)],
        dtype=float,
    ).reshape((-1, 3))
    pixels_added = int(added.sum())
    realized_pp = float(100.0 * pixels_added / interior.sum())
    realized_relative = (
        float(100.0 * pixels_added / baseline_canopy_pixels)
        if baseline_canopy_pixels
        else None
    )
    if config.dose_mode == "eqcap":
        target_met = len(accepted) >= int(target_trees)
    else:
        target_met = pixels_added >= target_pixels
    summary = {
        "strategy": strategy,
        "requested_strategy": requested_strategy,
        "configuration": asdict(config),
        "placement_geometry": config.placement_geometry,
        "spacing_mode": spacing_mode,
        "dose": {
            "mode": config.dose_mode,
            "requested_value": requested_dose,
            "requested_pixels": target_pixels,
            "requested_trees": target_trees,
            "realized_pixels": pixels_added,
            "realized_trees": len(accepted),
            "realized_added_pp": realized_pp,
            "realized_relative_canopy_percent": realized_relative,
            "target_met": target_met,
        },
        "candidate_centres": int(candidates.sum()),
        "rankable_candidate_centres": int(rankable_candidate_count),
        "tree_centres": len(accepted),
        "baseline_canopy_pixels": baseline_canopy_pixels,
        "target_pixels": target_pixels,
        "pixels_added": pixels_added,
        "realized_dose_pp": realized_pp,
        "minimum_centre_distance_px": minimum_distance,
        "adaptive_svf_threshold": ranking_metadata.get("svf_threshold_used"),
        "crown_footprint_pixels": int(len(dr)),
        "height_assignment": (
            "fixed" if config.fixed_tree_height_m is not None else "nearby_canopy"
        ),
        "height_mean_m": float(heights.mean()) if heights.size else None,
        "height_min_m": float(heights.min()) if heights.size else None,
        "height_max_m": float(heights.max()) if heights.size else None,
        "constrained": not target_met,
        "capacity": {
            "candidate_centres": int(rankable_candidate_count),
            "physical_candidate_centres": int(candidates.sum()),
            "rankable_candidate_centres": int(rankable_candidate_count),
            "accepted_before_exhaustion": len(accepted),
            "binding": not target_met,
        },
        "ranking": ranking_metadata,
        "ranking_hours_local": list(PEAK_HOURS_LOCAL),
        "eligibility": eligibility_audit,
        "model_configuration_sha256": config.model_configuration_sha256,
    }
    return modified, centres, summary


def calculate_eqcap_trees(
    cdsm: np.ndarray,
    building_dsm: np.ndarray,
    dem: np.ndarray,
    tsfc: np.ndarray,
    svf: np.ndarray,
    strategies: tuple[str, ...],
    config: PlacementConfig,
    **kwargs,
) -> tuple[int, dict[str, int]]:
    """Return the equal tree budget and per-strategy physical capacities.

    Capacity is measured with the *same* budget the scenario run will use, then
    re-measured until it stops shrinking.  This matters because two strategies
    read the requested budget while ranking, not only while accepting: the
    ``adaptive`` SVF gate relaxes 0.6 -> 0.4 when the gated pool looks smaller
    than the budget, and ``cluster_connect`` scales its seed count by it.  A
    single probe at an artificial budget would therefore rank over a different
    pool than the run, and could report a budget the run cannot actually place.
    """
    if not strategies:
        raise ValueError("eqcap requires at least one strategy")

    def measure(budget: int) -> dict[str, int]:
        probe = replace(config, dose_mode="eqcap", eqcap_trees=budget)
        measured = {}
        for strategy in strategies:
            _, centres, _ = place_trees(
                cdsm,
                building_dsm,
                dem,
                tsfc,
                svf,
                strategy,
                probe,
                allow_empty=True,
                **kwargs,
            )
            measured[canonical_strategy(strategy)] = int(len(centres))
        return measured

    capacities = measure(int(np.prod(cdsm.shape)))
    binding = min(capacities.values())
    for _ in range(4):
        if binding < 1:
            break
        capacities = measure(binding)
        settled = min(capacities.values())
        if settled >= binding:
            break
        binding = settled

    zero = [strategy for strategy, value in capacities.items() if value < 1]
    if zero:
        raise RuntimeError(
            f"eqcap is undefined because a strategy has zero capacity: {capacities}"
        )
    return min(capacities.values()), capacities


def run_city(city_dir: Path, output_dir: Path, strategy: str, config: PlacementConfig) -> dict:
    import rasterio

    from .experiment_contract import canonical_hash, input_provenance, source_code_hash
    from .geospatial_preprocessing.osm_eligibility import build_osm_placement_masks
    from .run_study import MODEL_OPTIONS

    if config.model_configuration_sha256 is None:
        config = replace(
            config,
            model_configuration_sha256=canonical_hash(MODEL_OPTIONS),
        )

    tsfc_path, svf_path = _base_output_pair(city_dir)
    tile_key = tsfc_path.stem.removeprefix("Tsfc_")
    tmrt_path = tsfc_path.with_name(f"TMRT_{tile_key}.tif")
    inputs = {
        "CDSM": _read_raster(city_dir / "CDSM.tif"),
        "Building_DSM": _read_raster(city_dir / "Building_DSM.tif"),
        "DEM": _read_raster(city_dir / "DEM.tif"),
        "Tsfc": _read_raster(tsfc_path),
        "SVF": _read_raster(svf_path),
    }
    landcover_path = city_dir / "landcover.tif"
    if not landcover_path.is_file():
        raise FileNotFoundError(f"placement landcover not found: {landcover_path}")
    inputs["landcover"] = _read_raster(landcover_path)
    if tmrt_path.is_file():
        inputs["TMRT"] = _read_raster(tmrt_path)
    _check_grids(inputs)
    for name in ("CDSM", "Building_DSM", "DEM", "SVF", "landcover"):
        if inputs[name][0].shape[0] != 1:
            raise ValueError(f"{name} must be a single-band raster")
    if inputs["Tsfc"][0].shape[0] < 1:
        raise ValueError("Tsfc must contain at least one band")
    osm_masks = build_osm_placement_masks(
        city_dir / "osm",
        city_dir / "CDSM.tif",
        config.placement_domain,
        config.road_scope,
    )
    tmrt = inputs["TMRT"][0] if "TMRT" in inputs else None
    modified, centres, summary = place_trees(
        inputs["CDSM"][0][0],
        inputs["Building_DSM"][0][0],
        inputs["DEM"][0][0],
        inputs["Tsfc"][0],
        inputs["SVF"][0][0],
        strategy,
        config,
        centre_eligibility=osm_masks.centre_eligible,
        additional_buildings=osm_masks.osm_buildings,
        eligibility_audit=osm_masks.audit,
        landcover=inputs["landcover"][0][0],
        tmrt=tmrt,
    )
    output_strategy = summary["strategy"]
    summary["source_code_sha256"] = source_code_hash()
    provenance_paths = [
        city_dir / "CDSM.tif",
        city_dir / "Building_DSM.tif",
        city_dir / "DEM.tif",
        landcover_path,
        tsfc_path,
        svf_path,
        *sorted((city_dir / "osm").glob("*.geojson")),
    ]
    if tmrt_path.is_file():
        provenance_paths.append(tmrt_path)
    summary["input_provenance"] = input_provenance(provenance_paths, root=city_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = inputs["CDSM"][1]
    profile.update(count=1, dtype="float32")
    nodata = profile.get("nodata")
    if nodata is None or not np.isfinite(nodata):
        profile.update(nodata=np.nan)
        output = modified.astype("float32")
    else:
        output = np.where(np.isfinite(modified), modified, nodata).astype("float32")
    with rasterio.open(output_dir / f"{output_strategy}_CDSM.tif", "w", **profile) as dst:
        dst.write(output, 1)
    transform = profile["transform"]
    centres_path = output_dir / "placement_centres.csv"
    with centres_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["tree_id", "row", "col", "x", "y", "height_m", "radius_m", "geometry"]
        )
        for tree_id, (row, col, height) in enumerate(centres, 1):
            x, y = rasterio.transform.xy(transform, int(row), int(col), offset="center")
            radius = config.crown_radius_px if config.placement_geometry == "crown" else 0
            writer.writerow(
                [tree_id, int(row), int(col), x, y, height, radius, config.placement_geometry]
            )
    (output_dir / "placement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def calculate_city_eqcap(
    city_dir: Path,
    strategies: tuple[str, ...],
    config: PlacementConfig,
) -> tuple[int, dict[str, int]]:
    """Calculate a binding equal-tree budget using the exact placement engine."""
    from .geospatial_preprocessing.osm_eligibility import build_osm_placement_masks

    tsfc_path, svf_path = _base_output_pair(city_dir)
    tile_key = tsfc_path.stem.removeprefix("Tsfc_")
    tmrt_path = tsfc_path.with_name(f"TMRT_{tile_key}.tif")
    inputs = {
        "CDSM": _read_raster(city_dir / "CDSM.tif"),
        "Building_DSM": _read_raster(city_dir / "Building_DSM.tif"),
        "DEM": _read_raster(city_dir / "DEM.tif"),
        "Tsfc": _read_raster(tsfc_path),
        "SVF": _read_raster(svf_path),
        "landcover": _read_raster(city_dir / "landcover.tif"),
    }
    if tmrt_path.is_file():
        inputs["TMRT"] = _read_raster(tmrt_path)
    _check_grids(inputs)
    cdsm = inputs["CDSM"][0][0]
    building = inputs["Building_DSM"][0][0]
    dem = inputs["DEM"][0][0]
    tsfc = inputs["Tsfc"][0]
    svf = inputs["SVF"][0][0]
    landcover = inputs["landcover"][0][0]
    tmrt = inputs["TMRT"][0] if "TMRT" in inputs else None
    masks = build_osm_placement_masks(
        city_dir / "osm",
        city_dir / "CDSM.tif",
        config.placement_domain,
        config.road_scope,
    )
    return calculate_eqcap_trees(
        cdsm,
        building,
        dem,
        tsfc,
        svf,
        strategies,
        config,
        centre_eligibility=masks.centre_eligible,
        additional_buildings=masks.osm_buildings,
        eligibility_audit=masks.audit,
        landcover=landcover,
        tmrt=tmrt,
    )


def main() -> None:
    from .experiment_contract import canonical_hash
    from .run_study import MODEL_OPTIONS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=(*STRATEGIES, *STRATEGY_ALIASES), required=True)
    parser.add_argument("--dose-pp", type=float, default=10.0)
    parser.add_argument("--analysis-buffer-px", type=int, default=200)
    parser.add_argument("--fixed-tree-height-m", type=float)
    parser.add_argument(
        "--placement-domain",
        choices=("everywhere", "street-verge"),
        default="everywhere",
    )
    parser.add_argument("--road-scope", choices=("local", "all"), default="all")
    parser.add_argument("--placement-geometry", choices=PLACEMENT_GEOMETRIES, default="crown")
    parser.add_argument("--spacing-mode", choices=SPACING_MODES, default="strategy")
    parser.add_argument("--dose-mode", choices=DOSE_MODES, default="absolute-pp")
    parser.add_argument("--dose-value", type=float)
    parser.add_argument("--eqcap-trees", type=int)
    args = parser.parse_args()
    config = PlacementConfig(
        dose_pp=args.dose_pp,
        analysis_buffer_px=args.analysis_buffer_px,
        fixed_tree_height_m=args.fixed_tree_height_m,
        placement_domain=args.placement_domain,
        road_scope=args.road_scope,
        placement_geometry=args.placement_geometry,
        spacing_mode=args.spacing_mode,
        dose_mode=args.dose_mode,
        dose_value=args.dose_value,
        eqcap_trees=args.eqcap_trees,
        model_configuration_sha256=canonical_hash(MODEL_OPTIONS),
    )
    print(json.dumps(run_city(args.city_dir, args.output_dir, args.strategy, config), indent=2))


if __name__ == "__main__":
    main()
