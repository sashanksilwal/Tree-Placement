"""Run a versioned baseline and selected UTherm placement strategies."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

from utherm import thermal_comfort

from .experiment_contract import (
    EXPERIMENT_SCHEMA_VERSION,
    build_manifest,
    canonical_hash,
    validate_resume,
    write_manifest,
)
from .tree_placement import (
    DOSE_MODES,
    PEAK_HOURS_LOCAL,
    PLACEMENT_GEOMETRIES,
    SPACING_MODES,
    STRATEGIES,
    PlacementConfig,
    calculate_city_eqcap,
    canonical_strategy,
    run_city,
)


MODEL_OPTIONS = {
    "tile_size": 1400,
    "overlap": 20,
    "use_own_met": True,
    "save_tmrt": True,
    "save_svf": True,
    "save_tsfc": True,
    "save_tair": True,
    "use_energy_balance": True,
    "use_canopy_eb": True,
    "canopy_type": "deciduous",
    "canopy_lai": 3.0,
    "z0": 0.01,
    "ground_material_type": "asphalt",
    "use_wind_field": True,
    "save_wind_speed": True,
}


def _merge_masked(sources):
    mosaic, transform = merge(sources, nodata=np.nan, dtype="float64")
    return np.ma.masked_invalid(mosaic), transform


def _prepare_run_directory(city_dir: Path, run_dir: Path, tree_raster: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "Building_DSM.tif": city_dir / "Building_DSM.tif",
        "DEM.tif": city_dir / "DEM.tif",
        "Trees.tif": tree_raster,
    }
    landcover = city_dir / "landcover.tif"
    if landcover.exists():
        sources["landcover.tif"] = landcover
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"model input not found: {source}")
        destination = run_dir / name
        if destination.exists() or destination.is_symlink():
            try:
                matches = destination.samefile(source)
            except FileNotFoundError:
                matches = False
            if not matches:
                raise RuntimeError(
                    f"existing model input {destination} does not point to {source}"
                )
            continue
        destination.symlink_to(source.resolve())


def _validate_model_output_rasters(run_dir: Path) -> None:
    """Require readable, georeferenced outputs with the full analysis window."""
    output_root = run_dir / "output_folder"
    for variable in ("TMRT", "Tsfc", "UTCI", "Tair"):
        paths = sorted(output_root.rglob(f"{variable}_*.tif"))
        if not paths:
            raise RuntimeError(f"missing {variable} output rasters: {run_dir}")
        family_has_finite_analysis_data = False
        for path in paths:
            try:
                with rasterio.open(path) as source:
                    if source.count <= max(PEAK_HOURS_LOCAL):
                        raise RuntimeError(
                            f"{path} lacks model hours 12--15; found {source.count} bands"
                        )
                    if source.crs is None or source.width < 1 or source.height < 1:
                        raise RuntimeError(f"invalid output raster grid: {path}")
                    values = source.read(
                        [hour + 1 for hour in PEAK_HOURS_LOCAL], masked=True
                    )
                    family_has_finite_analysis_data |= bool(
                        np.isfinite(values.filled(np.nan)).any()
                    )
            except RuntimeError:
                raise
            except Exception as error:
                raise RuntimeError(f"unreadable output raster {path}: {error}") from error
        if not family_has_finite_analysis_data:
            raise RuntimeError(
                f"{variable} outputs contain no finite data in hours 12--15: {run_dir}"
            )


def _run_model(
    run_dir: Path,
    met_file: Path,
    simulation_date: str,
    spinup_days: int,
    manifest: dict,
) -> str:
    required = ("TMRT_*.tif", "Tsfc_*.tif", "UTCI_*.tif", "Tair_*.tif")
    if validate_resume(run_dir, manifest, required):
        _validate_model_output_rasters(run_dir)
        return "resumed"
    if (run_dir / "output_folder").exists():
        raise RuntimeError(f"incomplete output directory must be archived before rerun: {run_dir}")
    options = dict(MODEL_OPTIONS)
    if (run_dir / "landcover.tif").exists():
        options["landcover_filename"] = "landcover.tif"
    write_manifest(run_dir / "run_manifest.json", manifest)
    thermal_comfort(
        base_path=str(run_dir),
        selected_date_str=simulation_date,
        own_met_file=str(met_file),
        spinup_days=spinup_days,
        **options,
    )
    if not validate_resume(run_dir, manifest, required):
        raise RuntimeError(f"model completed without creating outputs: {run_dir}")
    _validate_model_output_rasters(run_dir)
    return "completed"


def _delta_summary(
    base_dir: Path,
    scenario_dir: Path,
    buffer_px: int,
    street_analysis_mask: np.ndarray | None = None,
) -> dict:
    if not isinstance(buffer_px, int) or isinstance(buffer_px, bool) or buffer_px < 0:
        raise ValueError("buffer_px must be a nonnegative integer")
    output = {}
    for variable in ("TMRT", "Tsfc", "UTCI", "Tair"):
        base_root = base_dir / "output_folder"
        scenario_root = scenario_dir / "output_folder"
        base_files = {
            path.relative_to(base_root): path
            for path in base_root.rglob(f"{variable}_*.tif")
        }
        scenario_files = {
            path.relative_to(scenario_root): path
            for path in scenario_root.rglob(f"{variable}_*.tif")
        }
        if not base_files or not scenario_files:
            raise FileNotFoundError(f"missing {variable} output rasters")
        if set(base_files) != set(scenario_files):
            missing = sorted(str(path) for path in set(base_files) - set(scenario_files))
            extra = sorted(str(path) for path in set(scenario_files) - set(base_files))
            raise RuntimeError(
                f"{variable} output tiles do not match: missing={missing}, extra={extra}"
            )

        relative_paths = sorted(base_files, key=str)
        with ExitStack() as stack:
            base_sources = []
            scenario_sources = []
            for relative_path in relative_paths:
                base_src = stack.enter_context(rasterio.open(base_files[relative_path]))
                scenario_src = stack.enter_context(
                    rasterio.open(scenario_files[relative_path])
                )
                base_grid = (
                    base_src.count,
                    base_src.height,
                    base_src.width,
                    base_src.crs,
                    base_src.transform,
                )
                scenario_grid = (
                    scenario_src.count,
                    scenario_src.height,
                    scenario_src.width,
                    scenario_src.crs,
                    scenario_src.transform,
                )
                if base_grid != scenario_grid:
                    raise ValueError(
                        f"{variable} rasters are not aligned for {relative_path}"
                    )
                base_sources.append(base_src)
                scenario_sources.append(scenario_src)
            base_mosaic, base_transform = _merge_masked(base_sources)
            scenario_mosaic, scenario_transform = _merge_masked(scenario_sources)
        if base_mosaic.shape != scenario_mosaic.shape or base_transform != scenario_transform:
            raise ValueError(f"mosaicked {variable} outputs are not aligned")
        cooling = np.ma.filled(base_mosaic - scenario_mosaic, np.nan).astype("float32")
        if buffer_px:
            if 2 * buffer_px >= min(cooling.shape[-2:]):
                raise ValueError(f"buffer_px leaves no interior for {variable} mosaic")
            cooling = cooling[:, buffer_px:-buffer_px, buffer_px:-buffer_px]
        if cooling.shape[0] <= max(PEAK_HOURS_LOCAL):
            raise ValueError(
                f"{variable} must contain model hours 12--15 "
                f"(at least 16 bands); found {cooling.shape[0]}"
            )
        peak = np.mean(cooling[list(PEAK_HOURS_LOCAL)], axis=0)
        valid = peak[np.isfinite(peak)]
        if not valid.size:
            raise ValueError(f"{variable} outputs contain no finite values in the analysis area")
        output[f"cooling_{variable}_peak_mean_C"] = float(valid.mean())
        if street_analysis_mask is not None:
            mask = np.asarray(street_analysis_mask, dtype=bool)
            if buffer_px:
                mask = mask[buffer_px:-buffer_px, buffer_px:-buffer_px]
            if mask.shape != peak.shape:
                raise ValueError(
                    f"street analysis mask shape {mask.shape} differs from {variable} {peak.shape}"
                )
            street_valid = peak[mask & np.isfinite(peak)]
            if not street_valid.size:
                raise ValueError("street analysis mask contains no finite output pixels")
            output[f"cooling_{variable}_street_peak_mean_C"] = float(street_valid.mean())
    return output


def run_neighbourhood(
    city_dir: Path,
    simulation_date: str,
    dose_pp: float,
    analysis_buffer_px: int,
    fixed_tree_height_m: float | None,
    spinup_days: int = 3,
    placement_domain: str = "everywhere",
    road_scope: str = "all",
    placement_geometry: str = "crown",
    spacing_mode: str = "strategy",
    dose_mode: str = "absolute-pp",
    dose_value: float | None = None,
    strategies: tuple[str, ...] = ("adaptive", "hotspot"),
    seed: int = 42,
    eqcap_trees: int | None = None,
    allow_constrained: bool = False,
) -> dict:
    if (
        not isinstance(spinup_days, int)
        or isinstance(spinup_days, bool)
        or spinup_days < 0
    ):
        raise ValueError("spinup_days must be a nonnegative integer")
    if not isinstance(allow_constrained, bool):
        raise ValueError("allow_constrained must be boolean")
    required = ("Building_DSM.tif", "DEM.tif", "CDSM.tif", "landcover.tif", "met.txt")
    missing = [name for name in required if not (city_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing neighbourhood inputs: {missing}")
    output_root = city_dir / "output"
    base_dir = output_root / "base"
    _prepare_run_directory(city_dir, base_dir, city_dir / "CDSM.tif")
    city_id = city_dir.parent.name if city_dir.name.startswith("aoi_") else city_dir.name
    aoi_id = city_dir.name.removeprefix("aoi_") if city_dir.name.startswith("aoi_") else "root"
    model_configuration = {**MODEL_OPTIONS, "spinup_days": spinup_days}
    base_manifest = build_manifest(
        city_dir=city_dir,
        city_id=city_id,
        aoi_id=aoi_id,
        simulation_date=simulation_date,
        seed=seed,
        strategy="base",
        placement_configuration={"role": "unmodified_baseline"},
        model_configuration=model_configuration,
        input_paths=[
            city_dir / "Building_DSM.tif",
            city_dir / "DEM.tif",
            city_dir / "CDSM.tif",
            city_dir / "landcover.tif",
            city_dir / "met.txt",
        ],
    )
    base_status = _run_model(
        base_dir, city_dir / "met.txt", simulation_date, spinup_days, base_manifest
    )

    canonical = tuple(canonical_strategy(value) for value in strategies)
    if len(set(canonical)) != len(canonical):
        raise ValueError("strategies contain duplicate canonical arms")
    if any(value.startswith("street_") for value in canonical) and placement_domain != "street-verge":
        raise ValueError("street strategies require placement_domain='street-verge'")
    initial_eqcap = eqcap_trees if eqcap_trees is not None else (1 if dose_mode == "eqcap" else None)
    config = PlacementConfig(
        dose_pp=dose_pp,
        analysis_buffer_px=analysis_buffer_px,
        fixed_tree_height_m=fixed_tree_height_m,
        placement_domain=placement_domain,
        road_scope=road_scope,
        placement_geometry=placement_geometry,
        spacing_mode=spacing_mode,
        dose_mode=dose_mode,
        dose_value=dose_value,
        eqcap_trees=initial_eqcap,
        seed=seed,
        model_configuration_sha256=canonical_hash(model_configuration),
    )
    capacity = None
    if dose_mode == "eqcap" and eqcap_trees is None:
        binding, capacities = calculate_city_eqcap(city_dir, canonical, config)
        config = replace(config, eqcap_trees=binding)
        capacity = {"binding_tree_count": binding, "strategy_capacities": capacities}

    street_mask = None
    if placement_domain == "street-verge":
        from .geospatial_preprocessing.osm_eligibility import build_osm_placement_masks

        street_mask = build_osm_placement_masks(
            city_dir / "osm", city_dir / "CDSM.tif", placement_domain, road_scope
        ).street_analysis
    summary = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment": {
            "city_id": city_id,
            "aoi_id": aoi_id,
            "simulation_date": simulation_date,
            "seed": seed,
            "strategies": list(canonical),
            "configuration": asdict(config),
            "allow_constrained": allow_constrained,
            "eqcap": capacity,
            "model_configuration_sha256": canonical_hash(model_configuration),
            "base_experiment_sha256": base_manifest["experiment_sha256"],
        },
        "base": {
            "simulation_date": simulation_date,
            "spinup_days": spinup_days,
            "spinup_method": "periodic_selected_day",
            "placement_domain": placement_domain,
            "road_scope": road_scope,
            "ranking_and_evaluation_hours_local": list(PEAK_HOURS_LOCAL),
            "execution": base_status,
            "manifest": base_manifest,
        }
    }
    prepared: dict[str, tuple[Path, dict]] = {}
    for strategy in canonical:
        dose_label = (
            f"eqcap-{config.eqcap_trees}trees"
            if dose_mode == "eqcap"
            else f"{dose_mode}-{config.requested_dose:g}"
        )
        scenario_dir = output_root / (
            f"{strategy}__{placement_domain}__{placement_geometry}__"
            f"{config.spacing_mode}__{dose_label}"
        )
        placement = run_city(city_dir, scenario_dir, strategy, config)
        if placement["constrained"] and not allow_constrained:
            raise RuntimeError(
                f"{strategy} cannot meet the requested matched dose; "
                "no scenario simulation was started. Use --allow-constrained only "
                "for exploratory capacity-limited runs."
            )
        prepared[strategy] = (scenario_dir, placement)

    if dose_mode != "eqcap" and not allow_constrained:
        realized = {
            strategy: placement["dose"]["realized_pixels"]
            for strategy, (_, placement) in prepared.items()
        }
        targets = {
            placement["dose"]["requested_pixels"] for _, placement in prepared.values()
        }
        if len(targets) != 1:
            raise RuntimeError(
                "strategies were built against different pixel targets; "
                f"no scenario simulation was started: {targets}"
            )
        # A pixel-geometry arm lands on the target exactly, so the arms must
        # agree exactly.  A crown-geometry arm stops at the first crown that
        # reaches the target, so it overshoots by up to one crown footprint --
        # and by a different amount per arm, because spacing differs.  Demanding
        # exact equality there would reject every physically valid run.
        if placement_geometry == "crown":
            tolerance = max(
                placement["crown_footprint_pixels"]
                for _, placement in prepared.values()
            )
        else:
            tolerance = 0
        spread = max(realized.values()) - min(realized.values())
        if spread > tolerance:
            raise RuntimeError(
                "strategies did not realize a matched added-canopy pixel dose "
                f"(spread {spread} px exceeds the {tolerance} px tolerance); "
                f"no scenario simulation was started: {realized}"
            )

    for strategy, (scenario_dir, placement) in prepared.items():
        _prepare_run_directory(city_dir, scenario_dir, scenario_dir / f"{strategy}_CDSM.tif")
        scenario_manifest = build_manifest(
            city_dir=city_dir,
            city_id=city_id,
            aoi_id=aoi_id,
            simulation_date=simulation_date,
            seed=seed,
            strategy=strategy,
            placement_configuration=placement,
            model_configuration=model_configuration,
            input_paths=[
                city_dir / "Building_DSM.tif",
                city_dir / "DEM.tif",
                city_dir / "landcover.tif",
                city_dir / "met.txt",
                scenario_dir / f"{strategy}_CDSM.tif",
            ],
        )
        execution = _run_model(
            scenario_dir,
            city_dir / "met.txt",
            simulation_date,
            spinup_days,
            scenario_manifest,
        )
        summary[strategy] = {
            "placement": placement,
            "cooling": _delta_summary(
                base_dir, scenario_dir, analysis_buffer_px, street_mask
            ),
            "execution": execution,
            "manifest": scenario_manifest,
        }
    (output_root / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-dir", type=Path, required=True)
    parser.add_argument("--date", required=True, help="Simulation date in YYYY-MM-DD format")
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
    parser.add_argument(
        "--dose-value",
        type=float,
        help="dose magnitude; defaults to --dose-pp for backward compatibility",
    )
    parser.add_argument("--eqcap-trees", type=int)
    parser.add_argument(
        "--allow-constrained",
        action="store_true",
        help="permit exploratory arms that fail the requested dose contract",
    )
    parser.add_argument(
        "--strategies",
        default="adaptive,hotspot",
        help=f"comma-separated strategies; available: {','.join(STRATEGIES)}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spinup-days", type=int, default=3,
        help="periodic forcing days used to initialize thermal state (default: 3)",
    )
    args = parser.parse_args()
    strategies = tuple(value.strip() for value in args.strategies.split(",") if value.strip())
    result = run_neighbourhood(
        city_dir=args.city_dir,
        simulation_date=args.date,
        dose_pp=args.dose_pp,
        analysis_buffer_px=args.analysis_buffer_px,
        fixed_tree_height_m=args.fixed_tree_height_m,
        spinup_days=args.spinup_days,
        placement_domain=args.placement_domain,
        road_scope=args.road_scope,
        placement_geometry=args.placement_geometry,
        spacing_mode=args.spacing_mode,
        dose_mode=args.dose_mode,
        dose_value=args.dose_value,
        strategies=strategies,
        seed=args.seed,
        eqcap_trees=args.eqcap_trees,
        allow_constrained=args.allow_constrained,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
