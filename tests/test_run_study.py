import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from study_code.run_study import (
    _delta_summary,
    _prepare_run_directory,
    _validate_model_output_rasters,
    run_neighbourhood,
)


def _write_output(root, variable, key, value, transform, nodata=None, count=16):
    path = root / "output_folder" / key / f"{variable}_{key}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((count, 2, 2), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=count,
        dtype="float32",
        crs="EPSG:32616",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data)


def _write_all_variables(root, key, value, transform):
    for variable in ("TMRT", "Tsfc", "UTCI", "Tair"):
        _write_output(root, variable, key, value, transform)


def test_delta_summary_reports_positive_cooling_and_supports_multiple_tiles(tmp_path):
    base = tmp_path / "base"
    scenario = tmp_path / "adaptive"
    first = from_origin(0.0, 2.0, 1.0, 1.0)
    second = from_origin(2.0, 2.0, 1.0, 1.0)
    _write_all_variables(base, "0_0", 10.0, first)
    _write_all_variables(base, "2_0", 10.0, second)
    _write_all_variables(scenario, "0_0", 8.0, first)
    _write_all_variables(scenario, "2_0", 6.0, second)

    result = _delta_summary(base, scenario, buffer_px=0)

    assert result["cooling_TMRT_peak_mean_C"] == pytest.approx(3.0)
    assert result["cooling_UTCI_peak_mean_C"] == pytest.approx(3.0)


def test_delta_summary_rejects_missing_scenario_tile(tmp_path):
    base = tmp_path / "base"
    scenario = tmp_path / "adaptive"
    first = from_origin(0.0, 2.0, 1.0, 1.0)
    second = from_origin(2.0, 2.0, 1.0, 1.0)
    _write_all_variables(base, "0_0", 10.0, first)
    _write_all_variables(base, "2_0", 10.0, second)
    _write_all_variables(scenario, "0_0", 8.0, first)

    with pytest.raises(RuntimeError, match="output tiles do not match"):
        _delta_summary(base, scenario, buffer_px=0)


def test_delta_summary_excludes_nodata_pixels(tmp_path):
    base = tmp_path / "base"
    scenario = tmp_path / "adaptive"
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    for variable in ("TMRT", "Tsfc", "UTCI", "Tair"):
        _write_output(base, variable, "0_0", 10.0, transform, nodata=-9999.0)
        _write_output(scenario, variable, "0_0", 8.0, transform, nodata=-9999.0)
        path = scenario / "output_folder" / "0_0" / f"{variable}_0_0.tif"
        with rasterio.open(path, "r+") as dataset:
            data = dataset.read()
            data[:, 0, 0] = -9999.0
            dataset.write(data)

    result = _delta_summary(base, scenario, buffer_px=0)
    assert result["cooling_TMRT_peak_mean_C"] == pytest.approx(2.0)


def test_delta_summary_requires_complete_12_to_15_window(tmp_path):
    base = tmp_path / "base"
    scenario = tmp_path / "adaptive"
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    for variable in ("TMRT", "Tsfc", "UTCI", "Tair"):
        _write_output(base, variable, "0_0", 10.0, transform, count=15)
        _write_output(scenario, variable, "0_0", 8.0, transform, count=15)
    with pytest.raises(ValueError, match="hours 12--15"):
        _delta_summary(base, scenario, buffer_px=0)


def test_model_output_validation_rejects_short_or_unreadable_families(tmp_path):
    run_dir = tmp_path / "run"
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    _write_all_variables(run_dir, "0_0", 10.0, transform)
    _validate_model_output_rasters(run_dir)
    _write_output(run_dir, "TMRT", "0_0", 10.0, transform, count=15)
    with pytest.raises(RuntimeError, match="lacks model hours"):
        _validate_model_output_rasters(run_dir)


def test_prepare_run_directory_rejects_stale_input_link(tmp_path):
    city = tmp_path / "city"
    city.mkdir()
    for name in ("Building_DSM.tif", "DEM.tif", "CDSM.tif"):
        (city / name).touch()
    run_dir = city / "output" / "base"
    _prepare_run_directory(city, run_dir, city / "CDSM.tif")
    replacement = city / "replacement.tif"
    replacement.touch()
    with pytest.raises(RuntimeError, match="does not point"):
        _prepare_run_directory(city, run_dir, replacement)


def test_run_neighbourhood_rejects_invalid_spinup_before_writing(tmp_path):
    with pytest.raises(ValueError, match="spinup_days"):
        run_neighbourhood(tmp_path, "2023-07-15", 10.0, 0, 8.0, -1)
    assert not (tmp_path / "output").exists()


def _minimal_neighbourhood_inputs(city):
    city.mkdir()
    for name in ("Building_DSM.tif", "DEM.tif", "CDSM.tif", "landcover.tif", "met.txt"):
        (city / name).touch()


def test_run_neighbourhood_stops_before_scenario_model_when_dose_is_constrained(
    tmp_path, monkeypatch
):
    city = tmp_path / "city"
    _minimal_neighbourhood_inputs(city)
    model_calls = []
    monkeypatch.setattr(
        "study_code.run_study._run_model",
        lambda *args, **kwargs: model_calls.append(args[0]) or "completed",
    )
    monkeypatch.setattr(
        "study_code.run_study.run_city",
        lambda *args, **kwargs: {
            "constrained": True,
            "dose": {"realized_pixels": 3},
        },
    )
    with pytest.raises(RuntimeError, match="cannot meet the requested matched dose"):
        run_neighbourhood(city, "2023-07-15", 10.0, 0, 8.0)
    assert model_calls == [city / "output" / "base"]


def test_run_neighbourhood_checks_equal_realized_pixels_before_scenarios(
    tmp_path, monkeypatch
):
    city = tmp_path / "city"
    _minimal_neighbourhood_inputs(city)
    model_calls = []
    monkeypatch.setattr(
        "study_code.run_study._run_model",
        lambda *args, **kwargs: model_calls.append(args[0]) or "completed",
    )
    # Crown arms stop at the first crown that reaches the target, so they
    # overshoot by up to one crown footprint.  A spread inside that tolerance is
    # physically unavoidable and must be accepted; a spread beyond it is a real
    # dose mismatch and must stop the study before any scenario is simulated.
    realized = iter((1000, 1400))
    monkeypatch.setattr(
        "study_code.run_study.run_city",
        lambda *args, **kwargs: {
            "constrained": False,
            "crown_footprint_pixels": 49,
            "dose": {"realized_pixels": next(realized), "requested_pixels": 1000},
        },
    )
    with pytest.raises(RuntimeError, match="matched added-canopy pixel dose"):
        run_neighbourhood(city, "2023-07-15", 10.0, 0, 8.0)
    assert model_calls == [city / "output" / "base"]


def test_run_neighbourhood_accepts_crown_overshoot_within_one_footprint(
    tmp_path, monkeypatch
):
    city = tmp_path / "city"
    _minimal_neighbourhood_inputs(city)
    monkeypatch.setattr(
        "study_code.run_study._run_model",
        lambda *args, **kwargs: "completed",
    )
    monkeypatch.setattr(
        "study_code.run_study._delta_summary", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "study_code.run_study._prepare_run_directory", lambda *args, **kwargs: None
    )
    # _prepare_run_directory normally creates this; it is stubbed out here.
    (city / "output").mkdir(parents=True, exist_ok=True)
    # run_city is stubbed, so no scenario raster exists to hash.
    monkeypatch.setattr(
        "study_code.run_study.build_manifest",
        lambda **kwargs: {"experiment_sha256": kwargs["strategy"]},
    )
    realized = iter((1002, 1026))
    monkeypatch.setattr(
        "study_code.run_study.run_city",
        lambda *args, **kwargs: {
            "constrained": False,
            "crown_footprint_pixels": 49,
            "dose": {"realized_pixels": next(realized), "requested_pixels": 1000},
        },
    )
    summary = run_neighbourhood(city, "2023-07-15", 10.0, 0, 8.0)
    assert set(summary) >= {"adaptive", "hotspot"}
