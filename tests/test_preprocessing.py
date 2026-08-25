import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from study_code.geospatial_preprocessing.download_era5_cities import (
    MET_HEADER,
    erbs_decomposition,
    met_matches_date,
    relative_humidity,
    solar_zenith,
    validate_met,
    wind_speed_direction,
)
from study_code.geospatial_preprocessing import prepare_nature_cities
from study_code.geospatial_preprocessing import prepare_usgs_city
from study_code.geospatial_preprocessing.prepare_usgs_city import _project_year
from study_code.geospatial_preprocessing.prepare_nature_cities import (
    gap_fill,
    run_download,
    save_tif,
)


def _met_row(year=2023, doy=196):
    row = np.zeros(24, dtype=float)
    row[0:4] = (year, doy, 12, 0)
    row[9:15] = (2.0, 50.0, 25.0, 101.3, 0.0, 800.0)
    row[21:24] = (100.0, 700.0, 180.0)
    return row


def test_project_year_extracts_complete_calendar_year():
    assert _project_year("USGS LPC IN Lake County 2019") == 2019
    assert _project_year("Project 2018 updated 2024") == 2024


def test_project_year_understands_compact_project_code():
    assert _project_year("USGS_LPC_D22") == 2022


def test_existing_met_file_must_match_requested_date(tmp_path):
    path = tmp_path / "met.txt"
    np.savetxt(path, _met_row()[None, :], header=MET_HEADER, comments="")
    assert met_matches_date(path, "2023-07-15")
    assert not met_matches_date(path, "2023-07-16")


def test_fractional_met_date_is_not_accepted(tmp_path):
    path = tmp_path / "met.txt"
    row = _met_row()
    row[0] = 2023.5
    np.savetxt(path, row[None, :], header=MET_HEADER, comments="")
    assert not met_matches_date(path, "2023-07-15")


def test_solar_zenith_is_near_zero_at_equatorial_equinox_noon():
    zenith = solar_zenith(0.0, 0.0, 80, 12.0)
    assert np.degrees(zenith) < 3.0


def test_erbs_outputs_are_bounded_at_low_sun():
    dhi, dni = erbs_decomposition(500.0, 40.0, -87.0, 196, 11.0)
    assert 0.0 <= dhi <= 500.0
    assert 0.0 <= dni <= 1361.0


def test_erbs_rejects_negative_irradiance():
    with pytest.raises(ValueError, match="ghi"):
        erbs_decomposition(-1.0, 40.0, -87.0, 196, 18.0)


def test_humidity_and_wind_component_conventions():
    assert relative_humidity(20.0, 20.0) == pytest.approx(100.0)
    speed, direction = wind_speed_direction(0.0, -2.0)
    assert speed == pytest.approx(2.0)
    assert direction == pytest.approx(0.0)
    _, direction = wind_speed_direction(-2.0, 0.0)
    assert direction == pytest.approx(90.0)


def test_single_row_met_file_can_be_validated(tmp_path):
    path = tmp_path / "met.txt"
    rows = np.repeat(_met_row()[None, :], 24, axis=0)
    rows[:, 2] = np.arange(24)
    np.savetxt(path, rows, header=MET_HEADER, comments="")
    assert validate_met(path)


def test_all_nodata_raster_cannot_be_gap_filled():
    with np.testing.assert_raises_regex(ValueError, "only nodata"):
        gap_fill(np.full((4, 4), np.nan, dtype=np.float32))


def test_all_nodata_raster_cannot_be_saved(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "finite values"):
        save_tif(
            np.full((4, 4), np.nan, dtype=np.float32),
            tmp_path / "bad.tif",
            (0.0, 0.0, 4.0, 4.0),
            32616,
        )


def test_download_requires_city_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="cities.json"):
        run_download(tmp_path)


def test_download_rejects_unknown_city(tmp_path):
    (tmp_path / "cities.json").write_text('[{"id": 7, "name": "Example"}]')
    with pytest.raises(ValueError, match="city ID 8"):
        run_download(tmp_path, city_id=8)


class _BrokenPdal:
    class Pipeline:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def execute(self):
            raise RuntimeError("broken point-cloud input")


def test_las_failure_stops_preprocessing(monkeypatch):
    monkeypatch.setattr(prepare_usgs_city, "pdal", _BrokenPdal)
    with pytest.raises(RuntimeError, match="LAS read failed"):
        prepare_usgs_city.rasterize_las(
            ["broken.laz"], (0.0, 0.0, 4.0, 4.0), 32616, [2]
        )


def test_copc_failure_stops_preprocessing(monkeypatch):
    monkeypatch.setattr(prepare_nature_cities, "pdal", _BrokenPdal)
    item = SimpleNamespace(assets={"data": SimpleNamespace(href="broken.copc.laz")})
    with pytest.raises(RuntimeError, match="COPC read failed"):
        prepare_nature_cities.rasterize_copc(
            [item], (0.0, 0.0, 4.0, 4.0), 32616, [2]
        )


def test_failed_tile_download_leaves_no_partial_file(tmp_path, monkeypatch):
    def fail_download(url, path):
        path.write_bytes(b"partial")
        raise OSError("network interrupted")

    monkeypatch.setattr(prepare_usgs_city.urllib.request, "urlretrieve", fail_download)
    target = tmp_path / "tiles"
    with pytest.raises(RuntimeError, match="tile download failed"):
        prepare_usgs_city.download_tiles(["https://example.test/tile.laz"], target)
    assert not (target / "tile.laz").exists()
    assert not (target / "tile.laz.part").exists()


def test_extract_key_accepts_both_met_file_conventions():
    """Generated forcing carries a date; user-supplied met does not.

    preprocessor.py writes metfile_X_Y_YYYY-MM-DD.txt for generated forcing and
    metfile_X_Y.txt when the caller supplies their own met file. Both must map to
    the same tile key or the run aborts on a tile-key mismatch.
    """
    from utherm.utci_process import extract_key

    assert extract_key("metfile_0_0_2023-07-06.txt", is_metfile=True) == "0_0"
    assert extract_key("metfile_0_0.txt", is_metfile=True) == "0_0"
    assert extract_key("metfile_12_7.txt", is_metfile=True) == "12_7"
    assert extract_key("metfile_12_7_2019-08-01.txt", is_metfile=True) == "12_7"
    assert extract_key("notamet_0_0.txt", is_metfile=True) is None
