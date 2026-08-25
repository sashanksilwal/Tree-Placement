import pytest

from study_code.experiment_contract import (
    EXPERIMENT_SCHEMA_VERSION,
    build_manifest,
    canonical_hash,
    validate_resume,
    write_manifest,
)


def _manifest(tmp_path, strategy="random"):
    city = tmp_path / "17" / "aoi_1"
    city.mkdir(parents=True, exist_ok=True)
    source = city / "input.txt"
    source.write_text("stable input\n")
    return build_manifest(
        city_dir=city,
        city_id="17",
        aoi_id="1",
        simulation_date="2023-07-15",
        seed=42,
        strategy=strategy,
        placement_configuration={"placement_geometry": "pixel"},
        model_configuration={"use_energy_balance": True},
        input_paths=[source],
    )


def test_manifest_records_version_hashes_identity_and_inputs(tmp_path):
    first = _manifest(tmp_path)
    second = _manifest(tmp_path)
    assert first["schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert first["source_code_sha256"]
    assert first["model_configuration_sha256"] == canonical_hash(
        {"use_energy_balance": True}
    )
    assert first["experiment_sha256"] == second["experiment_sha256"]
    assert next(iter(first["inputs"].values()))["sha256"]


def test_resume_requires_matching_manifest_and_every_output(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "run"
    output = run / "output_folder" / "0_0"
    output.mkdir(parents=True)
    (output / "TMRT_0_0.tif").touch()
    write_manifest(run / "run_manifest.json", manifest)
    assert validate_resume(run, manifest, ["TMRT_*.tif"])
    with pytest.raises(RuntimeError, match="missing outputs"):
        validate_resume(run, manifest, ["TMRT_*.tif", "Tair_*.tif"])

    mismatch = dict(manifest)
    mismatch["experiment_sha256"] = "different"
    with pytest.raises(RuntimeError, match="does not match"):
        validate_resume(run, mismatch, ["TMRT_*.tif"])


def test_resume_rejects_unmanifested_legacy_output(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "legacy"
    (run / "output_folder").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no versioned manifest"):
        validate_resume(run, manifest, ["TMRT_*.tif"])
