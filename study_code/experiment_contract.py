"""Versioned manifests, hashes, and completion checks for placement experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXPERIMENT_SCHEMA_VERSION = "tree-placement-v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_code_hash(repository_root: Path | None = None) -> str:
    root = repository_root or Path(__file__).resolve().parents[1]
    paths: list[Path] = []
    for package in (root / "study_code", root / "utherm"):
        if package.is_dir():
            paths.extend(sorted(package.rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def input_provenance(paths: Iterable[Path], root: Path | None = None) -> dict[str, dict]:
    result = {}
    for path in paths:
        resolved = path.resolve()
        key = (
            resolved.relative_to(root.resolve()).as_posix()
            if root is not None and resolved.is_relative_to(root.resolve())
            else str(resolved)
        )
        result[key] = {
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    return result


def build_manifest(
    *,
    city_dir: Path,
    city_id: str,
    aoi_id: str,
    simulation_date: str,
    seed: int,
    strategy: str,
    placement_configuration: dict,
    model_configuration: dict,
    input_paths: Iterable[Path],
) -> dict:
    model_hash = canonical_hash(model_configuration)
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": str(city_id),
        "aoi_id": str(aoi_id),
        "simulation_date": simulation_date,
        "seed": seed,
        "strategy": strategy,
        "placement_configuration": placement_configuration,
        "model_configuration": model_configuration,
        "model_configuration_sha256": model_hash,
        "source_code_sha256": source_code_hash(),
        "inputs": input_provenance(input_paths, root=city_dir),
    }
    identity = dict(manifest)
    identity.pop("created_at_utc")
    manifest["experiment_sha256"] = canonical_hash(identity)
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_resume(
    run_dir: Path,
    expected_manifest: dict,
    required_patterns: Iterable[str],
) -> bool:
    """Return true only for a complete run with the exact expected identity.

    Existing outputs without a manifest, with a different experiment hash, or
    with missing completion artifacts fail loudly instead of being reused.
    """
    output_root = run_dir / "output_folder"
    if not output_root.exists():
        return False
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"existing run has no versioned manifest: {run_dir}")
    observed = json.loads(manifest_path.read_text())
    if observed.get("experiment_sha256") != expected_manifest.get("experiment_sha256"):
        raise RuntimeError(f"existing run manifest does not match requested experiment: {run_dir}")
    missing = [pattern for pattern in required_patterns if not list(output_root.rglob(pattern))]
    if missing:
        raise RuntimeError(f"existing run is incomplete; missing outputs {missing}: {run_dir}")
    return True
