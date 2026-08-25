"""Check that a UTherm public repository is complete and clean."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED = (
    ".gitattributes",
    ".github/workflows/release.yml",
    ".github/workflows/tests.yml",
    ".gitignore",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "MANIFEST.md",
    "pyproject.toml",
    "environment.yml",
    "environment-gpu.yml",
    "docs/COUPLED_ENERGY_BALANCE.md",
    "utherm/__init__.py",
    "utherm/coupled_pipeline.py",
    "utherm/energy_balance/coupled.py",
    "utherm/view_factors.py",
    "study_code/tree_placement.py",
    "study_code/experiment_contract.py",
    "study_code/run_study.py",
    "study_code/statistical_analysis.py",
    "study_code/geospatial_preprocessing/prepare_nature_cities.py",
    "study_code/geospatial_preprocessing/generate_coupled_geometry.py",
    "study_code/geospatial_preprocessing/osm_eligibility.py",
    "examples/evaluate_real_aoi_sensitivity.py",
    "tests/test_tree_placement.py",
    "tests/test_experiment_contract.py",
    "tests/test_osm_eligibility.py",
    "tests/test_placement_eligibility.py",
    "tests/test_statistical_analysis.py",
    "tests/test_core.py",
    "tests/test_solver.py",
    "tests/test_coupled_urban_eb.py",
    "tests/test_coupled_pipeline.py",
    "tests/test_view_factors.py",
)

PRIVATE_RELEASE_ARTIFACTS = {
    "CHECKSUMS.sha256",
    "QA_REPORT.md",
    "RELEASE_STATUS.md",
    "SCIENTIFIC_AUDIT.md",
    "ZENODO_METADATA.md",
}


def _declared_version(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


def check_versions(root: Path) -> list[str]:
    """Keep the packaged, importable, and cited versions aligned.

    These are separate declarations that nothing else compares, so they drift
    silently and ship a wheel whose metadata disagrees with the module it
    installs.
    """
    errors: list[str] = []
    project = root / "pyproject.toml"
    module = root / "utherm" / "__init__.py"
    citation = root / "CITATION.cff"
    if not (project.is_file() and module.is_file()):
        return errors

    packaged = _declared_version(
        project.read_text(), r'^version\s*=\s*["\']([^"\']+)["\']'
    )
    importable = _declared_version(
        module.read_text(), r'^__version__\s*=\s*["\']([^"\']+)["\']'
    )
    if packaged is None:
        errors.append("pyproject.toml declares no version")
    if importable is None:
        errors.append("utherm/__init__.py declares no __version__")
    if packaged and importable and packaged != importable:
        errors.append(
            f"version mismatch: pyproject.toml {packaged} != utherm.__version__ {importable}"
        )
    if citation.is_file():
        cited = _declared_version(citation.read_text(), r"^version:\s*([^\s]+)")
        if packaged and cited and packaged != cited:
            errors.append(
                f"version mismatch: pyproject.toml {packaged} != CITATION.cff {cited}"
            )
    return errors


def check_package(root: Path) -> list[str]:
    errors = [f"missing {name}" for name in REQUIRED if not (root / name).is_file()]
    errors.extend(check_versions(root))
    forbidden: list[str] = []
    forbidden_names = {
        ".DS_Store",
        "__MACOSX",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        ".idea",
        ".vscode",
        *PRIVATE_RELEASE_ARTIFACTS,
    }
    forbidden_dirs = {"build", "dist", "validation"}
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if ".git" in relative_parts:
            continue
        if (
            path.name in forbidden_names
            or path.suffix in {".pyc", ".log"}
            or any(part in forbidden_dirs or part.endswith(".egg-info") for part in relative_parts)
        ):
            forbidden.append(path.relative_to(root).as_posix())
    errors.extend(f"forbidden generated or internal file {name}" for name in forbidden)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check_package(args.root)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"preflight passed: {args.root}")


if __name__ == "__main__":
    main()
