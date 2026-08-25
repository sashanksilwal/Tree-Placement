# Repository contents

- `utherm/`: UTherm model, coupled physics, and command-line interface.
- `utherm/energy_balance/coupled.py`: opt-in seven-facet coupled energy balance.
- `utherm/coupled_pipeline.py`: SOLWEIG shortwave and coupled-radiosity bridge.
- `utherm/view_factors.py`: reciprocal surface and pedestrian-view geometry tracer.
- `study_code/tree_placement.py`: AOI-wide and street-verge placement strategies.
- `study_code/experiment_contract.py`: versioned manifests, hashes, and resume validation.
- `study_code/run_study.py`: base and placement simulation workflow.
- `study_code/geospatial_preprocessing/`: study-area and input preparation.
- `study_code/geospatial_preprocessing/osm_eligibility.py`: reproducible OSM
  downloads and everywhere/street-verge eligibility masks.
- `study_code/statistical_analysis.py`: paired and equal-cooling analyses.
- `tests/`: unit and workflow tests.
- `examples/`: placement, geometry, and coupled-physics examples.
- `docs/COUPLED_ENERGY_BALANCE.md`: coupled solver equations, inputs, and limits.
- `environment.yml`: CPU environment.
- `environment-gpu.yml`: NVIDIA GPU environment.
- `pyproject.toml`: Python package metadata and command-line entry points.
- `.github/`: issue templates, continuous integration, and tagged-release automation.
- `CONTRIBUTING.md`, `SECURITY.md`: project maintenance guidance.
- `CITATION.cff`, `NOTICE`, `LICENSE`: citation, attribution, and license terms.

Third-party input datasets, simulation outputs, and internal diagnostic reports are
not included.
