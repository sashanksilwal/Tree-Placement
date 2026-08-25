# Tree Placement

Tree Placement is the companion software for the study “Where Trees are Planted
Matters More Than How Many.” It contains the frozen microclimate engine,
geospatial preprocessing, tree-placement algorithms, simulation workflow, and
statistical analysis needed to compare Adaptive and Hotspot planting at matched
canopy doses.

The scientific question is narrow: given the same increase in
canopy cover, does distributing trees according to pedestrian radiant exposure
provide more cooling than concentrating them on the hottest modeled surfaces?

## Study workflow

The repository provides four connected stages:

1. Prepare aligned terrain, building, canopy, land-cover, and meteorological
   inputs for an urban neighborhood.
2. Run the unmodified baseline scene.
3. Add an equal canopy dose using Adaptive and Hotspot placement and rerun both
   scenes.
4. Calculate paired cooling differences and equal-cooling canopy requirements.

The model computes mean radiant temperature, UTCI, surface temperature, and
supporting flux diagnostics on CPU or CUDA-capable GPUs. The repository contains
the model source so that the study does not depend on an unpublished external
package.

## Installation

The supplied environments are the recommended installation route because GDAL
and PDAL are compiled dependencies.

CPU:

```bash
micromamba create -n tree-placement -f environment.yml
micromamba activate tree-placement
python -m pip install --no-deps .
```

For an NVIDIA GPU, substitute `environment-gpu.yml` when creating the
environment. Python 3.10 and 3.11 are supported; the environment files use
Python 3.11.

Verify a checkout with:

```bash
python -B -m study_code.preflight .
pytest -q
python examples/placement_smoke_test.py
```

## Required neighborhood inputs

Each neighborhood directory must contain co-registered, single-band rasters
with identical CRS, resolution, transform, and extent:

| File | Meaning |
|---|---|
| `DEM.tif` | Bare-earth elevation in metres |
| `Building_DSM.tif` | Terrain plus building elevation in metres |
| `CDSM.tif` | Canopy height above ground in metres; zero means no canopy |
| `landcover.tif` | Integer surface class used by the energy balance |
| `met.txt` | Hourly meteorology in the supported UMEP/SOLWEIG text format |

Placement also requires an `osm/` directory containing `highway.geojson`,
`sidewalk.geojson`, `building.geojson`, `parking.geojson`,
`residential.geojson`, `water.geojson`, and `pool.geojson`.
Download all seven layers for the exact raster extent with:

```bash
tree-placement-osm download --city-dir /path/to/neighborhood
```

The downloader records its time, bounding box, endpoints, and feature counts in
`osm/manifest.json`. Missing layers are an error; the placement workflow does
not silently fall back to unscreened ground.

The preprocessing modules support US neighborhood preparation from USGS 3DEP
or The National Map LAZ, Microsoft building footprints, ESA WorldCover, and
ERA5. Availability and source vintage vary by location; the workflow must not
be described as complete nationwide coverage.

## Placement methods

Temperature-ranked methods use the same 12:00–15:00 local-time window of the baseline surface
temperature: model-hour indices 12–15 (GeoTIFF bands 13–16). All four bands are
required. The cooling summaries use the same window and also fail if any band is
missing.

`adaptive` is the public and canonical strategy name; the legacy analysis name
`svf_gate` remains accepted only as an input alias. New filenames, manifests,
and summaries always use `adaptive`. The AOI-wide strategy set is
`random`, `high_svf`, `near_buildings`, `impervious`, `expand`, `hotspot`,
`hotspot_spread`, and `adaptive`. The street-verge set is `street_random`,
`street_hotspot`, `street_hotspot_spread`, `street_adaptive`,
`street_canopy_dilate`, `street_unshaded_hot`, `street_shade_greedy`,
`street_cluster_connect`, and `street_building_complement`.

`--placement-geometry pixel` creates one-canopy-pixel interventions;
`--placement-geometry crown` uses complete discrete circular crowns.
`--spacing-mode strategy` applies the documented strategy-specific rule: spread
strategies use archived grid thinning for pixel interventions and
non-overlapping centres for crowns. The resolved
spacing rule is recorded in every placement summary. Geometry alone does not
reproduce an archived experiment because timing, eligibility, dose, and spacing
also form part of the experiment identity.

Three dose contracts are available:

- `absolute-pp`: added canopy pixels as percentage points of the analysis area;
- `relative-canopy`: added pixels as a percentage of baseline canopy pixels;
- `eqcap`: an equal number of trees across the selected strategies. When
  `--eqcap-trees` is omitted, the smallest exact strategy capacity binds.

Every placement records requested and realized pixels, trees, added percentage
points, relative-canopy percentage, rankable capacity, and whether the target
was met. The study runner stops before scenario simulations if an arm cannot
meet its requested dose or fixed-dose arms realize different added-pixel totals.
`--allow-constrained` overrides this for exploratory runs; its outputs must not
be included in a matched-dose publication comparison.

There are two placement domains. `--placement-domain everywhere` allows
canopy-free, non-building centres after excluding OSM road surfaces, mapped
sidewalks/paths, parking areas, water, and swimming pools.

`--placement-domain street-verge` plants beside a road rather than on it, in a
per-class ring measured from the centreline, together with mapped sidewalk
buffers. Each ring starts at the carriageway half-width and stops at the
planting-strip cap for that road class:

| Road class | Ring from centreline |
|---|---|
| `motorway` | 15–25 m |
| `trunk` | 12–20 m |
| `primary` | 8–14 m |
| `secondary` | 6–11 m |
| `tertiary` | 5–9 m |
| `residential` | 3.5–7 m |
| `service`, `unclassified`, `living_street` | 3–6 m |

Because the inner radius is the carriageway half-width, the paved surface is
excluded by construction. A wide road cannot host a centre in its traffic lanes.

`--road-scope all` is the default and unions every mapped highway class, which
reproduces the archived study definition. `--road-scope local` drops motorway
and trunk. OSM buildings supplement the elevation-derived building mask and must
remain clear of the complete simulated crown. Other OSM exclusions constrain the
trunk centre only, so a verge crown may overhang a mapped road edge.

OSM `landuse=residential` is downloaded and audited but is not blanket-excluded:
that tag covers neighborhoods rather than specific unplantable surfaces. Parcel
ownership and yard access therefore remain outside the eligibility model.

These are reproducible model-eligibility rules, not a legal or operational
plantability assessment. OSM may be incomplete and does not establish parcel
ownership, planting permission, utility clearance, soil volume, or underground
infrastructure.

Candidate tree height can be fixed with `--fixed-tree-height-m`. Otherwise,
height is assigned from nearby observed canopy, expanding the search radius
when necessary and falling back to the neighborhood median.

Run a baseline and selected AOI-wide strategy arms for one neighborhood:

```bash
python -m study_code.run_study \
  --city-dir /path/to/neighborhood \
  --date 2023-07-15 \
  --dose-pp 10 \
  --dose-mode absolute-pp \
  --placement-geometry crown \
  --strategies random,expand,hotspot,hotspot_spread,adaptive \
  --analysis-buffer-px 200 \
  --placement-domain everywhere \
  --spinup-days 3
```

For an equal-capacity street experiment:

```bash
python -m study_code.run_study \
  --city-dir /path/to/neighborhood \
  --date 2023-07-15 \
  --placement-domain street-verge \
  --placement-geometry pixel \
  --dose-mode eqcap \
  --strategies street_random,street_hotspot,street_hotspot_spread,street_adaptive,street_canopy_dilate
```

Run directories contain a versioned `run_manifest.json` with city, AOI, date,
seed, strategy, placement configuration, input hashes, source-code hash, and
model-configuration hash. Resume is permitted only when the experiment hash and
all required output families match. Legacy or incomplete output directories
must be archived explicitly before reuse.

The default initialization repeats the selected daily forcing for three
periodic spin-up days and writes only the requested day. This is a numerical
initialization scenario, not observed antecedent weather.

## Statistical analysis

The paired analysis expects a CSV containing `city_id`, `strategy`, and the
selected cooling metric:

```bash
python -m study_code.statistical_analysis paired results.csv \
  --replicates 50000 \
  --output paired.json
```

Equal-cooling analysis expects `city_id`, `strategy`, `dose_pp`, and
`cooling_C`:

```bash
python -m study_code.statistical_analysis equal-cooling dose_response.csv \
  --target-dose-pp 20 \
  --output equal_cooling.json
```

The target Hotspot dose must lie inside the measured Hotspot range, and the
inferred Adaptive dose must lie inside the measured Adaptive range. The command
fails rather than reporting an extrapolated comparison.

## Reproducibility boundaries

- Results apply to the documented neighborhood, meteorology, analysis buffer,
  tree geometry, OSM snapshot, placement domain, dose ladder, and model
  configuration.
- The standard US workflow uses 1 m neighborhood scenes where suitable LiDAR
  coverage exists; it does not represent every neighborhood within a city.
- Periodic spin-up establishes numerical state but does not reconstruct prior
  weather.
- Diagnostic air-temperature and wind fields are reduced urban models. Mean
  radiant temperature is the primary quantity used by the placement comparison.
- Third-party source datasets and generated simulation outputs are not stored in
  this software repository. Their exact locations and archived derivatives must
  be identified in the paper’s Data Availability statement.

## Repository contents

- `study_code/tree_placement.py`: AOI-wide and street-verge placement strategies.
- `study_code/run_study.py`: matched baseline and scenario simulations.
- `study_code/statistical_analysis.py`: paired and equal-cooling inference.
- `study_code/geospatial_preprocessing/`: study-area and input preparation.
- `study_code/geospatial_preprocessing/osm_eligibility.py`: OSM download and
  placement-domain masks.
- `tests/`: unit, integration, and workflow tests.
- `examples/`: small reproducible examples.
- `docs/`: equations, assumptions, and scientific limitations for optional
  experimental components included in the source tree.

## Citation and license

Use the metadata in `CITATION.cff` when citing this software. Upstream model
attribution is recorded in `NOTICE`. The source is distributed under
GPL-3.0-or-later; see `LICENSE`.
