# Coupled urban energy balance

The experimental solver represents seven facets at every plan-area cell:
ground, roof, four wall orientations and canopy. Ground, roof and walls have
independent material columns. The canopy is an equilibrium leaf surface.

## Geometry and radiation

Facet area is written as `A_i`. The supplied exchange area `B_ij` must satisfy

```
B_ij = B_ji
A_i = B_i,sky + sum_j B_ij
```

The view factor is `F_ij = B_ij / A_i`. These two checks enforce reciprocity
and closure before a timestep starts.

For shortwave radiation, the gray-diffuse radiosity is

```
J_i = alpha_i (K_i,external + sum_j F_ij J_j)
K_i,absorbed = (1 - alpha_i) (K_i,external + sum_j F_ij J_j)
```

For longwave radiation,

```
J_i = epsilon_i sigma T_i^4
      + (1 - epsilon_i) (F_i,sky L_sky + sum_j F_ij J_j)
L_i,net = epsilon_i (F_i,sky L_sky + sum_j F_ij J_j - sigma T_i^4)
```

Both systems are solved as batched linear systems. Leaf temperature therefore
changes the longwave field seen by the other facets during the same coupled
iteration.

## Surface and storage budgets

Every active facet solves

```
K_absorbed + L_net = H + LE + G
```

`H` is evaluated against canyon air for ground, walls and leaves, and against
above-roof air for roofs. `LE` uses the iterated canyon vapor pressure,
aerodynamic and surface resistance, and cannot evaporate more water than the
stored amount. For the canopy, this is **wet-canopy interception evaporation**.
Although its conductance uses radiation, vapor-pressure-deficit and temperature
response factors, it is not root-supplied transpiration and must not be reported
as complete evapotranspiration.
Leaf storage and conduction are zero. Solid-facet storage `G` is coupled to an
implicit finite-volume material column with harmonic interface conductance.
The lower boundary is fixed deep-ground temperature for ground and indoor
temperature for roofs and walls.

## Canyon air

The canyon-air update is implicit:

```
(rho cp Hc / dt) (Tcan - Tcan_old)
  = sum_i A_i h_i (T_i - Tcan)
    - rho cp u_exchange (Tcan - Ta)
    + Q_anthropogenic
```

The roof is excluded from the canyon sum. Local URock wind may be supplied as
the wind field used for turbulent transfer and ventilation.

## Moisture and spin-up

Water storage is advanced from captured precipitation, evaporation and
overflow drainage. It is bounded between zero and the user-supplied capacity.
Periodic spin-up starts the interception stores dry. Rain in the repeated
forcing cycle then establishes the periodic water state; the model does not
invent an arbitrary half-full store for a rain-free clear-day cycle.
The low-level `CoupledRadiationBridge.solve_cycle` also accepts a distinct
`spinup_forcings` cycle. This permits an explicitly declared antecedent
scenario (for example, dry spin-up followed by observed-day rain) without
removing precipitation from the reported-day solve. It must not be presented
as observed antecedent weather unless those forcing data were actually used.
The public `thermal_comfort(use_coupled_eb=True)` path uses this dry-antecedent
scenario: it sets precipitation to zero only in the repeated spin-up cycle and
retains the input precipitation in the requested output cycle. This avoids
repeating one observed-day storm as fictitious antecedent rainfall, but it is
still an assumption that must be disclosed for wet-period applications.
Precipitation received during a timestep is available to evaporation in that
same timestep. Capture fractions are per facet, and the solver requires

```
sum_i A_i capture_i <= 1
```

at every plan-area cell. This prevents one unit of rainfall from being counted
independently on the ground, roof and canopy. A canopy interception model must
partition throughfall and interception into fractions that satisfy this
constraint.
If a geometry bundle omits `rain_capture_fraction`, the public bridge assigns
precipitation only to the mutually exclusive ground and roof plan surfaces.
Canopy interception is then inactive, even though the bundle retains a leaf
storage capacity for use with an explicit partition. Consequently, a default
run with rain can have `CanopyQE = 0`; enabling physically meaningful wet-canopy
evaporation requires a documented, area-closing capture partition in the
geometry bundle.
Evaporation from ground, walls and wet leaves enters a prognostic canyon
specific-humidity budget. Above-canyon ventilation supplies or removes both
heat and water vapor; roof evaporation exchanges directly with above-roof air.

Spin-up repeats the complete forcing cycle. It stops only after the maximum
drift in all surface, material-layer and canyon-air temperatures and the
maximum drift in water storage and canyon specific humidity are below their
separate tolerances. A strict run raises an error if either the timestep or
spin-up fails to converge.

## SOLWEIG integration and input boundary

`thermal_comfort(use_coupled_eb=True)` uses two passes. The first pass retains
SOLWEIG's pedestrian shortwave calculation and direct-normal and diffuse-
horizontal components. For the coupled surfaces, the direct beam is projected
onto each physical facet normal and traced toward the sun through the saved
building and canopy height fields. Diffuse irradiance uses the unprojected
surface sky factor. Canopy interception is divided over leaf area with the
same Beer-Lambert extinction parameters used by the geometry tracer. Direct
and diffuse power are closed separately over the tile to their plan-area
inputs; this corrects the local-enclosure approximation without changing its
spatial ordering. The correction is bounded by DNI for direct irradiance and
DHI for diffuse irradiance, and does not illuminate traced shadow zeros.
Reflected shortwave is added only by the radiosity system.
The complete forcing cycle is then spun up and solved by
`CoupledUrbanEnergyBalance`.
SOLWEIG's absorbed shortwave contribution to pedestrian radiant load is kept;
its parametric ground and wall longwave contribution is replaced by outgoing
radiosities from the coupled facets. `Lup` becomes the coupled ground or roof
outgoing radiosity. Four wall-temperature rasters can be written explicitly.
Ground thermal and optical properties are expanded from the land-cover raster
for every pixel. Roofs use the selected roof preset and walls use the wall
preset and public wall emissivity.

The solver does not invent urban view factors. `coupled_geometry_path` must be
an `.npz` file for a single tile, or a directory containing
`coupled_geometry_<tile-key>.npz` files. Every bundle contains:

```
area                  float32 (7, rows, cols)
sky_view_area         float32 (7, rows, cols)
exchange_area         float32 (7, 7, rows, cols)
body_view_factor      float32 (7, rows, cols)
body_sky_view_factor  float32 (rows, cols)
raw_surface_sky_view_factor float32 (7, rows, cols)
facet_origin_x        float32 (7, rows, cols)
facet_origin_y        float32 (7, rows, cols)
facet_origin_z        float32 (7, rows, cols)
trace_dem             float32 (trace_rows, trace_cols)
trace_building_dsm    float32 (trace_rows, trace_cols)
trace_canopy_height   float32 (trace_rows, trace_cols)
trace_landcover       int16 (trace_rows, trace_cols)
trace_output_window   int64 (4,)
config_json           scalar string
crs_wkt               scalar string
geotransform          float64 (6,)
```

Optional arrays are `water_capacity` and `rain_capture_fraction`, both shaped
`(7, rows, cols)`. Facet order is ground, roof, north wall, east wall, south
wall, west wall and canopy. Exchange areas must be reciprocal and close with
sky-view area. Pedestrian view factors plus the pedestrian sky factor must sum
to one. View assigned to an inactive facet is rejected. The CRS and complete
affine transform must match the processed raster tile; a same-shaped but
shifted geometry bundle is rejected.
The trace rasters include the ray-distance halo read by `utherm-geometry`.
`trace_output_window` locates the output tile inside that halo and the facet
origins use the same local pixel coordinate system. A public coupled run fails
if this solar-trace payload is absent or incomplete. The low-level solver can
still accept a caller-supplied `UrbanForcing` without it.

`utherm-geometry` now creates this bundle directly from aligned, projected DEM,
building-DSM, canopy-height and land-cover rasters. It traces deterministic
cosine-weighted surface-normal rays from representative ground, roof,
oriented-wall and canopy facets. Pedestrian rays preserve SOLWEIG's standing-body projected-area
weights: 0.06 upward and downward and 0.22 for each cardinal side. Canopy
interception uses Beer-Lambert attenuation with configurable LAI and
extinction. The directed exchanges are
projected to an exactly reciprocal, closed exchange-area matrix before the
bundle is written. The default 128 surface rays and 256 pedestrian rays were
checked against a 256/512-ray reference on a real San Diego scene.

```bash
utherm-geometry \
  --dem DEM.tif \
  --building-dsm Building_DSM.tif \
  --canopy-height CDSM.tif \
  --landcover landcover.tif \
  --window 535,1066,64,64 \
  --output coupled_geometry.npz
```

The generator reads a ray-distance halo around the requested window, records
the input hashes, CRS, transform, trace configuration and any centimetre-scale
DSM clamping, and fails on material DSM/DEM disagreement. The stored land-
cover halo assigns each representative ground facet the class at its traced
surface origin and must match the public input tile. It does not invent
exchange from a scalar sky-view factor.

Zero-valued canopy-height nodata is interpreted as no canopy. Other canopy
nodata, and nodata in terrain, building or land-cover inputs, is an error.
The land-cover input describes the ground surface. UMEP vegetation classes 3
and 4 are rejected because they do not identify the material below the canopy.
`derive_ground_cover.py` can replace them with the nearest observed class 1,
5, 6 or 7 and records that derivation in the output raster metadata.

This is a local seven-facet enclosure at every plan cell, matching the current
solver state. A facet state can represent the nearest traced member of that
facet class within the ray-distance neighbourhood; it is not necessarily a
physical surface co-located with the output pixel. Consequently, `TLeaf`, roof
and wall temperature rasters are representative local-facet diagnostics, not
maps asserting that a leaf, roof or wall exists at every finite output cell.
The scheme is not a global sparse mesh connecting every individual wall, roof
and ground polygon across the neighbourhood. `raw_surface_sky_view_factor` is
retained for auditing before the reciprocal projection.

## Deliberate limits of this experiment

This solver is not yet a comprehensive urban climate model. Moisture is a
surface interception store with evaporation and drainage; it does not yet
include soil-water diffusion, roots, irrigation, dew, snow or phase change.
Land-cover classes select per-pixel ground columns, but spatially varying wall
and roof construction maps are not yet inputs. Anthropogenic heat is available
in the low-level forcing object but is not yet read from the public
meteorological workflow.
The building interior is a fixed lower-boundary temperature, not a prognostic
indoor zone or HVAC model. Canyon air is one vertically mixed control volume;
horizontal transport is represented by prescribed ventilation rather than a
resolved scalar field. The low-level solver accepts a wind tensor, but the
public coupled path currently receives only station wind or the reduced raster
diagnostic. Official URock remains a separate workflow and is not iterated
with this energy balance.

Those omissions and the local-enclosure approximation require explicit
qualification even after numerical and observational validation. Validation
of mean radiant temperature does not validate unobserved canopy latent heat,
canyon humidity or canyon-air-temperature diagnostics.
