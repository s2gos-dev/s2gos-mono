# Vegetation Placement

Source: `packages/s2gos-generator/src/s2gos_generator/resources/vegetation.py`
Config: `packages/s2gos-generator/src/s2gos_generator/core/config/vegetation.py`
Tests:  `packages/s2gos-generator/tests/test_resources_vegetation.py`

---

## What this does

Reads a landcover raster and a DEM, then places 3-D vegetation assets (trees, shrubs, etc.)
across the scene. Each landcover pixel class maps to one or more species. Every species gets
its own density, scale range, and set of 3-D asset files. The output is a list of instances
— one per plant — with a world position, rotation, scale, and tilt.

---

## Inputs

### Raster files (loaded from disk)

| Input | Type | What it contains |
|---|---|---|
| `target_landcover` | xarray DataArray (int) | Integer class ID per pixel (e.g. ESA WorldCover: 10 = Tree cover, 20 = Shrubland) |
| `target_dem` | xarray DataArray (float) | Elevation in metres above sea level per pixel |

Both files are opened once and shared across all processing steps.

### Per-species parameters (`VegetationSpecies`)

One `VegetationSpecies` block is defined for each species assigned to a landcover class.

| Parameter | Type | Meaning |
|---|---|---|
| `name` | str | Species label (e.g. `"oak_trees"`) |
| `asset_xml_paths` | list or weighted dict | Eradiate XML asset file(s) to use. List = equal weight; dict = weighted random selection |
| `density_per_hectare` | float (0–4000) | How many instances of this species to place per hectare |
| `scale_min` / `scale_max` | float | Uniform random scale range applied to each placed instance |
| `spillover_enabled` | bool | Whether this species can bleed into adjacent landcover classes |
| `spillover_compatibility` | dict[int, float] or None | Per-species override of the global spillover compatibility map |

### Global parameters (`VegetationPlacementConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `enabled` | `True` | Master on/off switch |
| `landcover_species_mapping` | `{10: [oak]}` | Maps landcover class ID → list of `VegetationSpecies` |
| `min_spacing` | `2.0 m` | No two instances (of any species) may be closer than this |
| `density_variation` | `0.3` | Each pixel's density is scaled by `1 ± 30 %` randomly |
| `max_instances_per_pixel` | `50` | Hard cap on total instances placed in a single pixel (all species combined) |
| `rotation_range` | `360°` | Uniform random azimuth applied at finalisation |
| `tilt_range` | `6°` | Uniform random fore/aft and side tilt `±6°` applied at finalisation |
| `spillover_max_distance_m` | `30 m` | Maximum distance from the primary landcover boundary for spillover |
| `spillover_compatibility` | see config | Default map: landcover class → compatibility score (0–1) used when a species has no per-species override |

### Optional: exclusion zones

Zero or more geometries (circle, box, polygon) attached to `ctx.vegetation_exclusion_zones`.
Any instance whose position falls inside an exclusion zone is removed after all placement is done.

---

## Step-by-step walkthrough

```
process_target_vegetation(ctx)
│
├── Step 1  Guard checks
├── Step 2  Open datasets, compute pixel size
├── Step 3  Primary placement (per landcover class → per pixel → per species)
├── Step 4  Spillover placement (optional, per species)
├── Step 5  Elevation lookup
├── Step 6  Shuffle + global spacing filter
├── Step 7  Exclusion zone filter
└── Step 8  Finalise instances
```

---

### Step 1 — Guard checks

**Function**: `process_target_vegetation`

- If `vegetation_config.enabled` is `False`, return `[]` immediately.
- If `landcover_path` or `dem_path` is missing from `ctx.dependency_outputs`, raise `DataNotFoundError`.
- If either file does not exist on disk, raise `DataNotFoundError`.

Nothing is computed here. These are fast fail-early checks.

---

### Step 2 — Open datasets and compute pixel area

**Function**: `_process_vegetation_with_shared_datasets`

Both rasters are opened once as shared `xr.DataArray` objects and kept open for the full
run. Two derived values are computed from the landcover coordinate spacing:

```
y_resolution = abs(y_coords[1] - y_coords[0])   # metres per pixel (north–south)
x_resolution = abs(x_coords[1] - x_coords[0])   # metres per pixel (east–west)
pixel_area_ha = (y_resolution × x_resolution) / 10 000
```

**Why hectares?** Density is configured in instances per hectare (`density_per_hectare`).
Converting pixel area to hectares means the same species config works correctly at any
raster resolution — switching from 10 m to 30 m pixels requires no re-tuning.

---

### Step 3 — Primary placement

**Functions**: `_process_landcover_species`, `_generate_pixel_vegetation_positions`,
`_calculate_max_instances_per_pixel`

This is the main loop: for each landcover class that has at least one species configured,
and for each pixel in that class, place instances of every assigned species.

#### 3a — Find pixels for this landcover class

```python
landcover_mask = landcover_data == landcover_class   # boolean raster
y_indices, x_indices = np.where(landcover_mask)      # lists of pixel row/col indices
```

#### 3b — Per pixel, per species: how many instances?

For each pixel and each species assigned to that pixel's class:

```
base = density_per_hectare × pixel_area_ha
variation = density_variation × uniform(−1, 1)        # random ±30 % by default
instances_per_pixel = max(0, base × (1 + variation))
```

Cap by the physical spacing limit:

```
max_by_spacing = floor(x_resolution / min_spacing) × floor(y_resolution / min_spacing) × 0.75
instances_per_pixel = min(instances_per_pixel, max_by_spacing)
```

The **0.75 factor** accounts for the fact that random placement wastes roughly 25 % more
space than a perfect grid, so the cap is slightly lower than the theoretical grid maximum.

Convert the fractional count to an integer using **stochastic rounding** (preserves the
expected value of density):

```
n_base  = int(instances_per_pixel)
n_extra = 1 if random() < (instances_per_pixel − n_base) else 0
n       = n_base + n_extra
```

Example: if `instances_per_pixel = 2.7`, the pixel gets 3 plants with 70 % probability and
2 plants with 30 % probability. Using `round()` or `int()` alone would bias the totals.

#### 3c — Place instances within the pixel (rejection sampling)

**Function**: `_generate_pixel_vegetation_positions`

For each of the `n` instances to place, a random `(x, y)` position is drawn uniformly
within the pixel boundary. The position is accepted only if it is at least `min_spacing`
metres from every previously accepted position in this pixel:

```
x = uniform(center_x − half_width,  center_x + half_width)
y = uniform(center_y − half_height, center_y + half_height)
if distance(x, y, every accepted position) ≥ min_spacing → accept
```

The loop is bounded by `max_attempts = min(n × 5, 100)`. If the budget is exhausted before
all `n` instances are placed, a `logging.warning` is emitted and the pixel returns fewer
instances than requested. This happens when density and spacing settings are in tension.

Each accepted instance gets:
- `x`, `y` (scene coordinates, metres)
- `elevation = 0.0` (placeholder, filled in Step 5)
- `species` name
- `asset_xml` — one file chosen by weighted random from the species' asset list
- `scale_min`, `scale_max` (carried forward for Step 8)

#### 3d — Per-pixel cap across all species

After all species in a pixel are processed, if the combined count exceeds
`max_instances_per_pixel` (default 50), a random subset of that size is kept:

```python
if len(pixel_instances) > max_instances_per_pixel:
    pixel_instances = random.sample(pixel_instances, max_instances_per_pixel)
```

This is a performance safety valve. It prevents a single dense pixel from dominating scene
complexity when multiple high-density species share a class.

---

### Step 4 — Spillover placement (optional)

**Function**: `_process_spillover_vegetation`

Runs once per species in a landcover class, but only if `species.spillover_enabled = True`.
Spillover places instances of a species into *adjacent* landcover classes, producing natural
edge effects (e.g. forest trees thinning out into neighbouring grassland).

#### 4a — Distance from the primary class boundary

```python
primary_mask = (landcover_data == primary_class)
distance_from_primary = distance_transform_edt(~primary_mask)
# Result: every pixel's Euclidean distance (in pixels) to the nearest primary-class pixel
```

`distance_transform_edt` (scipy) computes this in O(n) time over the whole raster.

```
pixel_resolution = (x_resolution + y_resolution) / 2
max_distance_pixels = spillover_max_distance_m / pixel_resolution
```

#### 4b — Candidate pixels for each compatible target class

For each target class in the compatibility map with `compat_score > 0`:

```
candidate mask = (landcover == target_class) AND (distance_from_primary ≤ max_distance_pixels)
```

#### 4c — Vectorised density and Poisson sampling

All candidate pixels are processed in one numpy batch (no Python loop per pixel):

```
distances     = distance_from_primary[y_indices, x_indices]        # array
distance_decay = 1 − (distances / max_distance_pixels)             # linear 1 → 0
lam           = density_per_hectare × compat_score × distance_decay × pixel_area_ha
n_instances   = np.random.poisson(lam)                             # one draw per pixel
n_instances   = min(n_instances, max_instances_per_pixel)
```

Density therefore decays linearly from `density × compat_score` at the boundary to zero at
`spillover_max_distance_m`. Poisson sampling is used instead of stochastic rounding because
spillover pixels typically have sub-1 expected counts — Poisson handles the high probability
of zero instances correctly.

Only pixels where `n_instances > 0` (usually a small minority) then call
`_generate_pixel_vegetation_positions`, avoiding per-pixel Python overhead for the majority
of zero-instance pixels.

---

### Step 5 — Elevation lookup

**Function**: `_batch_elevation_lookup`

At this point every instance has `elevation = 0.0`. All x/y positions are now looked up
against the DEM in a single batch using `scipy.interpolate.RegularGridInterpolator`:

```python
interpolator = RegularGridInterpolator((y_grid, x_grid), dem_values, method="linear")
elevations = interpolator(np.column_stack([y_coords, x_coords]))
```

- **Linear interpolation** between the four surrounding DEM pixels.
- Out-of-bounds positions get `elevation = 0.0`.
- Falls back to nearest-neighbour if linear fails, then to all-zeros if that also fails.

Each instance's `elevation` field is updated in place. This single batch call replaces what
would otherwise be one xarray lookup per instance.

**Input**: list of all instances (primary + spillover combined)
**Output**: same list with `elevation` filled from DEM

---

### Step 6 — Shuffle then global spacing filter

**Functions**: `random.shuffle`, `_apply_spacing_filter_optimized`

Runs only if `min_spacing > 0` and there is more than one instance.

Within-pixel rejection sampling (Step 3c) enforces spacing *inside* a single pixel, but two
instances placed near the edges of adjacent pixels can still be closer than `min_spacing`.
This global pass enforces spacing across the entire scene.

#### 6a — Shuffle first

```python
random.shuffle(all_vegetation_instances)
```

Because positions are built landcover-class by class, then species by species, without the
shuffle the first species in the first class would systematically win all spacing conflicts.
The shuffle gives every instance an equal chance of being the "first-seen" winner.

#### 6b — Spatial grid filter

A hash-grid with cell size `min_spacing` is used so that each instance only needs to be
compared against the small number of positions in the 3×3 neighbourhood of cells:

```
grid_cell(x, y) = (int(x / min_spacing), int(y / min_spacing))

for each instance (in shuffled order):
    check 9 neighbouring cells for any existing instance closer than min_spacing
    if none found → accept and add to grid
    else          → discard
```

**First-seen wins.** Because the list was shuffled, no species has systematic priority.
**Complexity**: effectively O(n) — each instance checks only O(1) occupied neighbours.

**Input**: all instances with elevation filled
**Output**: subset with global spacing enforced

---

### Step 7 — Exclusion zone filter (optional)

**Function**: `_filter_by_exclusion_zones`

Skipped if `ctx.vegetation_exclusion_zones` is empty.

A Shapely `STRtree` spatial index is built from all exclusion zone geometries (circle, box,
or polygon). For each remaining instance, the index returns nearby zone candidates; exact
`contains` checks are run only on those candidates. Instances inside any zone are removed.

**Input**: spacing-filtered instances
**Output**: instances outside all exclusion zones

---

### Step 8 — Finalise

Back in `_process_vegetation_with_shared_datasets`, each surviving instance is converted
from its internal dict into the final output format:

```python
{
    "position":  [x, y, elevation],                         # metres, scene coordinate system
    "rotation":  uniform(0, rotation_range),                # degrees, azimuth around vertical
    "scale":     uniform(scale_min, scale_max),             # dimensionless multiplier
    "tilt_x":    uniform(−tilt_range, +tilt_range),         # degrees, fore/aft tilt
    "tilt_y":    uniform(−tilt_range, +tilt_range),         # degrees, side tilt
    "species":   species.name,                              # string label
    "asset_xml": path_to_asset_xml,                         # resolved file path
}
```

Random rotation and tilt are applied here, not earlier, so they are not affected by any
filtering steps. Scale is drawn from the per-species `[scale_min, scale_max]` range.

---

## Output

A Python `list` of dicts, one per placed instance, in the format shown in Step 8.

The list is also stored on the context as `ctx.vegetation_instances` so downstream resources
can access it without re-running placement.

Returns `[]` if vegetation is disabled, no species are configured, or all instances are
filtered out.

---

## How the output is used in simulation

Each dict in the output list is consumed by the Eradiate scene builder:

| Field | Used for |
|---|---|
| `position` | World-space translation of the asset in the scene (x, y in scene metres; z = DEM elevation) |
| `rotation` | Azimuth rotation of the asset around its vertical axis |
| `scale` | Uniform scale multiplier applied to the asset mesh |
| `tilt_x`, `tilt_y` | Small deviations from vertical to simulate natural leaning |
| `asset_xml` | Path to the Eradiate XML file that defines the plant geometry and BSDF |
| `species` | Used for logging, debugging, and optional per-species post-processing |

The assets are instanced (not duplicated) in the Eradiate scene, so placing thousands of
instances of the same asset adds only a small memory overhead per instance.

---

## Key numbers at a glance

| Parameter | Where set | Effect |
|---|---|---|
| `density_per_hectare` | per species | Average plants per 10 000 m² for that species |
| `density_variation = 0.3` | global | ±30 % random per-pixel scaling of density |
| `min_spacing = 2.0 m` | global | Minimum distance between any two plants in the scene |
| `max_instances_per_pixel = 50` | global | Hard cap per pixel (all species combined) |
| `spillover_max_distance_m = 30 m` | global | How far species can spill beyond their landcover boundary |
| `rotation_range = 360°` | global | Full azimuth randomisation |
| `tilt_range = 6°` | global | Small random lean in each axis |
| `0.75` packing factor | code constant | Discounts theoretical grid capacity by 25 % to account for random placement waste |
| `max_attempts = min(n×5, 100)` | code constant | Rejection-sampling budget per pixel before giving up and warning |