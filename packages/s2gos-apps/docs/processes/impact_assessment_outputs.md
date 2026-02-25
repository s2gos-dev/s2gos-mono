# Project Outputs and Tangible Value

## 1. Scientific Outputs

### 1.1 Physics-Based 3D Radiative Transfer Simulation Framework

The project delivers a fully coupled 3D radiative transfer simulation environment
built on the Eradiate Monte Carlo ray-tracing engine. Unlike conventional approaches
that decouple the atmosphere from the surface and treat each independently (e.g.,
MODTRAN, 6SV), this framework treats the atmosphere and the heterogeneous land
surface as a single unified radiative system. The practical consequence is that
interactions between surface structure and atmospheric scattering — effects that are
approximated or neglected in 1D codes — are captured naturally and without additional
modelling assumptions.

This unified treatment enables physically consistent simulation of Earth observation
measurements across satellite (CHIME, Sentinel-2/MSI), airborne (UAV), and
ground-based (HYPSTAR) platforms within the same radiative computation. The system
produces four distinct radiometric quantities — HDRF (Hemispherical-Directional
Reflectance Factor), BRF (Bidirectional Reflectance Factor), BHR (Bi-Hemispherical
Reflectance), and HCRF (Hemispherical-Conical Reflectance Factor) — each derived from
first principles with appropriate atmospheric coupling and white-reference
normalisation.

**Maturity level (TRL 4–5):** Core simulation capabilities have been validated in a
laboratory-equivalent environment across five Regions of Interest. End-to-end
workflows are functional and demonstrated with real-world input data, though not yet
deployed as an operational service.

**Accessibility:** Source code developed under ESA contract. The underlying Eradiate
radiative transfer model is open-source (LGPLv3). S2GOS-specific components are
currently [PLACEHOLDER — confirm licence and access terms with ESA].

**Reusability and interoperability:** The framework is sensor-agnostic by design —
new instruments require only a spectral response function and viewing geometry
definition. Simulation outputs conform to xarray/NetCDF conventions, ensuring
compatibility with standard EO analysis toolchains. The modular backend architecture
allows, in principle, the integration of alternative radiative transfer solvers
beyond Eradiate.

---

### 1.2 Seasonal Spectral-Radiometric Coupling Model

The project implements a seasonal phenology-radiation coupling approach that links
vegetation state (leaf-on/leaf-off), spectral material properties, thermophysical
atmospheric profiles (sourced from CAMS reanalysis), and snow cover to specific
observation dates. Rather than relying on static seasonal proxies, the framework
generates physically self-consistent scenes at target dates, capturing the full
radiometric impact of seasonal change: altered canopy transmittance, modified soil
exposure, shifted atmospheric composition, and altitude-dependent snow placement.

This capability has been demonstrated at the Parque Nacional Patagonia site, where
Nothofagus pumilio forests undergo pronounced seasonal variation including deciduous
leaf-drop and winter snow cover. Summer and winter scenes are constructed from the
same underlying terrain and landcover, with seasonal differences arising entirely
from physically motivated changes to material properties, vegetation structure, and
atmospheric state.

**Maturity level (TRL 4):** Demonstrated for summer and winter conditions at the PNP
site using TLS-derived 3D tree assets with seasonal variants. Extension to arbitrary
dates and continuous phenological transitions remains future work.

**Accessibility:** Integrated within the S2GOS generation pipeline. Seasonal material
and asset libraries are bundled with the generator package.

**Reusability and interoperability:** The approach is transferable to other biomes
provided that appropriate seasonal material definitions and vegetation assets are
available. Configuration is data-driven (JSON material databases, YAML scene
descriptions), making adaptation straightforward without code modification.

---

### 1.3 Pixel-Level Radiometric Reference Generation

The framework introduces a pixel-level mapping between satellite image pixels and the
underlying 3D scene, enabling sub-pixel computation of HDRF, BRF, and BHR for
individual satellite footprints. This goes beyond conventional point-based or
area-averaged comparisons: the system produces spatially explicit radiometric
reference values that can be directly matched against satellite-derived surface
reflectance products, pixel by pixel. For calibration and validation workflows, this
represents a meaningful step toward closing the gap between what a ground station
measures and what a satellite reports.

**Maturity level (TRL 3–4):** The methodology is implemented and functional.
Systematic validation against real satellite products has not yet been completed.

**Accessibility:** Available as configurable measurement types (PixelHDRF, PixelBRF,
PixelBHR) within the simulation configuration system.

**Reusability and interoperability:** Applicable to any gridded satellite product. The
pixel-to-scene coordinate mapping is generalised and not tied to a specific sensor or
resolution.

---

## 2. Technical Outputs

### 2.1 S2GOS Software Suite

The project produces a modular Python software suite comprising three core packages,
each addressing a distinct stage of the simulation pipeline.

The **s2gos-generator** automates 3D scene construction from geospatial data sources.
Starting from Copernicus GLO-30 DEM (30 m) and ESA WorldCover 2021 landcover (10 m),
it produces terrain meshes, assigns spectral materials based on landcover class,
places vegetation stochastically according to species-specific density models, and
configures atmospheric layers. The output is a complete, self-contained scene
description ready for simulation.

The **s2gos-simulator** provides an orchestration layer for the Eradiate radiative
transfer backend. It translates high-level sensor and observation specifications into
Eradiate-native configurations, manages Monte Carlo simulation execution, and handles
post-processing — including instrument-specific operations such as HYPSTAR circular
FOV masking, spectral response function convolution, and RGB composite generation.

A shared **s2gos-utils** package supplies the underlying infrastructure: coordinate
transformations, scene description data models, path resolution (supporting both
local and cloud storage via S3), and configuration management.

The suite supports a multi-scale scene hierarchy — a target area at full resolution
surrounded by buffer and background regions at progressively coarser resolutions — to
balance computational cost against radiative accuracy near scene boundaries.

**Maturity level (TRL 5):** The software has been exercised across five
geographically and ecologically diverse ROIs (San Rossore, Parque Nacional Patagonia,
Gobabeb, Kairouan, and Rome-Frascati), demonstrating generality well beyond a single
test case.

**Accessibility:** Developed on GitHub (`s2gos-dev` organisation) as three
independent Python packages installable via pip in editable mode. Environment
management uses Pixi (conda-based) for full reproducibility.

**Reusability and interoperability:** The clean separation between generation,
simulation, and application layers ensures that each package can evolve independently.
All three expose well-defined APIs through Pydantic-validated configuration models.
The suite is designed to run both locally and on cloud infrastructure, including the
ESA DestinE Platform.

---

### 2.2 OGC-API Process Workflows

All major workflows — scene generation, radiative transfer simulation, and combined
end-to-end pipelines — are exposed as OGC-API Processes through the
Procodile/Wraptile framework. This wraps the scientific computations in a
standards-compliant service interface that can be discovered, described, and invoked
through RESTful endpoints, bringing the simulation capabilities closer to operational
use.

Implemented processes include site-specific workflows for each ROI (Gobabeb PICS
calibration, PNP multi-temporal analysis, Frascati heterogeneity testing) alongside
generic generation and simulation processes that accept arbitrary configuration
inputs. This dual approach provides both ease-of-use for common workflows and full
flexibility for custom analyses.

**Maturity level (TRL 4–5):** Processes are functional and have been tested via local
service instances and JupyterLab-based interfaces.

**Accessibility:** Processes are registered through a central Python registry and can
be invoked via CLI, REST API, or programmatically from notebooks.

**Reusability and interoperability:** Conformance with the OGC-API Processes standard
enables integration with third-party workflow orchestration platforms and the broader
DestinE service ecosystem. Process inputs and outputs are described via JSON Schema,
supporting automated client generation.

---

### 2.3 Multi-Sensor Observation Configuration Library

The project delivers a comprehensive sensor configuration library supporting over
twenty distinct observation setups across three platform types. Satellite
configurations cover CHIME (hyperspectral, 380–2500 nm) and Sentinel-2/MSI (12
bands), both with configurable viewing geometry and spectral response function
convolution. Ground-based configurations include the HYPSTAR hyperspectral
radiometer, perspective cameras, pyranometers, and downward-looking hemispherical
(DHP) cameras — with support for asset-relative positioning (e.g., instruments
mounted on a tower mast), circular FOV masking, and multi-angular measurement
sequences. UAV configurations allow arbitrary altitude and viewing angle for
perspective cameras and custom sensor definitions.

**Maturity level (TRL 4–5):** CHIME and Sentinel-2 configurations have been validated
against known spectral response functions. HYPSTAR configurations include comparison
against real HYPERNETS L2A reference data from the Gobabeb site.

**Accessibility:** All sensor configurations are defined as Pydantic models with full
parameter validation and sensible default values.

**Reusability and interoperability:** New sensors require only spectral response
functions, viewing geometry, and platform type. The configuration schema is
serialisable to JSON, enabling non-programmatic specification of observation setups
by users who do not interact directly with the Python API.

---

### 2.4 Geospatial Input Data Pipeline

The generator package implements a cloud-ready data ingestion pipeline covering the
full range of inputs required for synthetic scene construction. DEM data is sourced
from Copernicus GLO-30 via cloud-optimised Zarr from the DestinE Earth Data Hub.
Landcover comes from ESA WorldCover 2021 via spatially indexed GeoTIFF with lazy
loading. Spectral material properties are managed through configurable JSON databases
with PROSPECT leaf model integration. Atmospheric profiles draw on CAMS reanalysis
thermophysical data in NetCDF time series format. 3D vegetation assets are
TLS-scanned tree models stored in Mitsuba 3 XML format, with seasonal variants for
deciduous species.

**Maturity level (TRL 5):** Data pipelines are operational and have been tested with
real data sources at all five ROIs.

**Accessibility:** Configured through a YAML settings file (`s2gos_settings.yaml`).
Remote data sources require appropriate credentials (e.g., Earth Data Hub access).

**Reusability and interoperability:** The data source abstraction layer — supporting
Zarr, indexed GeoTIFF, and NetCDF — is independent of the simulation framework and
could serve other applications requiring multi-source geospatial data access. Cloud
storage is supported natively via `upath` and `s3fs`.

---

## 3. Methodological Outputs

### 3.1 End-to-End Synthetic Scene Simulation Methodology

The project establishes a complete methodology for generating physically realistic
synthetic Earth observation data, from geographic coordinates to instrument-specific
data products.

The process begins with scene construction. Geographic coordinates and spatial extent
define a region of interest, from which DEM and landcover data are automatically
retrieved and processed into a multi-resolution terrain mesh. Materials are assigned
per-pixel based on landcover class, with spectral properties drawn from a curated
database. Vegetation is placed stochastically according to species-specific density
models derived from landcover classification, using TLS-scanned 3D assets that
preserve realistic canopy structure.

The constructed scene then enters the observation simulation phase. It is coupled with
a heterogeneous atmosphere — molecular profiles plus aerosol particle layers — and
illuminated according to the specified date, time, and solar geometry. One or more
sensors observe the scene from configurable platforms and viewing angles. The Eradiate
Monte Carlo solver computes the full radiative transfer, and results are
post-processed into instrument-specific data products.

What distinguishes this methodology is its multi-platform consistency. Ground-based,
airborne, and satellite observations of the same scene are computed within a single
radiative framework, ensuring physical consistency across observation scales. This is
particularly valuable for upscaling studies and cross-calibration analyses, where
assumptions about scale-dependent representativeness are otherwise difficult to verify
with real observations alone.

**Maturity level (TRL 4–5):** The methodology has been demonstrated end-to-end at
multiple sites. The step from demonstration to routine application will require
further automation and performance optimisation.

**Accessibility:** Documented through example Jupyter notebooks, MkDocs-generated
documentation, and process-level docstrings.

**Reusability and interoperability:** The methodology is site- and sensor-independent.
New use cases can be configured through YAML and JSON files without modifying source
code, and the OGC-API process interface allows the methodology to be invoked as a
service.

---

### 3.2 Multi-Temporal Radiometric Analysis Approach

The MTR (Multi-Temporal Radiometric) demonstration establishes a methodology for
assessing how seasonal environmental changes propagate through the observation
chain — from surface properties, through the atmosphere, to the final measured signal
at the sensor.

By simulating the same scene at two seasonal extremes (austral summer and winter at
Parque Nacional Patagonia), the approach isolates the radiometric impact of individual
factors: vegetation phenology (leafed vs. bare canopies), snow cover presence and
extent, atmospheric profile changes (temperature, humidity, aerosol loading), and
solar geometry variation. This provides a controlled experimental framework that would
be difficult to achieve with real observations, where confounding factors cannot be
independently varied.

**Maturity level (TRL 3–4):** Demonstrated for two-season comparison at one site.
Extension to continuous temporal sampling and additional environmental variables is
feasible but not yet implemented.

**Accessibility:** Implemented as dedicated OGC-API processes (`mtr_demo/generation`,
`mtr_demo/simulation`) with month and observation type as primary input parameters.

**Reusability and interoperability:** The temporal comparison framework is
transferable to any site where seasonal material and asset variants are available. It
could support pre-launch mission studies (e.g., CHIME, planned for 2028) and
long-term surface monitoring applications.

---

### 3.3 Cross-Scale Validation Strategy

The framework's ability to simulate ground-based (HYPSTAR), airborne (UAV), and
satellite (CHIME, Sentinel-2) observations of the same 3D scene within a unified
radiative transfer computation provides the basis for a cross-scale validation
strategy. Simulated ground measurements can be compared against real HYPERNETS
network data, and the same simulation simultaneously produces the corresponding
satellite observation. This closes the loop between field measurement and satellite
product without the assumptions inherent in empirical upscaling — a persistent
challenge in the calibration and validation community.

**Maturity level (TRL 3):** The infrastructure for cross-scale comparison is in
place. Systematic validation campaigns comparing simulated measurements against real
HYPERNETS and Sentinel-2 data are [PLACEHOLDER — confirm current status of validation
exercises].

**Accessibility:** Requires co-located ground and satellite data for the validation
sites. HYPERNETS L2A reference data from Gobabeb is already integrated as a
validation reference.

**Reusability and interoperability:** The strategy is applicable to any PICS or
HYPERNETS site. It could directly support the validation of upcoming missions such as
CHIME and contribute to the broader CEOS calibration and validation framework.

---

## 4. Dissemination Outputs

### 4.1 Publications

[PLACEHOLDER — list peer-reviewed publications, preprints, and technical notes
produced during the project. Consider including any submitted or published papers on
the S2GOS framework, the Eradiate reference paper (EGUsphere preprint, 2025) if
co-authored or directly related, and ESA technical notes or deliverables produced
under the contract.]

---

### 4.2 Conference Contributions

[PLACEHOLDER — list conference presentations, posters, and workshops. Known or likely
venues include the ESA Living Planet Symposium 2025 (LPS25), where a poster entitled
"Development of a General-Purpose Multi-Scale 3D Synthetic Scene Generator for
Simulation and Analysis" was presented, as well as any contributions to EGU, AGU,
IGARSS, ISPRS, ESA DTE programme workshops, DestinE stakeholder events, or
HYPERNETS/LANDHYPERNET community workshops.]

---

### 4.3 Interactive Demonstration Materials

The project includes a set of Jupyter notebooks that serve both as executable
documentation and as demonstration materials for stakeholder engagement. These cover
the full workflow from scene generation through multi-sensor simulation, and include
an interactive GUI built with Panel and ipyleaflet for site selection and parameter
configuration. The notebooks are designed to be self-explanatory: a new user with
access to the software environment can follow them from start to finish without
external guidance.

**Maturity level (TRL 5):** Notebooks are functional and have been used in live
demonstrations.

**Accessibility:** Included in the project repository under `example/`. Execution
requires the S2GOS software environment.

**Reusability and interoperability:** Notebooks run on any system with the Pixi
environment installed, including JupyterHub deployments on the DestinE Platform.

---

### 4.4 Technical Documentation

[PLACEHOLDER — list documentation deliverables such as the Software User Manual,
Algorithm Theoretical Basis Document (ATBD), Validation Plan and Report, and any ESA
milestone deliverables (SRR, PDR, CDR documentation).]
