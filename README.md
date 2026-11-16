# Forklift Load Volume Estimation from Video

## Overview

This project estimates the 3D volume (m³) of a load carried by a Linde forklift using a monocular RGB video.

The main challenge is estimating real-world dimensions from a 2D image without direct depth measurements. Instead of relying entirely on learned depth estimation, this approach uses the known dimensions of EUR pallets (1200 × 800 mm) as a physical reference to recover scale.

The pipeline combines pallet-based geometric measurement, crate-count validation, optional monocular depth estimation (MoGe-2), and Monte Carlo simulation to estimate volume and quantify uncertainty.

The objective is not just to produce a volume estimate, but also to identify assumptions, validate measurements across multiple frames, and understand where the estimation can fail.

## Approach

Since a single camera does not provide absolute scale, the solution uses multiple sources of geometric information:

1. **Pallet-based measurement:** Manually annotate pallet joints along the visible face. Each pallet segment provides a known physical reference of 1200 mm (or 800 mm, depending on orientation).
2. **Geometric reconstruction:** Estimate the load's length, depth, and height using pixel measurements, camera yaw, and perspective-aware local scaling.
3. **Crate-count cross-check:** Use the number of crates across the load and the number of vertical layers to obtain an independent volume estimate.
4. **Depth validation (optional):** Use MoGe-2 to obtain a metric point map and cross-check the estimated pallet depth and load geometry. The pallet dimensions remain the primary scale reference.
5. **Uncertainty estimation:** Run Monte Carlo simulations by perturbing annotation points, pallet dimensions, yaw (for single-face observations), and fill factor.

Multiple frames are evaluated, and a quality-control stage rejects measurements with excessive camera yaw or perspective distortion.

## Run Order

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Provide screenshots

Put the selected screenshots in:

```text
data/selected/
```

Alternatively, use your own screenshot directory:

```bash
python scripts/02_use_my_frames.py --src "path/to/screenshots"
```

Screenshots should be sharp, high-resolution, and preferably show the load's bottom edge and pallet joints clearly.

### 3. Annotate the load

```bash
python scripts/03_annotate.py
```

Manually annotate:
- Bottom-edge pallet joints along the main face.
- Two top corners of the load.
- Optional end-face corners for estimating yaw and depth.

The annotation tool also collects crate counts and the visible end-face orientation.

### 4. Optional segmentation check

```bash
python scripts/04_sam_check.py
```

Uses SAM 2 to check segmentation consistency against manually annotated boundaries. This is an optional validation step and is not required for the geometric measurement pipeline.

### 5. Calculate volume

```bash
python scripts/05_measure.py
```

Estimates the load's length, depth, height, and volume for each annotated frame.

It also calculates an independent crate-based estimate for comparison.

### 6. Estimate uncertainty

```bash
python scripts/07_uncertainty.py --deep 2 3
```

Compares two possible pallet-depth configurations (2 and 3 pallets deep) using Monte Carlo simulation.

The script generates separate volume distributions and 5th–95th percentile intervals for each scenario.

### 7. monocular depth validation

```bash
python scripts/06_depth_check.py
```

Uses MoGe-2 to estimate a metric point map and independently examine the load geometry and implied pallet depth.

MoGe-2 and SAM 2 require their respective additional model dependencies.

**Note:** `scripts/01_download_extract.py` is optional. It downloads the source video, extracts frames, detects shot changes, and records frame sharpness. It is not needed when screenshots are already available.

## Results

The selected load was analysed using three quality-approved frames from approximately 139.9–141.5 seconds in the video.

### Geometric Volume Estimates

| Frame timestamp | Estimated volume |
|---|---:|
| 139.92 s | 8.67 m³ |
| 140.40 s | 9.64 m³ |
| 141.48 s | 9.19 m³ |
| **Median** | **9.19 m³** |

The three measurements show reasonable consistency across the selected frames.

### Independent Crate-Count Validation

Using the assumed crate dimensions and observed crate arrangement:

- Crate footprint along the main face: 400 mm
- Assumed crate height: 310 mm
- Crates across the main face: 6
- Vertical layers: 5
- Pallets along the main face: 2

The resulting crate-based estimate is approximately **8.93 m³**, which is close to the geometric estimate.


### Uncertainty Analysis

The load configuration was established as:

- Pallets along the main face: 2
- Pallets along the depth: 3
- EUR pallet dimensions: 1200 × 800 mm

The three-pallet depth assumption is supported by the visible end face and independently cross-checked using MoGe-2.

The depth-validation script estimated the following pallet counts:

| Frame timestamp | Implied pallet depth |
|---|---:|
| 139.92 s | 3.086 |
| 140.40 s | 2.920 |
| 141.48 s | 2.702 |

These estimates support a three-pallet configuration.

Monte Carlo simulation was performed using the selected frames and the three-pallet configuration, accounting for annotation noise, pallet dimension tolerances and fill-factor variation.

| Metric | Estimated value |
|---|---:|
| Median volume | 8.90 m³ |
| 5th percentile | 7.11 m³ |
| 95th percentile | 10.64 m³ |

**Final estimated volume: approximately 8.9 m³.**

The interval represents uncertainty within the selected three-pallet configuration, rather than uncertainty over the number of pallets.


## Alternative Approaches Considered

Several alternative approaches were considered during the design process.

| Approach | Why considered | Why not primary |
|---|---|---|
| Pallet-based geometric measurement | Known physical dimensions provide a reliable scale reference. | Requires manual annotations. |
| Monocular depth (MoGe-2) | Provides approximate 3D geometry and depth information. | Absolute scale can be inaccurate and is domain-dependent. |
| 3D cuboid fitting | Uses camera geometry and visible 3D corners to reconstruct object dimensions. | Requires reliable camera intrinsics and visible geometric anchors. |
| Crate lattice counting | Exploits repeated, standardized crate dimensions to estimate dimensions independently. | Occlusion and different crate types make counting difficult. |
| YOLO/object detection | Could automate pallet and load localization. | Requires suitable labelled training/evaluation data and does not directly estimate metric volume. |
| SfM/NeRF | Could reconstruct scene geometry from multiple observations. | Adds unnecessary complexity for a mostly static camera and moving object. |

The pallet-based geometric approach was selected because the known pallet dimensions provide a direct physical scale reference. Other techniques are used as cross-checks or considered as potential extensions rather than being treated as the sole source of volume estimation.

## Quality Control

A quality gate is used to reject unreliable frame measurements.

Current thresholds:

- Maximum camera yaw: 40°.
- Maximum perspective ratio: 1.30.

Frames exceeding these thresholds are excluded from the final measurement set.

The QC report records the rejection reason for each frame, making the measurement process easier to inspect and reproduce.

## Outputs

The pipeline generates the following files inside `results/`:

| File | Description |
|---|---|
| `point_estimates.csv` | Per-frame geometric and crate-based volume estimates. |
| `qc_report.csv` | Frame acceptance/rejection decisions and reasons. |
| `depth_check.csv` | Optional MoGe-2 depth validation results. |
| `volume_hist.png` | Monte Carlo volume distributions for different pallet-depth assumptions. |
| `summary.json` | Median volume and 5th–95th percentile intervals for each scenario. |
| `overlays/` | Annotated frames showing measured geometric points and estimated dimensions. |

## Limitations and Failure Cases

- **Weak-perspective assumption:** The geometric model assumes the camera is approximately at load height, with limited perspective distortion.
- **Camera calibration:** Lens distortion is not explicitly corrected.
- **Planar load faces:** Visible load faces are assumed to be approximately planar and located at the pallet's depth.
- **Overhang:** Loads extending beyond the pallet boundaries can invalidate the pallet-based scale reference.
- **Hidden pallet depth:** The number of pallets along the depth is not always directly observable and must be verified.
- **Manual annotation:** Pallet joints and load boundaries are manually annotated. Fully automatic detection and measurement are not implemented.
- **Motion blur and occlusion:** Blurry frames, partially hidden pallet joints, and obstructed load boundaries can produce inaccurate measurements.
- **Crate assumptions:** Crate dimensions and heights can vary, affecting the independent crate-count estimate.

## Validation Status

The prototype is intended to demonstrate a physically grounded volume-estimation approach, explicit handling of uncertainty, and transparent evaluation of assumptions and failure cases.
