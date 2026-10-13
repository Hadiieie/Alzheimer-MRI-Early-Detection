# Alzheimer MRI Early-Detection Pipeline

A research-oriented machine-learning pipeline for **binary early Alzheimer/dementia detection from structural brain MRI slices** using deterministic preprocessing, whole-brain masking, handcrafted imaging features, leakage-aware evaluation, a frozen SVM classifier, reproducibility checks, and an OASIS-1 challenge-set domain-shift study.

> **Research use only.** This repository is not a medical device and must not be used for clinical diagnosis, treatment decisions, or patient management.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Objective](#research-objective)
- [Dataset](#dataset)
- [Primary Binary Task](#primary-binary-task)
- [Pipeline](#pipeline)
- [Brain Mask V4](#brain-mask-v4)
- [Engineered Features](#engineered-features)
- [Leakage Control and Data Split](#leakage-control-and-data-split)
- [Model Development](#model-development)
- [Frozen Final Model](#frozen-final-model)
- [Final Internal Test Results](#final-internal-test-results)
- [Bootstrap Robustness](#bootstrap-robustness)
- [Subgroup Performance](#subgroup-performance)
- [Explainability and Error Analysis](#explainability-and-error-analysis)
- [Inference Workflow](#inference-workflow)
- [Inference Reproducibility](#inference-reproducibility)
- [OASIS-1 Challenge-Set Evaluation](#oasis-1-challenge-set-evaluation)
- [External Domain Shift](#external-domain-shift)
- [Harmonization Pilot](#harmonization-pilot)
- [FreeSurfer Volume-Source Ablation](#freesurfer-volume-source-ablation)
- [OASIS V2 Development Status](#oasis-v2-development-status)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Data Availability](#data-availability)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Project Overview

This project investigates whether a classical machine-learning pipeline can separate **normal MRI slices** from **early dementia MRI slices** using a transparent, reproducible workflow rather than an end-to-end deep-learning model.

The development process includes:

1. dataset auditing,
2. whole-brain segmentation,
3. deterministic handcrafted feature extraction,
4. duplicate/near-duplicate leakage control,
5. group-aware train/validation/test splitting,
6. baseline model comparison,
7. hyperparameter tuning,
8. threshold optimization on validation data only,
9. one-time frozen test evaluation,
10. bootstrap uncertainty estimation,
11. subgroup and error analysis,
12. explainability,
13. frozen single-image inference,
14. external challenge testing on OASIS-1 / FreeSurfer data,
15. domain-shift analysis.

The final internal model is a **scaled RBF-kernel Support Vector Machine using all 30 engineered features**.

---

## Research Objective

The primary research task is:

> **Normal vs Early Demented MRI classification**

The project intentionally excludes the `ModerateDemented` class from the primary early-detection task so that the classifier focuses on the more difficult distinction between normal scans and earlier disease categories.

The binary mapping used in this project is:

| Original class | Binary target |
|---|---|
| `NonDemented` | `Normal` |
| `VeryMildDemented` | `Early_Demented` |
| `MildDemented` | `Early_Demented` |
| `ModerateDemented` | Excluded from the primary task |

This mapping is a **project-specific research definition** and should not be interpreted as a clinical diagnostic equivalence.

---

## Dataset

The audited source dataset contains **39,538 MRI images** across four original classes.

| Class | Images |
|---|---:|
| NonDemented | 9,984 |
| VeryMildDemented | 9,983 |
| MildDemented | 9,886 |
| ModerateDemented | 9,685 |
| **Total** | **39,538** |

Images were processed as 2D MRI slices at approximately **220 × 220 pixels**.

The final Brain Mask V4 audit passed:

**39,538 / 39,538 images**

with class counts preserved.

---

## Primary Binary Task

After excluding `ModerateDemented`, the primary early-detection dataset contains:

| Binary class | Images | Percentage |
|---|---:|---:|
| Early_Demented | 19,869 | 66.56% |
| Normal | 9,984 | 33.44% |
| **Total** | **29,853** | **100%** |

The `Early_Demented` class consists of:

- 9,983 `VeryMildDemented`
- 9,886 `MildDemented`

---

## Pipeline

The high-level workflow is:

```text
Raw MRI images
      |
      v
Dataset audit
      |
      v
Whole-brain masking
(Brain Mask V4)
      |
      v
30 handcrafted MRI features
      |
      v
Duplicate / near-duplicate
leakage grouping
      |
      v
Group-aware split
Train / Validation / Test
      |
      v
Scaling + model comparison
      |
      v
Hyperparameter tuning
      |
      v
Validation-only threshold selection
      |
      v
Frozen SVM-RBF model
      |
      v
One-time internal test
      |
      +--> Bootstrap robustness
      +--> Subgroup analysis
      +--> Error analysis
      +--> Explainability
      |
      v
Frozen inference pipeline
      |
      v
OASIS-1 challenge-set study
      |
      v
Domain-shift analysis
```

---

## Brain Mask V4

Several masking iterations were evaluated before freezing the final version.

Earlier versions were treated as development experiments. **Brain Mask V4** became the final preprocessing mask after passing the full dataset audit.

Full-dataset mask statistics included approximately:

- mean brain ratio: **0.4414**
- standard deviation: **0.0496**
- minimum: **0.3133**
- median: **0.4325**
- maximum: **0.5869**

Approximate class-level mean brain ratios:

| Class | Mean brain ratio |
|---|---:|
| NonDemented | 0.4643 |
| VeryMildDemented | 0.4468 |
| MildDemented | 0.4309 |
| ModerateDemented | 0.4232 |

The whole-brain approach was retained rather than using a manually selected ROI.

---

## Engineered Features

Feature extraction completed successfully for all **39,538 images**:

- requested: 39,538
- extracted: 39,538
- failures: 0
- numerical features: 30
- NaN values: 0
- infinite values: 0
- constant features: 0

The final feature set is:

| # | Feature |
|---:|---|
| 1 | `brain_area` |
| 2 | `brain_ratio` |
| 3 | `brain_perimeter` |
| 4 | `compactness` |
| 5 | `bbox_aspect_ratio` |
| 6 | `mean_intensity` |
| 7 | `std_intensity` |
| 8 | `median_intensity` |
| 9 | `entropy` |
| 10 | `p10` |
| 11 | `p25` |
| 12 | `p75` |
| 13 | `p90` |
| 14 | `edge_density` |
| 15 | `central_mean` |
| 16 | `central_std` |
| 17 | `central_median` |
| 18 | `central_entropy` |
| 19 | `central_dark_ratio` |
| 20 | `central_bright_ratio` |
| 21 | `left_mean` |
| 22 | `right_mean` |
| 23 | `left_dark_ratio` |
| 24 | `right_dark_ratio` |
| 25 | `intensity_asymmetry` |
| 26 | `dark_ratio_asymmetry` |
| 27 | `glcm_contrast` |
| 28 | `glcm_homogeneity` |
| 29 | `glcm_energy` |
| 30 | `glcm_correlation` |

These features capture:

- whole-brain geometry,
- intensity distribution,
- central-brain statistics,
- left/right asymmetry,
- edge structure,
- GLCM texture.

A reduced 21-feature set was also tested after correlation filtering on the training set only, but the full 30-feature representation performed better in final model selection.

---

## Leakage Control and Data Split

Potential duplicates / near-duplicates were organized into **leakage groups** before splitting.

Results:

- unique leakage groups: **28,466**
- groups containing both binary targets: **0**
- duplicate filenames: **0**

The split was performed at the **group level**, not the individual-image level.

### Final split

| Split | Images | Approx. share |
|---|---:|---:|
| Train | 20,891 | 69.98% |
| Validation | 4,482 | 15.01% |
| Test | 4,480 | 15.01% |

### Train

- Early_Demented: 13,905
- Normal: 6,986
- unique leakage groups: 19,926

Original classes:

- NonDemented: 6,986
- VeryMildDemented: 6,963
- MildDemented: 6,942

### Validation

- Early_Demented: 2,983
- Normal: 1,499
- unique leakage groups: 4,270

### Test

- Early_Demented: 2,981
- Normal: 1,499
- unique leakage groups: 4,270

### Leakage audit

```text
Train <-> Validation overlap: 0
Train <-> Test overlap:       0
Validation <-> Test overlap:  0
```

---

## Model Development

Two main feature configurations were compared:

- full 30-feature set,
- reduced 21-feature set.

Correlation reduction was performed **using training data only** with a threshold of `0.95`.

The reduced set removed:

```text
brain_ratio
median_intensity
p25
central_median
left_mean
right_mean
left_dark_ratio
right_dark_ratio
glcm_energy
```

Baseline experiments included Random Forest and SVM models.

The strongest validation model was the **RBF SVM using all 30 features**.

### Validation comparison

| Model | Validation accuracy | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| SVM-RBF, 30 features | 0.8179 | **0.8071** | **0.8883** | **0.9291** |
| SVM-RBF, 21 features | 0.8061 | 0.7963 | 0.8501 | 0.8761 |
| Random Forest variants | ~0.82 | ~0.78 | — | — |

The 30-feature SVM was selected and frozen.

---

## Frozen Final Model

The final classifier is a scikit-learn pipeline containing:

```text
scaler -> SVM classifier
```

The model exposes a `decision_function`.

The validation-set decision threshold was optimized **before final test evaluation**.

### Frozen threshold

```text
0.10872472921801979
```

At this threshold on validation data:

| Metric | Value |
|---|---:|
| Accuracy | 0.8130 |
| Balanced Accuracy | 0.8114 |
| Sensitivity | 0.8163 |
| Specificity | 0.8065 |
| Precision | 0.8936 |
| F1 | 0.8532 |

Validation confusion matrix:

```text
TN = 1209
FP = 290
FN = 548
TP = 2435
```

Once selected, both the model and threshold were treated as frozen for the internal test.

### Frozen model artifacts

```text
results/FINAL_MODEL/final_model.joblib
results/FINAL_MODEL/final_scaler.joblib
results/FINAL_MODEL/final_features.txt
results/FINAL_MODEL/final_model_config.json
results/FINAL_MODEL/freeze_manifest.json
```

Recorded model SHA-256:

```text
d6777df6cea0e1c388fb182c1ee8d03bf8ce3fb711286534b73339b513927264
```

---

## Final Internal Test Results

The frozen model was evaluated once on the **4,480-image held-out test set**.

| Metric | Final test |
|---|---:|
| Accuracy | **0.812054** |
| Balanced Accuracy | **0.810185** |
| Sensitivity / Recall | **0.815834** |
| Specificity | **0.804536** |
| Precision | **0.892477** |
| F1 score | **0.852436** |
| ROC-AUC | **0.887921** |
| PR-AUC | **0.929306** |

### Confusion matrix

```text
                Predicted
              Normal  Early
Actual Normal   1206    293
Actual Early     549   2432
```

Equivalent counts:

```text
TN = 1206
FP = 293
FN = 549
TP = 2432
```

These numbers are the primary internal performance results for the frozen V1 model.

---

## Bootstrap Robustness

A **group-aware bootstrap with 2,000 resamples** was used to estimate uncertainty on the frozen internal test set.

Approximate 95% confidence intervals:

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Accuracy | 0.8121 | 0.7998 – 0.8236 |
| Balanced Accuracy | 0.8102 | 0.7974 – 0.8222 |
| Sensitivity | 0.8158 | 0.8013 – 0.8307 |
| Specificity | 0.8045 | 0.7830 – 0.8239 |
| Precision | 0.8925 | 0.8799 – 0.9044 |
| F1 | 0.8524 | 0.8421 – 0.8626 |
| ROC-AUC | 0.8879 | 0.8771 – 0.8982 |
| PR-AUC | 0.9293 | 0.9190 – 0.9384 |

---

## Subgroup Performance

The final test set was also examined by original class.

### MildDemented

- n = 1,444
- correctly identified = 1,322
- errors = 122
- sensitivity = **0.9155**
- false-negative rate = **8.45%**

### VeryMildDemented

- n = 1,537
- correctly identified = 1,110
- errors = 427
- sensitivity = **0.7222**
- false-negative rate = **27.78%**

### NonDemented

- n = 1,499
- true negatives = 1,206
- false positives = 293
- specificity = **0.8045**
- false-positive rate = **19.55%**

The lower sensitivity in `VeryMildDemented` highlights the difficulty of the earliest disease stage and is an important limitation of the current system.

---

## Explainability and Error Analysis

Permutation importance was performed on the frozen pipeline.

Among the most influential features were:

1. `p10`
2. `central_median`
3. `central_bright_ratio`
4. `central_std`
5. `glcm_contrast`
6. `median_intensity`
7. `glcm_homogeneity`

Representative examples were also selected for:

- high-confidence true positive,
- borderline true positive,
- high-confidence true negative,
- high-confidence false positive,
- borderline false positive,
- high-confidence false negative,
- borderline false negative.

This analysis is intended to help understand **how the model behaves and where it fails**, not to provide clinical explanations.

---

## Inference Workflow

The final inference path is implemented through:

```text
Scripts/inference_core.py
Scripts/predict_mri.py
Scripts/verify_inference_reproducibility.py
```

Conceptually, prediction follows:

```text
Input MRI image
    |
    v
Brain Mask V4
    |
    v
30-feature extraction
    |
    v
Frozen scaler + SVM
    |
    v
decision_function score
    |
    v
Compare with frozen threshold
0.10872472921801979
    |
    +--> score >= threshold -> Early_Demented
    |
    +--> score < threshold  -> Normal
```

Inference QC checks include:

- valid Brain Mask V4 result,
- all 30 expected features present,
- zero NaN values,
- zero infinite values.

---

## Inference Reproducibility

The standalone inference implementation was compared against known pipeline outputs.

Result:

```text
12 / 12 reproducibility cases: PASS
```

Largest observed numerical feature difference:

```text
~2.84e-14
```

This is effectively floating-point numerical noise and indicates that the deployment-style inference implementation reproduces the development pipeline.

---

## OASIS-1 Challenge-Set Evaluation

A small OASIS-1 / FreeSurfer pilot was used to challenge the frozen pipeline outside its original data domain.

This experiment should be interpreted as a **challenge-set / domain-shift study**, not as definitive independent external validation.

### OASIS mapping used for the project

```text
CDR 0   -> Normal
CDR 0.5 -> Early_Demented
CDR 1   -> Early_Demented
```

Again, this mapping is specific to the project and is not a statement of clinical equivalence.

### FreeSurfer preprocessing

For available OASIS subjects, `brain.mgz` was used as the skull-stripped volume source.

The volume orientation was inspected and a deterministic central axial slice was extracted at approximately **50% of the occupied brain height**.

### Pilot accounting

The original pilot manifest contained 20 subjects.

- 19 FreeSurfer subjects were available locally.
- 1 subject was missing from the available archive.
- 1 of the 19 available `brain.mgz` cases failed baseline Brain Mask V4 ratio QC.
- 18 cases therefore completed frozen baseline inference.

The 18 successfully inferred cases consisted of:

- 9 Normal
- 9 Early_Demented

### Frozen V1 threshold on OASIS

| Metric | OASIS pilot |
|---|---:|
| Accuracy | 0.5000 |
| Balanced Accuracy | 0.5000 |
| Sensitivity | 1.0000 |
| Specificity | 0.0000 |
| Precision | 0.5000 |
| F1 | 0.6667 |
| ROC-AUC | 0.6914 |
| PR-AUC | 0.7355 |

Confusion matrix:

```text
TN = 0
FP = 9
FN = 0
TP = 9
```

All 18 successfully processed OASIS cases were classified as `Early_Demented` at the original frozen threshold.

This is strong evidence that the internal decision threshold **did not transfer across domains**.

---

## External Domain Shift

The OASIS study revealed a substantial positive shift in model decision scores.

### Internal test decision scores

#### Early_Demented

- n = 2,981
- mean = **1.0788**
- median = **1.0920**
- minimum = -3.2381
- maximum = 4.8879

#### Normal

- n = 1,499
- mean = **-0.9591**
- median = **-0.9924**
- minimum = -4.4845
- maximum = 3.6662

### OASIS `brain.mgz` decision scores

#### Early_Demented

- n = 9
- mean = **1.3390**
- median = **1.3413**
- range ≈ 1.0342 – 1.6992

#### Normal

- n = 9
- mean = **1.1430**
- median = **1.1970**
- range ≈ 0.6770 – 1.5989

The OASIS Normal score distribution moved from strongly negative internally to strongly positive externally. This explains why the frozen internal threshold generated many false positives despite retaining some ranking signal.

### Largest raw feature shifts

Selected standardized mean shifts between internal data and the OASIS pilot included:

| Feature | Approx. z-shift |
|---|---:|
| `p90` | -4.34 |
| `brain_area` | -3.91 |
| `brain_ratio` | -3.91 |
| `glcm_correlation` | -3.77 |
| `std_intensity` | -3.20 |
| `p75` | -3.10 |
| `brain_perimeter` | -2.87 |
| `edge_density` | +2.56 |
| `mean_intensity` | -2.40 |
| `bbox_aspect_ratio` | -2.35 |
| `median_intensity` | -2.25 |
| `central_std` | -2.20 |

These differences are consistent with a substantial acquisition / processing / representation domain mismatch.

---

## Harmonization Pilot

A development-only harmonization experiment was performed on the same repeatedly inspected OASIS pilot cases.

### Baseline

- successful cases: 18
- mean absolute feature z-shift: **1.5793**
- balanced accuracy: **0.5000**
- ROC-AUC: **0.6914**

### Spatial-only harmonization

- successful cases: 19
- mean absolute feature z-shift: **1.1992**
- balanced accuracy: **0.5500**
- sensitivity: 1.0000
- specificity: 0.1000
- ROC-AUC: **0.5889**

### Spatial + intensity harmonization

- successful cases: 19
- mean absolute feature z-shift: **0.5186**
- balanced accuracy: **0.4278**
- sensitivity: 0.5556
- specificity: 0.3000
- ROC-AUC: **0.3778**

Although feature-distribution mismatch was numerically reduced, discrimination became worse.

Therefore, this harmonization strategy was **not adopted**.

This result is also a reminder that minimizing distribution distance does not automatically improve classification validity.

---

## FreeSurfer Volume-Source Ablation

Different FreeSurfer-derived volume representations were compared on the same development subjects.

| Source | Successful | Balanced Acc. | Sensitivity | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `brain.mgz` | 18 | 0.5000 | 1.0000 | 0.0000 | 0.6914 | 0.7355 |
| `T1.mgz` masked | 18 | 0.5000 | 1.0000 | 0.0000 | **0.8025** | **0.8227** |
| `nu.mgz` masked | 18 | 0.5556 | 1.0000 | 0.1111 | 0.6543 | 0.7625 |

An important observation is that **masked T1 produced the strongest ranking performance by ROC-AUC and PR-AUC**, even though the original frozen threshold still transferred poorly.

Therefore, source selection and threshold transfer should be treated as separate issues.

---

## OASIS V2 Development Status

The original V1 model and threshold remain frozen.

A separate **OASIS V2 development path** is being explored for domain-specific calibration.

Current principles:

- V1 is not modified retrospectively.
- The existing OASIS pilot cases are now development/calibration cases because they have been repeatedly inspected.
- Threshold calibration on these cases is therefore **not an unbiased external test**.
- Any V2 candidate must be frozen before evaluation on new untouched subjects.
- A genuinely new held-out OASIS set is required for the next honest assessment.
- Dataset provenance and possible subject overlap with the original source dataset should be investigated before making strong claims of independent external validation.

For this reason, this repository currently describes the OASIS experiment as a:

> **challenge-set / pilot external-domain evaluation**

rather than definitive independent external validation.

---

## Repository Structure

A simplified repository layout is:

```text
Alzheimer-MRI-Early-Detection/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- Scripts/
|   |-- dataset_audit.py
|   |-- dataset_analysis.py
|   |-- brain_mask_pilot.py
|   |-- brain_mask_pilot_v2.py
|   |-- brain_mask_pilot_v3.py
|   |-- brain_mask_pilot_v4.py
|   |-- build_primary_early_detection_dataset.py
|   |-- build_leakage_groups.py
|   |-- baseline_models.py
|   |-- inference_core.py
|   |-- predict_mri.py
|   |-- verify_inference_reproducibility.py
|   |-- diagnose_oasis_mask_failures.py
|   |-- analyze_oasis_domain_shift.py
|   |-- oasis_harmonization_pilot.py
|   |-- oasis_volume_source_ablation.py
|   `-- ...
|
`-- results/
    |-- FINAL_MODEL/
    |-- FINAL_PROJECT/
    |-- TRAINING_PREPARATION/
    |-- BASELINE_MODELS/
    |-- HYPERPARAMETER_TUNING/
    |-- OASIS_FREESURFER_EVALUATION/
    |-- OASIS_DOMAIN_SHIFT/
    |-- OASIS_HARMONIZATION_PILOT/
    |-- OASIS_VOLUME_SOURCE_ABLATION/
    `-- ...
```

The repository contains the research code and selected result artifacts, while large MRI datasets and FreeSurfer volumes are intentionally excluded.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hadiieie/Alzheimer-MRI-Early-Detection.git
cd Alzheimer-MRI-Early-Detection
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Core packages used by the project include:

```text
numpy
pandas
matplotlib
opencv-python
scikit-learn
scipy
scikit-image
joblib
nibabel
```

---

## Running the Project

Because this repository documents the full research pipeline, scripts are separated by stage.

### Verify frozen inference reproducibility

```bash
python Scripts/verify_inference_reproducibility.py
```

Expected project result:

```text
12 / 12 PASS
```

### Run the single-MRI inference script

```bash
python Scripts/predict_mri.py
```

The script uses the frozen inference implementation in:

```text
Scripts/inference_core.py
```

and the frozen model artifacts under:

```text
results/FINAL_MODEL/
```

If local paths are configured differently, update the relevant path variables or arguments in the script before execution.

### OASIS challenge workflow

The OASIS-related workflow is represented by scripts such as:

```text
diagnose_oasis_mask_failures.py
extract_oasis_mid_axial.py
extract_oasis_freesurfer_batch_v2.py
evaluate_oasis_freesurfer_batch_v2.py
analyze_oasis_domain_shift.py
oasis_harmonization_pilot.py
oasis_volume_source_ablation.py
oasis_threshold_calibration.py
```

These scripts require locally obtained OASIS / FreeSurfer files and are not expected to run without those data.

---

## Data Availability

The MRI datasets are **not redistributed in this repository**.

Excluded data include:

- raw training MRI images,
- OASIS FreeSurfer subject directories,
- `.mgz` volumes,
- large compressed archives,
- bulk generated QC imagery.

Users must obtain any required source datasets from their original providers and comply with the corresponding licenses, terms, and data-use requirements.

OASIS data should be obtained directly from the official OASIS distribution rather than from this repository.

---

## Reproducibility Notes

Several safeguards were used to improve reproducibility:

- deterministic feature extraction,
- explicit feature ordering,
- leakage-group-aware splitting,
- training-only preprocessing decisions,
- validation-only model/threshold selection,
- one-time frozen internal test,
- saved model/scaler artifacts,
- saved feature list,
- saved configuration and freeze manifests,
- SHA-256 model checksum,
- inference reproducibility testing,
- group-aware bootstrap confidence intervals.

The final internal test should not be reused for model selection.

Likewise, repeatedly inspected OASIS pilot cases should not be treated as a fresh validation set for future V2 decisions.

---

## Limitations

This project has several important limitations.

### 1. 2D slice-based representation

The final model operates on selected 2D MRI slices rather than full 3D volumes, so potentially useful volumetric information is lost.

### 2. Handcrafted features

The 30 features are interpretable and computationally efficient, but they cannot represent all anatomical information present in MRI.

### 3. Dataset provenance

The exact provenance of the original training dataset must be fully documented before making strong claims about independence from external datasets.

### 4. External domain shift

The OASIS pilot demonstrated severe threshold-transfer failure.

Internal performance therefore must not be interpreted as evidence of equivalent performance on other scanners, preprocessing pipelines, institutions, or populations.

### 5. Small OASIS pilot

The current OASIS challenge set is very small and has already been used during development.

It is not sufficient for definitive external validation.

### 6. Early-stage difficulty

Performance was substantially weaker for `VeryMildDemented` than for `MildDemented`.

This is particularly important because the intended research objective emphasizes early disease.

### 7. Label semantics

Dataset class labels and project CDR mappings are research labels, not complete clinical diagnoses.

### 8. Clinical applicability

No claim of clinical safety, efficacy, or diagnostic utility is made.

A clinically deployable system would require substantially broader validation, including multi-site testing, protocol harmonization, demographic analysis, calibration studies, prospective evaluation, and appropriate regulatory review.

---

## Roadmap

Planned next steps include:

- [ ] finish OASIS V2 threshold-development analysis,
- [ ] freeze any V2 candidate before new testing,
- [ ] collect a new untouched OASIS holdout,
- [ ] investigate original dataset provenance and possible subject overlap,
- [ ] evaluate calibration in addition to ranking metrics,
- [ ] expand external testing across acquisition domains,
- [ ] assess robustness across demographic and scanner subgroups when metadata permit,
- [ ] compare 2D handcrafted methods with 3D and deep-learning baselines,
- [ ] improve automated path/config handling for reproducible execution,
- [ ] add publication-quality figures and experiment summaries.

---

## Intended Use

This repository is intended for:

- educational use,
- academic experimentation,
- reproducible machine-learning research,
- investigation of MRI feature engineering,
- study of leakage control and domain shift.

It is **not intended for clinical diagnosis**.

---

## Summary

The project achieved strong internal discrimination with a leakage-aware classical ML pipeline:

```text
Internal ROC-AUC: 0.8879
Internal PR-AUC:  0.9293
Balanced Acc.:    0.8102
F1:               0.8524
```

However, the OASIS challenge study showed that the original threshold did not transfer across domains:

```text
OASIS ROC-AUC:       0.6914
OASIS Balanced Acc.: 0.5000
OASIS Specificity:   0.0000
```

The central conclusion is therefore not simply that the model performs well internally, but that **careful leakage control, frozen evaluation, reproducibility, and external domain-shift testing are essential before interpreting MRI classification performance as generalizable**.
