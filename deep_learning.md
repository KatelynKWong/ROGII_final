# Project Specification: Agentic Multi-Family ML Workspace for Wellbore Geology Prediction

## Task Objective
Build a modular, extensible machine learning experimentation harness in the `deep_learning_pipeline.ipynb` notebook. The framework must allow an autonomous execution agent to dynamically build, train, and evaluate a multi-family predictive framework. 

The core task is to predict True Vertical Thickness (TVT) by combining structural tabular spatial features with a physics-grounded sequence alignment module. The evaluation metric is strictly Root Mean Squared Error (RMSE) evaluated over a robust 10-fold GroupKFold split by `WELLNAME`.

## 1. Input Schema & File Structure Accounted For
The pipeline must dynamically parse and ingest data matching the following structured schema:

### Training Directory: `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train/`
Contains the training data where each unique well has three associated files:
1. **`{WELLNAME}__horizontal_well.csv`** — Trajectory, geological surfaces, and log data sampled at 1 ft intervals.
   - `MD` — Measured Depth (ft): The total length of the wellbore from the surface.
   - `X` — Easting (ft) & `Y` — Northing (ft): Spatial coordinates in the horizontal plane.
   - `Z` — True Vertical Depth (ft): The vertical distance below sea level.
   - `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` — Predicted vertical depth of various geological formations (Available in Training only).
   - `GR` — Gamma Ray (API): Log measuring natural radioactivity of the rock.
   - `TVT` — True Vertical Thickness (ft): The manually interpreted geological position for each 1 ft of the lateral well. **[TARGET VARIABLE - Training Only]**
   - `TVT_input` — Input Target (ft): A copy of `TVT` provided as a feature. This column contains `NaN` values for the evaluation zone.
2. **`{WELLNAME}__typewell.csv`** — Vertical reference log for geological correlation.
   - `TVT` — Vertical Depth Index (ft): Primary depth reference for the vertical log. Corresponds directly to the TVT (geological position) of the associated horizontal well.
   - `GR` — Gamma Ray (API): The vertical Gamma Ray signature used for correlation.
   - `Geology` — Formation Label: Categorical label indicating the geological unit (e.g., `EGFDL`, `BUDA`).
3. **`{WELLNAME}.png`** — Structural visualization of the well path and geological cross-section.

### Testing Directory: `/kaggle/input/competitions/rogii-wellbore-geology-prediction/test/`
Contains the evaluation data for approximately 200 wells. Each well has two associated files:
1. **`{WELLNAME}__horizontal_well.csv`** — Trajectory and log data. In these files, the `TVT` target is hidden (replaced with `NaN`) in the evaluation zone. 
2. **`{WELLNAME}__typewell.csv`** — Vertical reference log for the test well containing `TVT`, `GR`, and `Geology`.

## 2. Modular Core Architecture & Iterative Execution Strategy
The autonomous agent must execute and log a hierarchical evaluation framework, moving strictly from baseline audits to cross-domain fusion.

### Phase 2.0: Target Leakage & Boundary Audit
* **Objective:** Audit the missingness topology of `TVT_input` to exploit geometric starting conditions.
* **Execution:** Characterize the hidden evaluation zones (e.g., terminal vs. gapped intervals). Extract the last known valid `TVT_input` value, its local gradient, and continuous curvature metrics to use as primary structural boundary features.

### Phase 2.1: Tabular Baselines (LightGBM / CatBoost / XGBoost)
* **Objective:** Quantify the maximum spatial information available in trajectory geometry and surface indicators.
* **Feature Engineering:** Compute absolute vertical distance deltas between the well path `Z` and all provided geological formation surfaces (`Z - ANCC`, `Z - BUDA`, etc.). Append the boundary tracking features from Phase 2.0.
* **Model:** Benchmark LightGBM, CatBoost, and XGBoost against each other using a strict 10-fold GroupKFold cross-validation split by `WELLNAME`.

### Phase 2.2a: Classical Alignment Baseline
* **Objective:** Establish a non-neural benchmark for physical log correlation.
* **Execution:** Implement a fast, algorithmic sequence matching protocol (e.g., Dynamic Time Warping (DTW) or rolling cross-correlation) utilizing only the `Horizontal_Well_GR` and `Typewell_GR` to map alignment distances.

### Phase 2.2b: Neural Direct Sequence Alignment Network
* **Objective:** Map horizontal and vertical rock properties using an end-to-end differentiable sequence layer.
* **Architecture:** 1. Dual parallel 1D CNNs to extract localized stratigraphic signatures from `Horizontal_Well_GR` and `Typewell_GR`.
  2. A Cross-Attention layer treating the horizontal logs as Queries ($Q$) and the Typewell logs as Keys/Values ($K, V$) to generate a dense alignment matrix.
  3. **Monotonicity Enforcer:** Apply a smoothness penalty loss or local constraint mask to the cross-attention matrix to ensure predicted layer transitions evolve realistically without erratic, unphysical vertical jumping.

### Phase 2.3: Staged Late Fusion & Error Decomposition
* **Objective:** Evaluate if combining structural tabular forecasts with sequence alignment reduces error.
* **Mechanism:** Train an Out-Of-Fold (OOF) Ridge Regression meta-learner to blend the predictions. If the sequence network's validation RMSE underperforms the tabular baseline to an extent that hurts the ensemble, the agent must preserve the tabular-only model as the primary path forward.
* **Diagnostic Logging:** For each well, log validation RMSE, Mean Absolute Error (MAE), structural bias, and cumulative spatial drift. Generate diagnostic line plots comparing `Predicted TVT` vs. `Actual TVT` across Measured Depth (`MD`) to locate specific failure zones.

## 3. Output and Submission Requirements
- **File Format:** Write final blended predictions strictly into a file named exactly `submission.csv`.
- **Structure:** The output file must contain exactly two columns: `id` and `tvt`.
- **ID Templating:** The `id` column must match the format `{WELLNAME}_{row_index}` where `{WELLNAME}` is the unique well identifier string and `{row_index}` is the 0-indexed row integer corresponding to the evaluation zone lines where the original test data had hidden `TVT` fields. No NaN values are allowed in the final output.