# Project Specification: Agentic Multi-Family ML Workspace for Wellbore Geology Prediction

## Task Objective
Build a modular, extensible machine learning experimentation harness in Python. The framework must allow an autonomous execution agent to dynamically register, train, evaluate, and ensemble 7 distinct model families to predict True Vertical Thickness (TVT). The evaluation metric is strictly Root Mean Squared Error (RMSE) evaluated over a robust 5-fold GroupKFold split by `WELLNAME`.

## 1. Input Schema & File Structure Accounted For
The pipeline must dynamically parse and ingest data matching the following structured schema:
### Directory: `train/`
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

### Directory: `test/`
Contains the evaluation data for approximately 200 wells. Each well has two associated files:
1. **`{WELLNAME}__horizontal_well.csv`** — Trajectory and log data. In these files, the `TVT` target is hidden (replaced with `NaN`) in the evaluation zone. 
2. **`{WELLNAME}__typewell.csv`** — Vertical reference log for the test well containing `TVT`, `GR`, and `Geology`.

## 2. Modular Core Architecture
The pipeline must feature strict object-oriented separation so an agent can plug in new model architectures seamlessly without breaking the evaluation loops:

- `AbstractBaseModel`: Generic blueprint with `.fit(X, y)` and `.predict(X)` interfaces.
- `FeaturePipeline`: Centralized data transformer that handles tabular data engineering, 1D sequential windowing, signal formatting, and continuous target scaling/standardization.
- `ExperimentOrchestrator`: An automated harness that tracks runs, enforces a global `FAST_DEBUG` flag (downsampling to a subset of wells for rapid agent exploration), executes cross-validation, and logs metadata to a centralized `metrics.json` file. **Crucially, this module must save out-of-fold (OOF) predictions for every registered model family to serve as the training features for the stacking meta-model.**

## 3. Model Family Sandboxes to Implement
Codex must implement seven specific subclasses inheriting from `AbstractBaseModel`:

### Family A: TreeEnsembleModel (Tabular Machine Learning)
- **Algorithms:** LightGBM, CatBoost, and XGBoost regressors.
- **Features:** Input spatial kinematics ($\Delta X, \Delta Y, \Delta Z$, and bit inclination $\theta$), moving window `GR` metrics, and geometric distances to structural surface columns (`ANCC`, `ASTNU`, etc.).

### Family B: DeepSequenceModel (Sequence Learning)
- **Algorithms:** PyTorch 1D Temporal Convolutional Network (1D-TCN) or Bidirectional LSTM.
- **Behavior:** Models the spatial well path as a continuous sequence along `MD` to preserve geological continuity.

### Family C: StateSpaceMatcher (Physics-Based Signal Alignment)
- **Algorithms:** Particle Filtering (PF) or Windowed Dynamic Time Warping (DTW) with a Beam Search decoder.
- **Behavior:** Maps the horizontal `GR` signal deviations dynamically against the vertical reference frame of the `typewell GR` log profile to track stratigraphic positioning.

### Family D: TestTimeAdaptationModel (Transductive Learning)
- **Algorithms:** Pseudo-labeling MLP or Self-Supervised Domain Adaptation Network.
- **Behavior:** Utilizes the partially available `TVT_input` in the test wellbore data at runtime to perform real-time localized model calibration prior to generating evaluation zone predictions.

### Family E: SubsequenceProfileMatcher (Non-Parametric Instance Learning)
- **Algorithms:** MASS (M_App_Sub_Sequence) or Fast-KNN utilizing Shape-Based Distance.
- **Behavior:** Correlates 1D local sliding windows of the test well directly against the global historical training well-log registry to look up literal structural matching profiles.

### Family F: DenseSegmentationUnet (Spatial CNN)
- **Algorithms:** 1D U-Net with skip-connections and multi-scale downsampling.
- **Behavior:** Maps a fixed spatial array window of the horizontal log directly to an equivalent continuous spatial window of standardized `TVT` coordinates, utilizing encoder-decoder context to balance micro-layer boundaries with macro-geological trends.

### Family G: KinematicTrendBaseline (Geometric Splines)
- **Algorithms:** Thin-Plate Splines / Ordinary Kriging Regressors.
- **Behavior:** Models the structural elevation trends strictly from the spatial coordinates (`X`, `Y`, `Z`) and the surface column geometries, creating a smooth baseline uncorrupted by logging sensor noise.

## 4. Strict Validation & Leakage Prevention
- **Fold Splitting:** Use a 5-fold `GroupKFold` split grouped strictly by the `WELLNAME` identifier extracted from file headers or titles. 
- **Isolation:** Features (especially rolling statistics and signal-matching probabilities) must be computed purely within each cross-validation fold or locally per individual well to eliminate look-ahead leakage. **This exact cross-validation structure must be strictly mirrored when fitting the meta-model to prevent data leakage during stacking.**

## 5. Meta-Ensemble Blending (CRITICAL STACKING COMPONENT)
- **Meta-Learning:** Implement a `MetaBlender` class that aggregates the Out-Of-Fold (OOF) predictions generated by all 7 registered model families.
- **Constrained Ridge Stacking:** Train a scikit-learn `Ridge` regressor or an explicit Non-Negative Least Squares (NNLS) solver using the $7$-column OOF prediction matrix as features and the true `TVT` targets as labels. 
- **Hyperparameter Tuning:** The agent must automatically optimize the Ridge $\alpha$ regularization penalty via an internal `RidgeCV` loop to mitigate severe multicollinearity across the 7 families.
- **Weights Constraint:** Enforce non-negative coefficients ($\beta_i \ge 0$) and normalize weights to sum to 1 to ensure structural stability and prevent geometric scale drifting.

## 6. Output and Submission Requirements
- **File Format:** Write final blended predictions strictly into a file named exactly `submission.csv`.
- **Structure:** The output file must contain exactly two columns: `id` and `tvt`.
- **ID Templating:** The `id` column must match the format `{WELLNAME}_{row_index}` where `{WELLNAME}` is the unique well identifier string and `{row_index}` is the 0-indexed row integer corresponding to the evaluation zone lines where the original test data had hidden `TVT` fields. No NaN values are allowed in the final output.