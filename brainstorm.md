Objective: Achieve top 10 leaderboard on ROGII - Wellbore Geology Prediction

Competition: develop ML model that predicts the geology encountered along a horizontal wellbore. Model should identify favorable layers from drilling data and guide well placement more accurately

Competition link: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview

Dataset Schema

- training
    - horizontal well file: actual drilled well trajectory (1 foot along the well)
        - MD: measured depth, total distance along drilled path
        - X, Y: surface-plane coordinates
        - Z: true vertical depth, how deep below sea level
        - ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA (geological surface column): predicted vertical depths of geological boundaries
        - gamma ray log: natural radioactivity of different rocks
        - TVT (target variable): true vertical thickness
        - TVT_input: copy of TVT that contains NaN values for evaluation zone
    - typewell file:
        - TVT: Vertical Depth Index (ft): Primary depth reference for the vertical log. Corresponds to TVT (geological position) of the associated horizontal well.
        - GR: vertical gamma ray signature (critical because you can align horizontal GR with vertical GR signal)
        - Geology: categorical rock formation labels that tell what rock unit exists at each TVT level
    - \<wellname\>.png: visualization of well path and geological cross-section
- test: both horizontal well and typewell files except horizontal well does not have TVT in evaluation zone.

Potential Feature Engineering
- geometric baseline features
    - bit inclination: local angle of path relative to vertical
    - apparent dip eature: distance from known gelogical formation surfaces provided in training data
- target and feature transformation:
    - SDF Target Engineering: transform the categorical formation boundaries from the typewell into a continuous Signed Distance Field (SDF).
    - Multi-Task Learning (MTL): Train the sequential model to simultaneously predict the continuous $TVT$ value ($RMSE$ loss) and classify the corresponding categorical Geology label (Cross-Entropy loss) mapping from the typewell. The secondary classification loss acts as an incredibly strong structural regularizer.

Potential model approaches:
- signal alignment: dynamic time warping (with Sakoe-Chiba band), normalized cross-correlation (checking shapes) convolutional models, PatchTST
- sequential models: LSTMs, transformers, temporal CNNs
- spatial modeling: X/Y/Z trajectory, formation surfaces

Experiments:
### Ridge Stacking??? 
(see agent_workspace.md)
1. Non-Parametric Signal Matchers: Particle Filtering (PF) or Windowed DTW. Use the vertical Typewell GR as the emission distribution state space, letting the agent optimize transition noise and local dip parameters.
2. Gradient Boosted Trees (GBDTs): Ensembles of CatBoost, LightGBM, and XGBoost. Trees excel at maps involving spatial geometric differences (e.g., $Z - \text{marker\_surface}$) and statistical rolling window aggregates.
3. Deep Sequence Models: 1D Unet, Dilated Temporal Convolutional Networks (TCN), or Bidirectional LSTMs mapping the inputs to a continuous $TVT$ curve. 

Steps
1. warp using a sliding window approach or evaluate sequence matching allowing for structural inversions (reversals in geology sequence). Alternatively, map $(X, Y, Z)$ to an estimated structural frame before attempting signal alignment.
2. feature engineering
3. train model

Validation: 
- GroupKFold by Well: You must never use random train-test splits. The evaluation zone covers entirely unseen continuous segments of test wells. Implement a GroupKFold split where each unique WELLNAME is kept entirely intact in its own validation fold.

Additional resources/discussion:
- applied advice from literature: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701041
- CNN + SDF example: https://www.kaggle.com/code/hengck23/cnn-sdf-example/notebook
- CNN + MTP example: https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395
- geophysicist approach (domain priors + Q-3D tortuosity): https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702131

Literature:
- arvix paper: "Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning" - Sergey Alyaev
https://arxiv.org/pdf/2201.01871 https://nfes.org/assets/workshop2022/ambrus_sequential_multi_mode_inversion_poster.pdf
- Wheeler, C. W., & Hale, D. (2014). Tying horizontal wells to vertical wells with dynamic time warping. SEG Technical Program Expanded Abstracts.
- Griffiths, R. (2019). Well Placement Fundamentals. Schlumberger.
- Zhong, J., et al. (2023). Coordinate-based Deep Learning for Subsurface Structural Modeling. Geophysics.


Evaluation: scored on RMSE

Code Requirements
- submission file: each row has id, tvt
- submissions made through Notebooks
- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named submission.csv
