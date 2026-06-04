# ROGII Subsurface Wellbore Trajectory Prediction

An end-to-end, high-throughput machine learning framework built to predict true vertical thickness (TVT) trajectories in horizontal lateral wells. 

By shifting the learning objective from high-variance absolute depths to relative, foot-by-foot stratigraphic thickness variations ($dtvt$), this pipeline eliminates baseline vertical drift and achieves state-of-the-art trajectory alignment.

### Current Benchmark
* **Public Leaderboard Score:** **15.82 RMSE** 
* **Full Pipeline Runtime:** ~1 Hour (Cross-validation + Inference across all 773 wells)
* **Validation Strategy:** Robust 10-Fold `GroupKFold` grouped strictly by `WELLNAME`

### Core Architecture
1. **Hybrid Surface Fallback Estimator:** A deterministic typewell lookup paired with an auxiliary GBDT layer that eliminates `NaN` boundaries on unseen test wells to ensure 100% geometric feature coverage.
2. **Delta-Target Optimization:** Feature engineering pipeline that maps relative geological changes: $dtvt = TVT.diff()$.
3. **Stacked Tabular Ensemble:** A diversified blend of regularized `LightGBM` (leaf-wise with extremely randomized trees), `CatBoost` (symmetric structure), and `XGBoost` (level-wise histogram bins).
4. **Meta-Learner & Anchor Loop:** A Ridge Regression meta-learner blends out-of-fold predictions before an edge-anchored forward cumulative sum (`cumsum`) reconstructs absolute trajectory depths.

### Collaboration Notes
* **Branch Protection Active:** The `main` branch is frozen to protect the 15.0 RMSE core production architecture. 

