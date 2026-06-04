from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm

try:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - notebook execution shim
    ROOT = Path(os.getcwd()).resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - notebook flattening shim
    from src.pipeline import AbstractBaseModel, FeaturePipeline, make_stratified_group_folds
except ModuleNotFoundError:  # pragma: no cover - flattened notebook execution shim
    required = {"AbstractBaseModel", "FeaturePipeline", "make_stratified_group_folds"}
    if required.issubset(globals()):
        AbstractBaseModel = globals()["AbstractBaseModel"]
        FeaturePipeline = globals()["FeaturePipeline"]
        make_stratified_group_folds = globals()["make_stratified_group_folds"]
    else:
        raise


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


class LinearEnsembleModel(AbstractBaseModel):
    """Independent linear baselines for the wellbore TVT task."""

    BACKEND_ORDER = ("ridge", "lasso", "elasticnet")
    FAMILY_LABEL = "Family C"
    MAX_TRAIN_ROWS = 50000
    BACKEND_DISPLAY_NAMES = {
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elasticnet": "ElasticNet",
    }

    def __init__(
        self,
        feature_pipeline: Optional[FeaturePipeline] = None,
        group_col: str = "WELLNAME",
        target_col: str = "TVT",
        target_input_col: str = "TVT_input",
        n_splits: int = 5,
        scale_target: bool = True,
        random_state: int = 42,
        metrics_path: str | Path | None = None,
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.target_input_col = target_input_col
        self.n_splits = int(n_splits)
        self.scale_target = bool(scale_target)
        self.random_state = int(random_state)
        self.metrics_path = _resolve_path(metrics_path) if metrics_path else None

        self.feature_pipeline = feature_pipeline or FeaturePipeline(
            group_col=group_col,
            target_col=target_col,
            target_input_col=target_input_col,
            scale_target=scale_target,
        )
        self.feature_pipeline.scale_target = self.scale_target

        self.feature_columns_: List[str] = []
        self.backend_params_: Dict[str, Dict[str, Any]] = self._default_backend_params()
        self.fold_models_: Dict[str, List[Any]] = {backend: [] for backend in self.BACKEND_ORDER}
        self.full_models_: Dict[str, Any] = {}
        self.fold_scores_: Dict[str, List[float]] = {backend: [] for backend in self.BACKEND_ORDER}
        self.oof_predictions_ridge: Optional[np.ndarray] = None
        self.oof_predictions_lasso: Optional[np.ndarray] = None
        self.oof_predictions_elasticnet: Optional[np.ndarray] = None
        self.oof_predictions_: Dict[str, np.ndarray] = {}
        self.scaled_oof_predictions_: Dict[str, np.ndarray] = {}
        self.groups_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def _scale_residual(self, residual: np.ndarray, mean: float, std: float) -> np.ndarray:
        residual_arr = np.asarray(residual, dtype=float).reshape(-1)
        if not self.scale_target:
            return residual_arr
        return (residual_arr - float(mean)) / (float(std) or 1.0)

    def _unscale_residual(self, residual_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
        residual_arr = np.asarray(residual_scaled, dtype=float).reshape(-1)
        if not self.scale_target:
            return residual_arr
        return residual_arr * (float(std) or 1.0) + float(mean)

    def _training_clip_bounds(self, target: Sequence[float], padding: float = 250.0) -> Tuple[float, float]:
        target_arr = np.asarray(target, dtype=float).reshape(-1)
        finite = target_arr[np.isfinite(target_arr)]
        if finite.size == 0:
            raise ValueError("Cannot derive training clip bounds from an empty target array.")
        safe_min = float(np.min(finite) - float(padding))
        safe_max = float(np.max(finite) + float(padding))
        return (safe_min, safe_max) if safe_min <= safe_max else (safe_max, safe_min)

    def _prediction_clip_bounds(self, prior: Sequence[float], padding: float = 150.0) -> Tuple[float, float]:
        prior_arr = np.asarray(prior, dtype=float).reshape(-1)
        finite = prior_arr[np.isfinite(prior_arr)]
        if finite.size == 0:
            raise ValueError("Cannot derive prediction clip bounds from an empty prior array.")
        safe_min = float(np.min(finite) - float(padding))
        safe_max = float(np.max(finite) + float(padding))
        return (safe_min, safe_max) if safe_min <= safe_max else (safe_max, safe_min)

    def fit(self, X: Any, y: Any) -> "LinearEnsembleModel":
        df = self._ensure_dataframe(X)
        target = self._resolve_target(df, y)
        self._validate_groups(df)

        self.feature_pipeline.scale_target = self.scale_target
        self.feature_pipeline.fit(df, y=target)

        feature_frame = self._build_feature_frame(df)
        self.feature_columns_ = list(feature_frame.columns)

        groups = df[self.group_col].astype(str).to_numpy()
        folds = make_stratified_group_folds(
            df,
            target_col=self.target_col,
            group_col=self.group_col,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        n_folds = len(folds)

        self.fold_models_ = {backend: [] for backend in self.BACKEND_ORDER}
        self.fold_scores_ = {backend: [] for backend in self.BACKEND_ORDER}
        self.oof_predictions_ = {}
        self.scaled_oof_predictions_ = {}

        for backend_name in self.BACKEND_ORDER:
            backend_scaled_oof = np.full(len(feature_frame), np.nan, dtype=float)
            backend_original_oof = np.full(len(feature_frame), np.nan, dtype=float)
            backend_display = self.BACKEND_DISPLAY_NAMES.get(backend_name, backend_name)

            with tqdm(
                total=n_folds,
                desc=f"Training {self.FAMILY_LABEL}: {backend_display}",
                dynamic_ncols=True,
                leave=False,
            ) as fold_bar:
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    fold_bar.set_description(
                        f"Training {self.FAMILY_LABEL}: {backend_display} | Fold {fold_idx + 1}/{n_folds}"
                    )
                    train_df = df.iloc[train_idx].copy()
                    val_df = df.iloc[val_idx].copy()
                    y_train = target[train_idx]
                    y_val = target[val_idx]
                    train_safe_min, train_safe_max = self._training_clip_bounds(y_train)

                    if len(train_df) > self.MAX_TRAIN_ROWS:
                        print("📉 Sub-sampling Family C training rows down to 50,000...")
                        rng = np.random.default_rng(self.random_state)
                        sampled_idx = np.sort(rng.choice(len(train_df), size=self.MAX_TRAIN_ROWS, replace=False))
                        train_df = train_df.iloc[sampled_idx].copy()
                        y_train = y_train[sampled_idx]

                    fold_pipeline = deepcopy(self.feature_pipeline)
                    fold_pipeline.scale_target = self.scale_target
                    fold_pipeline.fit(train_df, y=y_train)

                    train_prior = fold_pipeline.predict_structural_prior(train_df)
                    val_prior = fold_pipeline.predict_structural_prior(val_df)
                    train_residual = y_train - train_prior
                    residual_mean = float(np.mean(train_residual))
                    residual_std = float(np.std(train_residual, ddof=0) or 1.0)

                    train_features = self._build_feature_frame(train_df, pipeline=fold_pipeline)
                    val_features = self._build_feature_frame(
                        val_df,
                        pipeline=fold_pipeline,
                        reference_columns=train_features.columns,
                    )

                    estimator = self._make_estimator(backend_name)
                    y_train_residual_scaled = self._scale_residual(train_residual, residual_mean, residual_std)
                    estimator.fit(train_features, y_train_residual_scaled)
                    val_pred_residual_scaled = np.asarray(estimator.predict(val_features), dtype=float).reshape(-1)
                    val_pred_residual = self._unscale_residual(val_pred_residual_scaled, residual_mean, residual_std)
                    val_pred = np.clip(val_prior + val_pred_residual, train_safe_min, train_safe_max)

                    backend_scaled_oof[val_idx] = self.feature_pipeline.transform_target(val_pred)
                    backend_original_oof[val_idx] = val_pred

                    fold_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
                    self.fold_scores_[backend_name].append(fold_rmse)
                    self.fold_models_[backend_name].append(
                        {
                            "fold_index": fold_idx,
                            "pipeline": fold_pipeline,
                            "model": estimator,
                            "feature_columns": list(train_features.columns),
                            "residual_mean": residual_mean,
                            "residual_std": residual_std,
                            "fold_rmse": fold_rmse,
                            "backend_state": self._serializable_model_state(estimator),
                        }
                    )
                    fold_bar.update(1)

            if np.isnan(backend_original_oof).any():
                raise RuntimeError(f"OOF predictions for '{backend_name}' contain unfilled rows.")

            self.scaled_oof_predictions_[backend_name] = backend_scaled_oof
            self.oof_predictions_[backend_name] = backend_original_oof

        self.oof_predictions_ridge = self.oof_predictions_["ridge"]
        self.oof_predictions_lasso = self.oof_predictions_["lasso"]
        self.oof_predictions_elasticnet = self.oof_predictions_["elasticnet"]
        self.groups_ = groups

        full_df = df
        full_target = target
        if len(df) > self.MAX_TRAIN_ROWS:
            print("📉 Sub-sampling Family C final full-model training rows down to 50,000...")
            rng = np.random.default_rng(self.random_state)
            sampled_idx = np.sort(rng.choice(len(df), size=self.MAX_TRAIN_ROWS, replace=False))
            full_df = df.iloc[sampled_idx].copy()
            full_target = target[sampled_idx]

        full_pipeline = deepcopy(self.feature_pipeline)
        full_pipeline.scale_target = self.scale_target
        full_pipeline.fit(full_df, y=full_target)
        full_features = self._build_feature_frame(full_df, pipeline=full_pipeline)
        full_prior = full_pipeline.predict_structural_prior(full_df)
        full_residual = full_target - full_prior
        full_residual_mean = float(np.mean(full_residual))
        full_residual_std = float(np.std(full_residual, ddof=0) or 1.0)
        full_target_scaled = self._scale_residual(full_residual, full_residual_mean, full_residual_std)

        self.full_models_ = {}
        for backend_name in self.BACKEND_ORDER:
            estimator = self._make_estimator(backend_name)
            estimator.fit(full_features, full_target_scaled)
            self.full_models_[backend_name] = {
                "pipeline": full_pipeline,
                "model": estimator,
                "feature_columns": list(full_features.columns),
                "residual_mean": full_residual_mean,
                "residual_std": full_residual_std,
                "backend_state": self._serializable_model_state(estimator),
            }

        self.is_fitted_ = True
        self._log_training_summary()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("LinearEnsembleModel must be fit before calling predict.")
        df = self._ensure_dataframe(X)
        backend_preds = self.predict_all(df)
        stacked = np.column_stack([backend_preds[name] for name in self.BACKEND_ORDER])
        prior = self.full_models_[self.BACKEND_ORDER[0]]["pipeline"].predict_structural_prior(df)
        safe_min, safe_max = self._prediction_clip_bounds(prior)
        return np.clip(np.mean(stacked, axis=1), safe_min, safe_max)

    def predict_all(self, X: Any) -> Dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("LinearEnsembleModel must be fit before calling predict.")

        df = self._ensure_dataframe(X)
        self._validate_groups(df)

        outputs: Dict[str, np.ndarray] = {}
        for backend_name in self.BACKEND_ORDER:
            model_bundle = self.full_models_[backend_name]
            pipeline = model_bundle["pipeline"]
            prior = pipeline.predict_structural_prior(df)
            safe_min, safe_max = self._prediction_clip_bounds(prior)
            feature_frame = self._build_feature_frame(
                df,
                pipeline=pipeline,
                reference_columns=model_bundle["feature_columns"],
            )
            pred_residual_scaled = np.asarray(model_bundle["model"].predict(feature_frame), dtype=float).reshape(-1)
            pred_residual = self._unscale_residual(
                pred_residual_scaled,
                model_bundle["residual_mean"],
                model_bundle["residual_std"],
            )
            outputs[backend_name] = np.clip(prior + pred_residual, safe_min, safe_max)
        return outputs

    def predict_oof(self) -> Dict[str, np.ndarray]:
        if not self.oof_predictions_:
            raise RuntimeError("OOF predictions are not available before fit.")
        return self.oof_predictions_

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if not self.feature_columns_:
                raise ValueError("NumPy input is only supported after fitting on a DataFrame.")
            return pd.DataFrame(X, columns=self.feature_columns_)
        raise TypeError("LinearEnsembleModel expects a pandas DataFrame or NumPy array.")

    def _resolve_target(self, df: pd.DataFrame, y: Any) -> np.ndarray:
        if y is not None:
            target = np.asarray(y, dtype=float).reshape(-1)
        elif self.target_col in df.columns:
            target = pd.to_numeric(df[self.target_col], errors="coerce").to_numpy(dtype=float)
        else:
            raise ValueError("Target values must be provided via y or the target column.")

        if len(target) != len(df):
            raise ValueError("Target length must match the number of rows in X.")
        if np.isnan(target).any():
            raise ValueError("Target values contain NaNs after resolution.")
        return target

    def _validate_groups(self, df: pd.DataFrame) -> None:
        if self.group_col not in df.columns:
            raise ValueError(f"LinearEnsembleModel requires '{self.group_col}' for GroupKFold.")

    def _scale_target(self, y: np.ndarray) -> np.ndarray:
        if not self.scale_target:
            return y.astype(float, copy=True)
        return self.feature_pipeline.transform_target(y)

    def _unscale_target(self, y: np.ndarray) -> np.ndarray:
        if not self.scale_target:
            return y.astype(float, copy=True)
        return self.feature_pipeline.inverse_transform_target(y)

    def _build_feature_frame(
        self,
        df: pd.DataFrame,
        pipeline: Optional[FeaturePipeline] = None,
        reference_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        pipe = pipeline or self.feature_pipeline
        feature_frame = pipe.get_numeric_feature_frame(df, target_col=self.target_col, fillna=0.0)
        if reference_columns is not None:
            feature_frame = feature_frame.reindex(columns=list(reference_columns), fill_value=0.0)
        return feature_frame

    def _default_backend_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ridge": {"alpha": 10.0, "fit_intercept": True, "random_state": self.random_state},
            "lasso": {
                "alpha": 0.001,
                "fit_intercept": True,
                "max_iter": 10000,
                "tol": 1e-4,
                "random_state": self.random_state,
            },
            "elasticnet": {
                "alpha": 0.001,
                "l1_ratio": 0.5,
                "fit_intercept": True,
                "max_iter": 10000,
                "tol": 1e-4,
                "random_state": self.random_state,
            },
        }

    def _make_estimator(self, backend_name: str) -> Any:
        params = dict(self.backend_params_[backend_name])
        scaler = RobustScaler()
        if backend_name == "ridge":
            model = Ridge(**params)
        elif backend_name == "lasso":
            model = Lasso(**params)
        elif backend_name == "elasticnet":
            model = ElasticNet(**params)
        else:
            raise KeyError(f"Unknown backend '{backend_name}'.")
        return make_pipeline(scaler, model)

    def _serializable_model_state(self, estimator: Any) -> Dict[str, Any]:
        model = estimator[-1] if hasattr(estimator, "__getitem__") else estimator
        params = {}
        if hasattr(model, "get_params"):
            raw = model.get_params(deep=False)
            params = {
                key: value
                for key, value in raw.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
        state = {"backend": model.__class__.__name__, "params": params}
        if hasattr(model, "coef_"):
            state["coef_shape"] = list(np.asarray(model.coef_).shape)
        if hasattr(model, "alpha"):
            state["alpha"] = float(getattr(model, "alpha"))
        return state

    def _log_training_summary(self) -> None:
        if self.metrics_path is None:
            return

        summary = {
            "model_family": "LinearEnsembleModel",
            "n_splits": self.n_splits,
            "scale_target": self.scale_target,
            "fold_rmse_mean": {
                backend: float(np.mean(scores)) if scores else None
                for backend, scores in self.fold_scores_.items()
            },
            "backend_params": self.backend_params_,
        }

        existing: List[Dict[str, Any]] = []
        if self.metrics_path.exists():
            try:
                loaded = json.loads(self.metrics_path.read_text())
                if isinstance(loaded, list):
                    existing = loaded
                elif isinstance(loaded, dict):
                    existing = [loaded]
            except json.JSONDecodeError:
                existing = []

        existing.append(summary)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    wells = [f"WELL_{i}" for i in range(1, 6)]
    rows_per_well = 9
    frames = []

    for well_idx, wellname in enumerate(wells):
        md = np.arange(rows_per_well, dtype=float)
        x = 250.0 + well_idx * 15.0 + np.cumsum(rng.normal(1.0, 0.08, size=rows_per_well))
        y = 500.0 + well_idx * 8.0 + np.cumsum(rng.normal(0.7, 0.08, size=rows_per_well))
        z = 3500.0 - np.cumsum(np.abs(rng.normal(0.3, 0.02, size=rows_per_well)))
        gr = 60.0 + np.sin(md / 2.0) * 4.0 + rng.normal(0.0, 0.4, size=rows_per_well)
        tvt = 12.0 + well_idx * 0.8 + md * 0.22 + rng.normal(0.0, 0.08, size=rows_per_well)

        frames.append(
            pd.DataFrame(
                {
                    "WELLNAME": wellname,
                    "MD": md,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "ANCC": z + 18.0,
                    "ASTNU": z + 14.0,
                    "ASTNL": z + 12.0,
                    "EGFDU": z + 9.0,
                    "EGFDL": z + 4.0,
                    "BUDA": z + 1.5,
                    "GR": gr,
                    "TVT": tvt,
                    "TVT_input": np.where(md < 4, tvt, np.nan),
                }
            )
        )

    mock_df = pd.concat(frames, ignore_index=True)
    model = LinearEnsembleModel(metrics_path=None)
    model.fit(mock_df, mock_df["TVT"].to_numpy())
    preds = model.predict(mock_df)

    assert len(model.oof_predictions_ridge) == len(mock_df)
    assert len(model.oof_predictions_lasso) == len(mock_df)
    assert len(model.oof_predictions_elasticnet) == len(mock_df)
    assert not np.isnan(model.oof_predictions_ridge).any()
    assert not np.isnan(model.oof_predictions_lasso).any()
    assert not np.isnan(model.oof_predictions_elasticnet).any()
    assert len(preds) == len(mock_df)
    assert not np.isnan(preds).any()

    print("models_linear.py smoke test passed.")
