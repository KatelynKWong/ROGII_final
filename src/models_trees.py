from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

if __package__ in {None, ""}:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.pipeline import AbstractBaseModel, FeaturePipeline


try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:  # pragma: no cover - optional dependency
    import catboost as cb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cb = None

try:  # pragma: no cover - optional dependency
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    xgb = None


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    implementation: str
    params: Dict[str, Any]


class TreeEnsembleModel(AbstractBaseModel):
    """Sequential tree-ensemble model family for tabular wellbore features.

    The class performs a 5-fold GroupKFold by WELLNAME, trains three peer tree
    regressors independently, and stores out-of-fold predictions for each
    backend in the original TVT scale.
    """

    BACKEND_ORDER = ("lightgbm", "catboost", "xgboost")
    FAMILY_LABEL = "Family A"
    BACKEND_DISPLAY_NAMES = {
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
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
        fast_debug: bool = False,
        metrics_path: str | Path | None = None,
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.target_input_col = target_input_col
        self.n_splits = int(n_splits)
        self.scale_target = bool(scale_target)
        self.random_state = int(random_state)
        self.fast_debug = bool(fast_debug)
        self.metrics_path = Path(metrics_path) if metrics_path else None

        self.feature_pipeline = feature_pipeline or FeaturePipeline(
            group_col=group_col,
            target_col=target_col,
            target_input_col=target_input_col,
            scale_target=scale_target,
        )
        self.feature_pipeline.scale_target = self.scale_target

        self.feature_columns_: List[str] = []
        self.backend_specs_: Dict[str, _BackendSpec] = {}
        self.hyperparameter_log_: List[Dict[str, Any]] = []
        self.fold_models_: Dict[str, List[Any]] = {}
        self.full_models_: Dict[str, Any] = {}
        self.fold_scores_: Dict[str, List[float]] = {}
        self.oof_predictions_: Dict[str, np.ndarray] = {}
        self.scaled_oof_predictions_: Dict[str, np.ndarray] = {}
        self.groups_: np.ndarray | None = None
        self.backend_feature_importance_: Dict[str, List[Dict[str, Any]]] = {}
        self.is_fitted_: bool = False

    def fit(self, X: Any, y: Any) -> "TreeEnsembleModel":
        df = self._ensure_dataframe(X)
        target = self._resolve_target(df, y)
        self._validate_groups(df)

        # Learn target scaling and feature imputations on the raw schema.
        self.feature_pipeline.scale_target = self.scale_target
        self.feature_pipeline.fit(df, y=target)

        feature_frame = self._build_feature_frame(df)
        self.feature_columns_ = list(feature_frame.columns)

        groups = df[self.group_col].astype(str).to_numpy()
        target_scaled = self._scale_target(target)

        splitter = GroupKFold(n_splits=min(self.n_splits, len(np.unique(groups))))
        indices = np.arange(len(feature_frame))
        folds = list(splitter.split(indices, groups=groups))
        n_folds = len(folds)

        self.backend_specs_ = self._build_backend_specs()
        self.hyperparameter_log_ = []
        self.fold_models_ = {backend: [] for backend in self.BACKEND_ORDER}
        self.fold_scores_ = {backend: [] for backend in self.BACKEND_ORDER}
        self.backend_feature_importance_ = {backend: [] for backend in self.BACKEND_ORDER}
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
                    X_train = feature_frame.iloc[train_idx]
                    X_val = feature_frame.iloc[val_idx]
                    y_train = target_scaled[train_idx]
                    y_val_original = target[val_idx]

                    fit_result = self._fit_single_backend(
                        backend_name=backend_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        fold_index=fold_idx,
                    )

                    scaled_pred = fit_result["val_pred_scaled"]
                    original_pred = self._unscale_target(scaled_pred)

                    backend_scaled_oof[val_idx] = scaled_pred
                    backend_original_oof[val_idx] = original_pred

                    fold_rmse = float(np.sqrt(np.mean((original_pred - y_val_original) ** 2)))
                    self.fold_scores_[backend_name].append(fold_rmse)
                    self.fold_models_[backend_name].append(fit_result["model"])
                    self.backend_feature_importance_[backend_name].append(
                        {
                            "fold_index": fold_idx,
                            "feature_importance": fit_result["feature_importance"],
                            "fold_rmse": fold_rmse,
                        }
                    )
                    self.hyperparameter_log_.append(
                        {
                            "backend": backend_name,
                            "fold_index": fold_idx,
                            "implementation": self.backend_specs_[backend_name].implementation,
                            "hyperparameters": self._serializable_params(fit_result["model"]),
                            "fold_rmse": fold_rmse,
                        }
                    )
                    fold_bar.update(1)

            if np.isnan(backend_original_oof).any():
                raise RuntimeError(f"OOF predictions for '{backend_name}' were not filled for every training row.")

            self.scaled_oof_predictions_[backend_name] = backend_scaled_oof
            self.oof_predictions_[backend_name] = backend_original_oof

        self.groups_ = groups

        # Fit each peer model on all data for inference.
        self.full_models_ = {}
        for backend_name in self.BACKEND_ORDER:
            full_fit = self._fit_single_backend(
                backend_name=backend_name,
                X_train=feature_frame,
                y_train=target_scaled,
                X_val=None,
                fold_index=None,
            )
            self.full_models_[backend_name] = full_fit["model"]

        self.is_fitted_ = True
        self._log_training_summary()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("TreeEnsembleModel must be fit before calling predict.")

        df = self._ensure_dataframe(X)
        backend_preds = self.predict_all(df)
        stacked = np.column_stack(list(backend_preds.values()))
        return np.mean(stacked, axis=1)

    def predict_oof(self) -> Dict[str, np.ndarray]:
        if not self.oof_predictions_:
            raise RuntimeError("OOF predictions are not available before fit.")
        return self.oof_predictions_

    def predict_backend(self, backend_name: str, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("TreeEnsembleModel must be fit before calling predict.")
        if backend_name not in self.full_models_:
            raise KeyError(f"Unknown backend '{backend_name}'.")

        df = self._ensure_dataframe(X)
        self._validate_groups(df)
        feature_frame = self._build_feature_frame(df)
        model = self.full_models_[backend_name]
        scaled_pred = np.asarray(model.predict(feature_frame), dtype=float).reshape(-1)
        return self._unscale_target(scaled_pred)

    def predict_all(self, X: Any) -> Dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("TreeEnsembleModel must be fit before calling predict.")

        df = self._ensure_dataframe(X)
        self._validate_groups(df)
        feature_frame = self._build_feature_frame(df)
        return {
            backend_name: self._unscale_target(
                np.asarray(self.full_models_[backend_name].predict(feature_frame), dtype=float).reshape(-1)
            )
            for backend_name in self.BACKEND_ORDER
        }

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if not self.feature_columns_:
                raise ValueError(
                    "NumPy input is only supported after feature columns have been learned "
                    "from a DataFrame fit."
                )
            return pd.DataFrame(X, columns=self.feature_columns_)
        raise TypeError("TreeEnsembleModel expects a pandas DataFrame or NumPy array.")

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
            raise ValueError(f"TreeEnsembleModel requires '{self.group_col}' for GroupKFold.")

    def _build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_frame = self.feature_pipeline.get_numeric_feature_frame(df, target_col=self.target_col, fillna=0.0)
        if self.feature_columns_:
            feature_frame = feature_frame.reindex(columns=self.feature_columns_, fill_value=0.0)
        return feature_frame

    def _scale_target(self, y: np.ndarray) -> np.ndarray:
        if not self.scale_target:
            return y.astype(float, copy=True)
        return self.feature_pipeline.transform_target(y)

    def _unscale_target(self, y: np.ndarray) -> np.ndarray:
        if not self.scale_target:
            return y.astype(float, copy=True)
        return self.feature_pipeline.inverse_transform_target(y)

    def _build_backend_specs(self) -> Dict[str, _BackendSpec]:
        specs: Dict[str, _BackendSpec] = {}

        specs["lightgbm"] = _BackendSpec(
            name="lightgbm",
            implementation="lightgbm.LGBMRegressor" if lgb is not None else "sklearn.HistGradientBoostingRegressor",
            params={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 31,
                "random_state": self.random_state,
            },
        )
        specs["catboost"] = _BackendSpec(
            name="catboost",
            implementation="catboost.CatBoostRegressor" if cb is not None else "sklearn.GradientBoostingRegressor",
            params={
                "iterations": 250,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "RMSE",
                "random_seed": self.random_state,
                "verbose": False,
            },
        )
        specs["xgboost"] = _BackendSpec(
            name="xgboost",
            implementation="xgboost.XGBRegressor" if xgb is not None else "sklearn.GradientBoostingRegressor",
            params={
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": self.random_state,
            },
        )
        return specs

    def _make_backend_estimator(self, backend_name: str) -> Any:
        spec = self.backend_specs_[backend_name]
        params = dict(spec.params)

        if backend_name == "lightgbm" and lgb is not None:  # pragma: no cover - optional dependency
            return lgb.LGBMRegressor(**params)
        if backend_name == "catboost" and cb is not None:  # pragma: no cover - optional dependency
            return cb.CatBoostRegressor(**params)
        if backend_name == "xgboost" and xgb is not None:  # pragma: no cover - optional dependency
            return xgb.XGBRegressor(**params)

        # Deterministic sklearn fallback when the target package is unavailable.
        if backend_name == "lightgbm":
            return HistGradientBoostingRegressor(
                learning_rate=params["learning_rate"],
                max_depth=6,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
        if backend_name == "catboost":
            return GradientBoostingRegressor(
                n_estimators=params["iterations"],
                learning_rate=params["learning_rate"],
                max_depth=3,
                random_state=self.random_state,
            )
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=4,
            random_state=self.random_state,
            subsample=0.9,
        )

    def _fit_single_backend(
        self,
        backend_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame],
        fold_index: Optional[int],
    ) -> Dict[str, Any]:
        estimator = self._make_backend_estimator(backend_name)
        estimator.fit(X_train, y_train)
        train_pred = np.asarray(estimator.predict(X_train), dtype=float).reshape(-1)
        val_pred = np.asarray(estimator.predict(X_val), dtype=float).reshape(-1) if X_val is not None else np.array([], dtype=float)

        return {
            "model": estimator,
            "train_pred_scaled": train_pred,
            "val_pred_scaled": val_pred,
            "feature_importance": self._feature_importance(estimator),
        }

    def _feature_importance(self, estimator: Any) -> List[Dict[str, Any]]:
        importance: List[Dict[str, Any]] = []
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float).reshape(-1)
            importance = [
                {"feature": feature, "importance": float(score)}
                for feature, score in zip(self.feature_columns_, values)
            ]
        elif hasattr(estimator, "get_feature_importance"):
            values = np.asarray(estimator.get_feature_importance(), dtype=float).reshape(-1)
            importance = [
                {"feature": feature, "importance": float(score)}
                for feature, score in zip(self.feature_columns_, values)
            ]
        return importance

    def _serializable_params(self, estimator: Any) -> Dict[str, Any]:
        if hasattr(estimator, "get_params"):
            params = estimator.get_params(deep=False)
            clean_params: Dict[str, Any] = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    clean_params[key] = value
            return clean_params
        return {}

    def _log_training_summary(self) -> None:
        if self.metrics_path is None:
            return

        summary = {
            "model_family": "TreeEnsembleModel",
            "n_splits": self.n_splits,
            "scale_target": self.scale_target,
            "fold_rmse_mean": {
                backend: float(np.mean(scores)) if scores else None
                for backend, scores in self.fold_scores_.items()
            },
            "backend_specs": {k: {"implementation": v.implementation, "params": v.params} for k, v in self.backend_specs_.items()},
            "fold_scores": self.fold_scores_,
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
        self.metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


if __name__ == "__main__":
    rng = np.random.default_rng(21)
    wells = [f"WELL_{i}" for i in range(1, 6)]
    rows_per_well = 10

    frames = []
    for well_idx, wellname in enumerate(wells):
        md = np.arange(rows_per_well, dtype=float)
        x = 500.0 + well_idx * 25.0 + np.cumsum(rng.normal(1.0, 0.05, size=rows_per_well))
        y = 750.0 + well_idx * 10.0 + np.cumsum(rng.normal(0.7, 0.05, size=rows_per_well))
        z = 2500.0 - np.cumsum(np.abs(rng.normal(0.4, 0.03, size=rows_per_well)))
        gr = 85.0 + np.sin(md / 2.5) * 4.0 + rng.normal(0.0, 0.5, size=rows_per_well)
        tvt = 20.0 + well_idx * 1.5 + md * 0.3 + rng.normal(0.0, 0.1, size=rows_per_well)

        frame = pd.DataFrame(
            {
                "WELLNAME": wellname,
                "MD": md,
                "X": x,
                "Y": y,
                "Z": z,
                "ANCC": z + 15.0,
                "ASTNU": z + 12.0,
                "ASTNL": z + 9.0,
                "EGFDU": z + 6.0,
                "EGFDL": z + 3.0,
                "BUDA": z + 1.0,
                "GR": gr,
                "TVT": tvt,
                "TVT_input": np.where(md < 6, tvt, np.nan),
            }
        )
        frames.append(frame)

    mock_df = pd.concat(frames, ignore_index=True)
    model = TreeEnsembleModel(metrics_path=None)
    model.fit(mock_df, mock_df["TVT"].to_numpy())
    preds = model.predict(mock_df)

    assert set(model.oof_predictions_.keys()) == set(TreeEnsembleModel.BACKEND_ORDER)
    assert all(len(values) == len(mock_df) for values in model.oof_predictions_.values())
    assert all(len(values) == len(mock_df) for values in model.scaled_oof_predictions_.values())
    assert len(preds) == len(mock_df)
    assert not np.isnan(preds).any()

    print("models_trees.py smoke test passed.")
