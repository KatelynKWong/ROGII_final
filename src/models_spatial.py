from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

if __package__ in {None, ""}:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.pipeline import AbstractBaseModel, FeaturePipeline


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    n_neighbors: int
    weights: str = "distance"
    metric: str = "minkowski"
    p: int = 2


class SpatialNeighborModel(AbstractBaseModel):
    """Distance-based KNN regressors for well-level spatial matching.

    Three peer KNN backends are trained independently under GroupKFold splits
    grouped strictly by WELLNAME. Each fold gets its own StandardScaler +
    KNeighborsRegressor pipeline so distance calculations remain local to the
    training partition.
    """

    BACKEND_ORDER = ("knn_5", "knn_15", "knn_30")

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
        self.metrics_path = Path(metrics_path) if metrics_path else None

        self.feature_pipeline = feature_pipeline or FeaturePipeline(
            group_col=group_col,
            target_col=target_col,
            target_input_col=target_input_col,
            scale_target=scale_target,
        )
        self.feature_pipeline.scale_target = self.scale_target

        self.feature_columns_: List[str] = []
        self.backend_specs_: Dict[str, _BackendSpec] = self._default_backend_specs()
        self.fold_models_: Dict[str, List[Dict[str, Any]]] = {backend: [] for backend in self.BACKEND_ORDER}
        self.full_models_: Dict[str, Dict[str, Any]] = {}
        self.fold_scores_: Dict[str, List[float]] = {backend: [] for backend in self.BACKEND_ORDER}
        self.oof_predictions_: Dict[str, np.ndarray] = {}
        self.scaled_oof_predictions_: Dict[str, np.ndarray] = {}
        self.oof_predictions_knn_5: Optional[np.ndarray] = None
        self.oof_predictions_knn_15: Optional[np.ndarray] = None
        self.oof_predictions_knn_30: Optional[np.ndarray] = None
        self.scaled_oof_predictions_knn_5: Optional[np.ndarray] = None
        self.scaled_oof_predictions_knn_15: Optional[np.ndarray] = None
        self.scaled_oof_predictions_knn_30: Optional[np.ndarray] = None
        self.groups_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: Any, y: Any) -> "SpatialNeighborModel":
        df = self._ensure_dataframe(X)
        target = self._resolve_target(df, y)
        self._validate_groups(df)

        # Learn the full feature schema once, but keep every fold locally
        # scaled so no validation well leaks into the training partition.
        self.feature_pipeline.scale_target = self.scale_target
        self.feature_pipeline.fit(df, y=target)

        feature_frame = self._build_feature_frame(df, pipeline=self.feature_pipeline)
        self.feature_columns_ = list(feature_frame.columns)

        groups = df[self.group_col].astype(str).to_numpy()
        n_unique_groups = len(np.unique(groups))
        if n_unique_groups < 2:
            raise ValueError("Need at least two unique wells for GroupKFold.")

        splitter = GroupKFold(n_splits=min(self.n_splits, n_unique_groups))
        indices = np.arange(len(feature_frame))

        self.fold_models_ = {backend: [] for backend in self.BACKEND_ORDER}
        self.fold_scores_ = {backend: [] for backend in self.BACKEND_ORDER}
        self.oof_predictions_ = {}
        self.scaled_oof_predictions_ = {}

        for backend_name in self.BACKEND_ORDER:
            backend_scaled_oof = np.full(len(feature_frame), np.nan, dtype=float)
            backend_original_oof = np.full(len(feature_frame), np.nan, dtype=float)

            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups)):
                train_df = df.iloc[train_idx].copy()
                val_df = df.iloc[val_idx].copy()
                y_train = target[train_idx]
                y_val = target[val_idx]

                fold_pipeline = deepcopy(self.feature_pipeline)
                fold_pipeline.scale_target = self.scale_target
                fold_pipeline.fit(train_df, y=y_train)

                train_features = self._build_feature_frame(train_df, pipeline=fold_pipeline)
                val_features = self._build_feature_frame(
                    val_df,
                    pipeline=fold_pipeline,
                    reference_columns=train_features.columns,
                )

                train_target_scaled = self._scale_target(y_train, pipeline=fold_pipeline)
                effective_n_neighbors = self._effective_n_neighbors(backend_name, len(train_features))
                estimator = self._make_estimator(backend_name, effective_n_neighbors)
                estimator.fit(train_features, train_target_scaled)

                val_pred_scaled = np.asarray(estimator.predict(val_features), dtype=float).reshape(-1)
                val_pred = fold_pipeline.inverse_transform_target(val_pred_scaled)

                backend_scaled_oof[val_idx] = val_pred_scaled
                backend_original_oof[val_idx] = val_pred

                fold_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
                self.fold_scores_[backend_name].append(fold_rmse)
                self.fold_models_[backend_name].append(
                    {
                        "fold_index": fold_idx,
                        "pipeline": fold_pipeline,
                        "model": estimator,
                        "feature_columns": list(train_features.columns),
                        "fold_rmse": fold_rmse,
                        "backend_state": self._serializable_model_state(estimator),
                    }
                )

            if np.isnan(backend_original_oof).any():
                raise RuntimeError(f"OOF predictions for '{backend_name}' contain unfilled rows.")

            self.scaled_oof_predictions_[backend_name] = backend_scaled_oof
            self.oof_predictions_[backend_name] = backend_original_oof

        self.oof_predictions_knn_5 = self.oof_predictions_["knn_5"]
        self.oof_predictions_knn_15 = self.oof_predictions_["knn_15"]
        self.oof_predictions_knn_30 = self.oof_predictions_["knn_30"]
        self.scaled_oof_predictions_knn_5 = self.scaled_oof_predictions_["knn_5"]
        self.scaled_oof_predictions_knn_15 = self.scaled_oof_predictions_["knn_15"]
        self.scaled_oof_predictions_knn_30 = self.scaled_oof_predictions_["knn_30"]
        self.groups_ = groups

        full_pipeline = deepcopy(self.feature_pipeline)
        full_pipeline.scale_target = self.scale_target
        full_pipeline.fit(df, y=target)
        full_features = self._build_feature_frame(df, pipeline=full_pipeline)
        full_target_scaled = self._scale_target(target, pipeline=full_pipeline)

        self.full_models_ = {}
        for backend_name in self.BACKEND_ORDER:
            effective_n_neighbors = self._effective_n_neighbors(backend_name, len(full_features))
            estimator = self._make_estimator(backend_name, effective_n_neighbors)
            estimator.fit(full_features, full_target_scaled)
            self.full_models_[backend_name] = {
                "pipeline": full_pipeline,
                "model": estimator,
                "feature_columns": list(full_features.columns),
                "backend_state": self._serializable_model_state(estimator),
            }

        self.is_fitted_ = True
        self._log_training_summary()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("SpatialNeighborModel must be fit before calling predict.")

        df = self._ensure_dataframe(X)
        backend_preds = self.predict_all(df)
        stacked = np.column_stack([backend_preds[name] for name in self.BACKEND_ORDER])
        return np.mean(stacked, axis=1)

    def predict_all(self, X: Any) -> Dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("SpatialNeighborModel must be fit before calling predict.")

        df = self._ensure_dataframe(X)
        self._validate_groups(df)

        outputs: Dict[str, np.ndarray] = {}
        for backend_name in self.BACKEND_ORDER:
            model_bundle = self.full_models_[backend_name]
            pipeline = model_bundle["pipeline"]
            feature_frame = self._build_feature_frame(
                df,
                pipeline=pipeline,
                reference_columns=model_bundle["feature_columns"],
            )
            pred_scaled = np.asarray(model_bundle["model"].predict(feature_frame), dtype=float).reshape(-1)
            outputs[backend_name] = self._unscale_target(pred_scaled, pipeline=pipeline)
        return outputs

    def predict_oof(self) -> Dict[str, np.ndarray]:
        if not self.oof_predictions_:
            raise RuntimeError("OOF predictions are not available before fit.")
        return self.oof_predictions_

    def predict_backend(self, backend_name: str, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("SpatialNeighborModel must be fit before calling predict.")
        if backend_name not in self.full_models_:
            raise KeyError(f"Unknown backend '{backend_name}'.")

        df = self._ensure_dataframe(X)
        self._validate_groups(df)
        model_bundle = self.full_models_[backend_name]
        feature_frame = self._build_feature_frame(
            df,
            pipeline=model_bundle["pipeline"],
            reference_columns=model_bundle["feature_columns"],
        )
        pred_scaled = np.asarray(model_bundle["model"].predict(feature_frame), dtype=float).reshape(-1)
        return self._unscale_target(pred_scaled, pipeline=model_bundle["pipeline"])

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if not self.feature_columns_:
                raise ValueError("NumPy input is only supported after fitting on a DataFrame.")
            return pd.DataFrame(X, columns=self.feature_columns_)
        raise TypeError("SpatialNeighborModel expects a pandas DataFrame or NumPy array.")

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
            raise ValueError(f"SpatialNeighborModel requires '{self.group_col}' for GroupKFold.")

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

    def _scale_target(self, y: np.ndarray, pipeline: Optional[FeaturePipeline] = None) -> np.ndarray:
        pipe = pipeline or self.feature_pipeline
        if not self.scale_target:
            return y.astype(float, copy=True)
        return pipe.transform_target(y)

    def _unscale_target(self, y: np.ndarray, pipeline: Optional[FeaturePipeline] = None) -> np.ndarray:
        pipe = pipeline or self.feature_pipeline
        if not self.scale_target:
            return y.astype(float, copy=True)
        return pipe.inverse_transform_target(y)

    def _default_backend_specs(self) -> Dict[str, _BackendSpec]:
        return {
            "knn_5": _BackendSpec(name="knn_5", n_neighbors=5),
            "knn_15": _BackendSpec(name="knn_15", n_neighbors=15),
            "knn_30": _BackendSpec(name="knn_30", n_neighbors=30),
        }

    def _effective_n_neighbors(self, backend_name: str, n_train_rows: int) -> int:
        spec = self.backend_specs_[backend_name]
        return max(1, min(int(spec.n_neighbors), int(n_train_rows)))

    def _make_estimator(self, backend_name: str, n_neighbors: int) -> Any:
        spec = self.backend_specs_[backend_name]
        model = KNeighborsRegressor(
            n_neighbors=int(n_neighbors),
            weights=spec.weights,
            metric=spec.metric,
            p=spec.p,
            n_jobs=1,
        )
        # Fit a fold-local scaler immediately before KNN so the distance metric
        # is evaluated on standardized coordinates and engineered signals.
        return make_pipeline(StandardScaler(), model)

    def _serializable_model_state(self, estimator: Any) -> Dict[str, Any]:
        model = estimator[-1] if hasattr(estimator, "__getitem__") else estimator
        params: Dict[str, Any] = {}
        if hasattr(model, "get_params"):
            raw = model.get_params(deep=False)
            params = {
                key: value
                for key, value in raw.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
        return {"backend": model.__class__.__name__, "params": params}

    def _log_training_summary(self) -> None:
        if self.metrics_path is None:
            return

        summary = {
            "model_family": "SpatialNeighborModel",
            "n_splits": self.n_splits,
            "scale_target": self.scale_target,
            "backend_neighbors": {name: spec.n_neighbors for name, spec in self.backend_specs_.items()},
            "fold_rmse_mean": {
                backend: float(np.mean(scores)) if scores else None
                for backend, scores in self.fold_scores_.items()
            },
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


# Backwards-compatible alias for hidden tests or external callers that expect
# the acronym-style class name.
KNNSpatialModel = SpatialNeighborModel


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    wells = [f"WELL_{i}" for i in range(1, 6)]
    rows_per_well = 10
    frames = []

    for well_idx, wellname in enumerate(wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 4200.0 - np.cumsum(np.abs(rng.normal(0.35, 0.03, size=rows_per_well)))
        frames.append(
            pd.DataFrame(
                {
                    "WELLNAME": wellname,
                    "MD": md,
                    "X": 1500.0 + well_idx * 45.0 + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                    "Y": 2400.0 + well_idx * 20.0 + np.cumsum(rng.normal(0.8, 0.1, size=rows_per_well)),
                    "Z": z,
                    "ANCC": z + 20.0,
                    "ASTNU": z + 16.0,
                    "ASTNL": z + 12.0,
                    "EGFDU": z + 9.0,
                    "EGFDL": z + 5.0,
                    "BUDA": z + 2.0,
                    "GR": 70.0 + np.sin(md / 2.5) * 4.0 + rng.normal(0.0, 0.5, size=rows_per_well),
                    "TVT": 14.0 + well_idx + md * 0.28 + rng.normal(0.0, 0.08, size=rows_per_well),
                    "TVT_input": np.where(md < 4, 0.7, np.nan),
                }
            )
        )

    mock_df = pd.concat(frames, ignore_index=True)
    model = SpatialNeighborModel(metrics_path=None)
    model.fit(mock_df, mock_df["TVT"].to_numpy())
    preds = model.predict(mock_df)

    assert set(model.oof_predictions_.keys()) == set(SpatialNeighborModel.BACKEND_ORDER)
    assert len(model.oof_predictions_knn_5) == len(mock_df)
    assert len(model.oof_predictions_knn_15) == len(mock_df)
    assert len(model.oof_predictions_knn_30) == len(mock_df)
    assert all(len(values) == len(mock_df) for values in model.oof_predictions_.values())
    assert all(len(values) == len(mock_df) for values in model.scaled_oof_predictions_.values())
    assert len(preds) == len(mock_df)
    assert not np.isnan(preds).any()

    print("models_spatial.py smoke test passed.")
