from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - notebook execution shim
    ROOT = Path(os.getcwd()).resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")

from src.pipeline import AbstractBaseModel, FeaturePipeline


class _TabularMLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _TorchTabularRegressor:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        random_state: int,
    ) -> None:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))

        self.device = _select_device()
        self.net = _TabularMLPNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchTabularRegressor":
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                preds = self.net(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32, device=self.device)
            preds = self.net(tensor_x).detach().cpu().numpy()
        return np.asarray(preds, dtype=float).reshape(-1)

    def get_state(self) -> Dict[str, Any]:
        return {
            "backend": "tabular_mlp",
            "state_dict_keys": sorted(self.net.state_dict().keys()),
        }


class DeepTabularModel(AbstractBaseModel):
    """PyTorch MLP for well-level tabular TVT regression."""

    FAMILY_LABEL = "Family F"
    BACKEND_DISPLAY_NAME = "Tabular MLP"

    def __init__(
        self,
        feature_pipeline: Optional[FeaturePipeline] = None,
        group_col: str = "WELLNAME",
        target_col: str = "TVT",
        target_input_col: str = "TVT_input",
        n_splits: int = 5,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.15,
        epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        scale_target: bool = True,
        random_state: int = 42,
        metrics_path: str | Path | None = None,
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.target_input_col = target_input_col
        self.n_splits = int(n_splits)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
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
        self.fold_models_: List[Dict[str, Any]] = []
        self.full_model_: Dict[str, Any] = {}
        self.fold_scores_: List[float] = []
        self.oof_predictions_tabular_mlp: Optional[np.ndarray] = None
        self.scaled_oof_predictions_tabular_mlp: Optional[np.ndarray] = None
        self.oof_predictions_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: Any, y: Any) -> "DeepTabularModel":
        df = self._ensure_dataframe(X)
        target = self._resolve_target(df, y)
        self._validate_groups(df)

        groups = df[self.group_col].astype(str).to_numpy()
        n_unique_groups = len(np.unique(groups))
        if n_unique_groups < 2:
            raise ValueError("Need at least two unique wells for GroupKFold.")

        splitter = GroupKFold(n_splits=min(self.n_splits, n_unique_groups))
        indices = np.arange(len(df))
        folds = list(splitter.split(indices, groups=groups))
        n_folds = len(folds)

        oof_scaled = np.full(len(df), np.nan, dtype=float)
        oof_original = np.full(len(df), np.nan, dtype=float)

        self.fold_models_ = []
        self.fold_scores_ = []

        with tqdm(
            total=n_folds,
            desc=f"Training {self.FAMILY_LABEL}: {self.BACKEND_DISPLAY_NAME}",
            dynamic_ncols=True,
            leave=False,
        ) as fold_bar:
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                fold_bar.set_description(
                    f"Training {self.FAMILY_LABEL}: {self.BACKEND_DISPLAY_NAME} | Fold {fold_idx + 1}/{n_folds}"
                )
                train_df = df.iloc[train_idx].copy()
                val_df = df.iloc[val_idx].copy()
                y_train = target[train_idx]
                y_val = target[val_idx]

                fold_pipeline = deepcopy(self.feature_pipeline)
                fold_pipeline.scale_target = self.scale_target
                fold_pipeline.fit(train_df, y=y_train)

                train_features = self._build_numeric_features(fold_pipeline, train_df)
                val_features = self._build_numeric_features(
                    fold_pipeline,
                    val_df,
                    reference_columns=train_features.columns,
                )

                feature_scaler = StandardScaler()
                train_features_scaled = feature_scaler.fit_transform(train_features)
                val_features_scaled = feature_scaler.transform(val_features)

                y_train_scaled = self._scale_target(y_train, pipeline=fold_pipeline)
                backend = self._make_backend(
                    input_dim=train_features_scaled.shape[1],
                    seed=self.random_state + fold_idx,
                )
                backend.fit(train_features_scaled, y_train_scaled)

                val_pred_scaled = backend.predict(val_features_scaled)
                val_pred = fold_pipeline.inverse_transform_target(val_pred_scaled)

                oof_scaled[val_idx] = val_pred_scaled
                oof_original[val_idx] = val_pred

                fold_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
                self.fold_scores_.append(fold_rmse)
                self.fold_models_.append(
                    {
                        "fold_index": fold_idx,
                        "pipeline": fold_pipeline,
                        "feature_scaler": feature_scaler,
                        "backend": backend,
                        "feature_columns": list(train_features.columns),
                        "fold_rmse": fold_rmse,
                        "backend_state": backend.get_state(),
                    }
                )
                fold_bar.update(1)

        if np.isnan(oof_original).any():
            raise RuntimeError("OOF predictions for the tabular MLP contain unfilled rows.")

        self.scaled_oof_predictions_tabular_mlp = oof_scaled
        self.oof_predictions_tabular_mlp = oof_original
        self.oof_predictions_ = oof_original

        full_pipeline = deepcopy(self.feature_pipeline)
        full_pipeline.scale_target = self.scale_target
        full_pipeline.fit(df, y=target)
        full_features = self._build_numeric_features(full_pipeline, df)
        full_scaler = StandardScaler()
        full_features_scaled = full_scaler.fit_transform(full_features)
        full_target_scaled = self._scale_target(target, pipeline=full_pipeline)

        full_backend = self._make_backend(
            input_dim=full_features_scaled.shape[1],
            seed=self.random_state,
        )
        full_backend.fit(full_features_scaled, full_target_scaled)

        self.feature_columns_ = list(full_features.columns)
        self.full_model_ = {
            "pipeline": full_pipeline,
            "feature_scaler": full_scaler,
            "backend": full_backend,
            "feature_columns": list(full_features.columns),
            "backend_state": full_backend.get_state(),
        }

        self.is_fitted_ = True
        self._log_training_summary()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("DeepTabularModel must be fit before calling predict.")
        df = self._ensure_dataframe(X)
        model_bundle = self.full_model_
        feature_frame = self._build_numeric_features(
            model_bundle["pipeline"],
            df,
            reference_columns=model_bundle["feature_columns"],
        )
        feature_frame_scaled = model_bundle["feature_scaler"].transform(feature_frame)
        pred_scaled = model_bundle["backend"].predict(feature_frame_scaled)
        return self._unscale_target(pred_scaled, pipeline=model_bundle["pipeline"])

    def predict_oof(self) -> np.ndarray:
        if self.oof_predictions_tabular_mlp is None:
            raise RuntimeError("OOF predictions are not available before fit.")
        return self.oof_predictions_tabular_mlp

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if not self.feature_columns_:
                raise ValueError("NumPy input is only supported after fitting on a DataFrame.")
            return pd.DataFrame(X, columns=self.feature_columns_)
        raise TypeError("DeepTabularModel expects a pandas DataFrame or NumPy array.")

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
            raise ValueError(f"DeepTabularModel requires '{self.group_col}' for GroupKFold.")

    def _build_numeric_features(
        self,
        pipeline: FeaturePipeline,
        df: pd.DataFrame,
        reference_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        feature_frame = pipeline.get_numeric_feature_frame(df, target_col=self.target_col, fillna=0.0)
        if reference_columns is not None:
            feature_frame = feature_frame.reindex(columns=list(reference_columns), fill_value=0.0)
        return feature_frame.astype(float)

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

    def _make_backend(self, input_dim: int, seed: int) -> _TorchTabularRegressor:
        return _TorchTabularRegressor(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            random_state=seed,
        )

    def _log_training_summary(self) -> None:
        if self.metrics_path is None:
            return

        summary = {
            "model_family": "DeepTabularModel",
            "n_splits": self.n_splits,
            "scale_target": self.scale_target,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "fold_rmse_mean": float(np.mean(self.fold_scores_)) if self.fold_scores_ else None,
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


TabularMLPModel = DeepTabularModel


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    wells = [f"WELL_{i}" for i in range(1, 6)]
    rows_per_well = 9
    frames = []

    for well_idx, wellname in enumerate(wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 4700.0 - np.cumsum(np.abs(rng.normal(0.38, 0.02, size=rows_per_well)))
        frames.append(
            pd.DataFrame(
                {
                    "WELLNAME": wellname,
                    "MD": md,
                    "X": 1200.0 + well_idx * 28.0 + np.cumsum(rng.normal(1.0, 0.08, size=rows_per_well)),
                    "Y": 1600.0 + well_idx * 16.0 + np.cumsum(rng.normal(0.8, 0.08, size=rows_per_well)),
                    "Z": z,
                    "ANCC": z + 18.0,
                    "ASTNU": z + 14.0,
                    "ASTNL": z + 11.0,
                    "EGFDU": z + 8.0,
                    "EGFDL": z + 5.0,
                    "BUDA": z + 2.0,
                    "GR": 69.0 + np.sin(md / 2.0) * 3.5 + rng.normal(0.0, 0.4, size=rows_per_well),
                    "TVT": 19.0 + well_idx + md * 0.27 + rng.normal(0.0, 0.08, size=rows_per_well),
                    "TVT_input": np.where(md < 4, 0.9, np.nan),
                }
            )
        )

    mock_df = pd.concat(frames, ignore_index=True)
    model = DeepTabularModel(metrics_path=None, epochs=5)
    model.fit(mock_df, mock_df["TVT"].to_numpy())
    preds = model.predict(mock_df)

    assert len(model.oof_predictions_tabular_mlp) == len(mock_df)
    assert not np.isnan(model.oof_predictions_tabular_mlp).any()
    assert not np.isnan(preds).any()
    print("models_tabnet.py smoke test passed.")
