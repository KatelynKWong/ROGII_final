from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

if __package__ in {None, ""}:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.pipeline import AbstractBaseModel, FeaturePipeline

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore


class _BiLSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class _TorchSequenceRegressor:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        random_state: int,
    ) -> None:
        torch.manual_seed(random_state)
        self.device = torch.device("cpu")
        self.net = _BiLSTMNet(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchSequenceRegressor":
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
            "backend": "torch_bilstm",
            "state_dict_keys": sorted(self.net.state_dict().keys()),
        }


class DeepSequenceModel(AbstractBaseModel):
    """Sequence learner that respects well-level grouping and causal windows."""

    FAMILY_LABEL = "Family B"
    BACKEND_DISPLAY_NAME = "BiLSTM"

    def __init__(
        self,
        feature_pipeline: Optional[FeaturePipeline] = None,
        group_col: str = "WELLNAME",
        md_col: str = "MD",
        target_col: str = "TVT",
        target_input_col: str = "TVT_input",
        sequence_length: int = 16,
        hidden_size: int = 32,
        epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        n_splits: int = 5,
        scale_target: bool = True,
        architecture: str = "bilstm",
        random_state: int = 42,
        metrics_path: str | Path | None = None,
    ) -> None:
        self.group_col = group_col
        self.md_col = md_col
        self.target_col = target_col
        self.target_input_col = target_input_col
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.n_splits = int(n_splits)
        self.scale_target = bool(scale_target)
        self.architecture = architecture
        self.random_state = int(random_state)
        self.metrics_path = Path(metrics_path) if metrics_path else None

        self.feature_pipeline = feature_pipeline or FeaturePipeline(
            group_col=group_col,
            md_col=md_col,
            target_col=target_col,
            target_input_col=target_input_col,
            scale_target=scale_target,
        )
        self.feature_pipeline.scale_target = self.scale_target

        self.sequence_feature_columns_: List[str] = []
        self.fold_models_: List[Dict[str, Any]] = []
        self.full_model_: Dict[str, Any] = {}
        self.fold_scores_: List[float] = []
        self.oof_predictions_sequence: Optional[np.ndarray] = None
        self.oof_predictions_sequence_scaled: Optional[np.ndarray] = None
        self.oof_predictions_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: Any, y: Any) -> "DeepSequenceModel":
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
                val_features = self._build_numeric_features(fold_pipeline, val_df, reference_columns=train_features.columns)

                train_seq, train_targets_scaled, _ = self._build_causal_sequences(
                    train_df=train_df,
                    feature_frame=train_features,
                    target=y_train,
                    pipeline=fold_pipeline,
                    row_ids=train_idx,
                )
                val_seq, _, val_order = self._build_causal_sequences(
                    train_df=val_df,
                    feature_frame=val_features,
                    target=y_val,
                    pipeline=fold_pipeline,
                    row_ids=val_idx,
                )

                backend = self._make_backend(input_size=train_seq.shape[-1], seed=self.random_state + fold_idx)
                backend.fit(train_seq, train_targets_scaled)
                val_pred_scaled = backend.predict(val_seq)
                val_pred = fold_pipeline.inverse_transform_target(val_pred_scaled)

                oof_scaled[val_order] = val_pred_scaled
                oof_original[val_order] = val_pred

                fold_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
                self.fold_scores_.append(fold_rmse)
                self.fold_models_.append(
                    {
                        "fold_index": fold_idx,
                        "pipeline": fold_pipeline,
                        "backend": backend,
                        "feature_columns": list(train_features.columns),
                        "fold_rmse": fold_rmse,
                        "backend_state": backend.get_state(),
                    }
                )
                fold_bar.update(1)

        if np.isnan(oof_original).any():
            raise RuntimeError("OOF predictions for the sequence model contain unfilled rows.")

        self.oof_predictions_sequence_scaled = oof_scaled
        self.oof_predictions_sequence = oof_original
        self.oof_predictions_ = oof_original

        full_pipeline = deepcopy(self.feature_pipeline)
        full_pipeline.scale_target = self.scale_target
        full_pipeline.fit(df, y=target)
        full_features = self._build_numeric_features(full_pipeline, df)
        full_seq, full_targets_scaled, _ = self._build_causal_sequences(
            train_df=df,
            feature_frame=full_features,
            target=target,
            pipeline=full_pipeline,
            row_ids=np.arange(len(df)),
        )
        full_backend = self._make_backend(input_size=full_seq.shape[-1], seed=self.random_state)
        full_backend.fit(full_seq, full_targets_scaled)

        self.full_model_ = {
            "pipeline": full_pipeline,
            "backend": full_backend,
            "feature_columns": list(full_features.columns),
            "backend_state": full_backend.get_state(),
        }

        self.sequence_feature_columns_ = list(full_features.columns)
        self.is_fitted_ = True
        self._log_training_summary()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("DeepSequenceModel must be fit before calling predict.")
        if not self.full_model_:
            raise RuntimeError("Full sequence model is unavailable.")

        df = self._ensure_dataframe(X)
        self._validate_groups(df)

        pipeline = self.full_model_["pipeline"]
        feature_columns = self.full_model_["feature_columns"]
        feature_frame = self._build_numeric_features(pipeline, df, reference_columns=feature_columns)
        seq, _, order = self._build_causal_sequences(
            train_df=df,
            feature_frame=feature_frame,
            target=None,
            pipeline=pipeline,
            row_ids=np.arange(len(df)),
        )
        pred_scaled = self.full_model_["backend"].predict(seq)
        pred = pipeline.inverse_transform_target(pred_scaled)

        output = np.empty(len(df), dtype=float)
        output[order] = pred
        return output

    def predict_oof(self) -> np.ndarray:
        if self.oof_predictions_sequence is None:
            raise RuntimeError("OOF predictions are not available before fit.")
        return self.oof_predictions_sequence

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("DeepSequenceModel expects a pandas DataFrame.")

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
            raise ValueError(f"DeepSequenceModel requires '{self.group_col}' for GroupKFold.")
        if self.md_col not in df.columns:
            raise ValueError(f"DeepSequenceModel requires '{self.md_col}' for sequence ordering.")

    def _build_numeric_features(
        self,
        pipeline: FeaturePipeline,
        df: pd.DataFrame,
        reference_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        feature_frame = pipeline.get_numeric_feature_frame(df, target_col=self.target_col, fillna=0.0)
        if reference_columns is not None:
            feature_frame = feature_frame.reindex(columns=list(reference_columns), fill_value=0.0)
        return feature_frame

    def _build_causal_sequences(
        self,
        train_df: pd.DataFrame,
        feature_frame: pd.DataFrame,
        target: Optional[np.ndarray],
        pipeline: FeaturePipeline,
        row_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        working = train_df.copy()
        if row_ids is None:
            row_ids = np.arange(len(working))
        row_ids = np.asarray(row_ids, dtype=int)
        if len(row_ids) != len(working):
            raise ValueError("row_ids must match the number of rows used to build sequences.")
        working["_row_id"] = row_ids
        sort_cols = [self.group_col, self.md_col, "_row_id"]
        working = working.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        feature_values = feature_frame.to_numpy(dtype=float, copy=True)
        if len(feature_values) != len(working):
            raise ValueError("feature_frame must align one-to-one with the provided rows.")

        id_to_position = {int(row_id): pos for pos, row_id in enumerate(row_ids)}

        sequences: List[np.ndarray] = []
        targets: List[float] = []
        ordered_indices: List[int] = []

        for _, group in working.groupby(self.group_col, sort=False):
            group_row_ids = group["_row_id"].to_numpy(dtype=int)
            ordered_positions = np.asarray([id_to_position[int(row_id)] for row_id in group_row_ids], dtype=int)
            ordered_features = feature_values[ordered_positions]
            ordered_targets = target[ordered_positions] if target is not None else None

            for pos in range(len(group_row_ids)):
                start = max(0, pos - self.sequence_length + 1)
                window = ordered_features[start : pos + 1]
                if len(window) < self.sequence_length:
                    pad = np.zeros((self.sequence_length - len(window), ordered_features.shape[1]), dtype=float)
                    window = np.vstack([pad, window])
                sequences.append(window.astype(np.float32, copy=False))
                ordered_indices.append(int(group_row_ids[pos]))
                if ordered_targets is not None:
                    targets.append(float(ordered_targets[pos]))

        seq_array = np.asarray(sequences, dtype=np.float32)
        target_array = np.asarray(targets, dtype=float) if target is not None else None
        order_array = np.asarray(ordered_indices, dtype=int)

        if target is not None:
            target_array = pipeline.transform_target(target_array)
        return seq_array, target_array, order_array

    def _make_backend(self, input_size: int, seed: int) -> Any:
        if self.architecture.lower() not in {"bilstm", "lstm"}:
            raise ValueError("DeepSequenceModel currently supports only the BiLSTM architecture.")
        return _TorchSequenceRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            random_state=seed,
        )

    def _log_training_summary(self) -> None:
        if self.metrics_path is None:
            return

        summary = {
            "model_family": "DeepSequenceModel",
            "n_splits": self.n_splits,
            "sequence_length": self.sequence_length,
            "scale_target": self.scale_target,
            "fold_rmse_mean": float(np.mean(self.fold_scores_)) if self.fold_scores_ else None,
            "backend": self.full_model_.get("backend_state", {}),
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
    rng = np.random.default_rng(17)
    wells = [f"WELL_{i}" for i in range(1, 6)]
    rows_per_well = 10
    frames = []

    for well_idx, wellname in enumerate(wells):
        md = np.arange(rows_per_well, dtype=float)
        x = 1000.0 + well_idx * 20.0 + np.cumsum(rng.normal(1.0, 0.05, size=rows_per_well))
        y = 1500.0 + well_idx * 12.0 + np.cumsum(rng.normal(0.8, 0.05, size=rows_per_well))
        z = 4000.0 - np.cumsum(np.abs(rng.normal(0.45, 0.03, size=rows_per_well)))
        gr = 75.0 + np.sin(md / 2.5) * 5.0 + rng.normal(0.0, 0.4, size=rows_per_well)
        tvt = 10.0 + well_idx + md * 0.2 + rng.normal(0.0, 0.1, size=rows_per_well)

        frames.append(
            pd.DataFrame(
                {
                    "WELLNAME": wellname,
                    "MD": md,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "ANCC": z + 25.0,
                    "ASTNU": z + 20.0,
                    "ASTNL": z + 15.0,
                    "EGFDU": z + 10.0,
                    "EGFDL": z + 5.0,
                    "BUDA": z + 2.0,
                    "GR": gr,
                    "TVT": tvt,
                    "TVT_input": np.where(md < 5, tvt, np.nan),
                }
            )
        )

    mock_df = pd.concat(frames, ignore_index=True)
    model = DeepSequenceModel(metrics_path=None)
    model.fit(mock_df, mock_df["TVT"].to_numpy())
    preds = model.predict(mock_df)

    assert model.oof_predictions_sequence is not None
    assert len(model.oof_predictions_sequence) == len(mock_df)
    assert not np.isnan(model.oof_predictions_sequence).any()
    assert model.full_model_["backend_state"]["backend"] == "torch_bilstm"
    assert len(preds) == len(mock_df)
    assert not np.isnan(preds).any()

    print("models_sequences.py smoke test passed.")
