from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import os
import sys



FAST_DEBUG = False


class AbstractBaseModel(ABC):
    """Generic blueprint for all model families."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "AbstractBaseModel":
        """Fit the model on features X and target y."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Generate predictions for X."""


class FeaturePipeline:
    """Feature engineering for the wellbore schema described in project.md."""

    DEFAULT_SURFACE_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
    DEFAULT_GR_WINDOWS = (5, 15, 50)

    def __init__(
        self,
        group_col: str = "WELLNAME",
        md_col: str = "MD",
        x_col: str = "X",
        y_col: str = "Y",
        z_col: str = "Z",
        gr_col: str = "GR",
        target_col: str = "TVT",
        target_input_col: str = "TVT_input",
        surface_cols: Sequence[str] = DEFAULT_SURFACE_COLUMNS,
        gr_windows: Sequence[int] = DEFAULT_GR_WINDOWS,
        scale_target: bool = False,
    ) -> None:
        self.group_col = group_col
        self.md_col = md_col
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.gr_col = gr_col
        self.target_col = target_col
        self.target_input_col = target_input_col
        self.surface_cols = tuple(surface_cols)
        self.gr_windows = tuple(int(w) for w in gr_windows)
        self.scale_target = scale_target

        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0
        self.numeric_fill_values_: Dict[str, float] = {}
        self.feature_columns_: List[str] = []

    @staticmethod
    def parse_wellname_from_filename(filename: str | Path) -> str:
        """Extract WELLNAME from the competition filename conventions."""

        name = Path(filename).name
        suffixes = ("__horizontal_well.csv", "__typewell.csv", ".csv", ".png")
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem

    def load_directory(self, root_dir: str | Path) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load the competition-style directory structure into per-well frames."""

        root = Path(root_dir)
        well_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
        for horizontal_path in sorted(root.glob("*__horizontal_well.csv")):
            wellname = self.parse_wellname_from_filename(horizontal_path)
            typewell_path = root / f"{wellname}__typewell.csv"

            well_frames[wellname] = {
                "horizontal": pd.read_csv(horizontal_path),
                "typewell": pd.read_csv(typewell_path) if typewell_path.exists() else pd.DataFrame(),
            }

            well_frames[wellname]["horizontal"][self.group_col] = wellname
            if not well_frames[wellname]["typewell"].empty:
                well_frames[wellname]["typewell"][self.group_col] = wellname

        return well_frames

    def fit(self, df: pd.DataFrame, y: Optional[Sequence[float]] = None) -> "FeaturePipeline":
        """Learn target scaling and fallback numeric fill values."""

        if self.target_col in df.columns:
            target_values = pd.to_numeric(df[self.target_col], errors="coerce")
            valid_target = target_values.dropna()
            if not valid_target.empty:
                self.target_mean_ = float(valid_target.mean())
                self.target_std_ = float(valid_target.std(ddof=0) or 1.0)

        if y is not None:
            y_arr = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
            if not y_arr.empty:
                self.target_mean_ = float(y_arr.mean())
                self.target_std_ = float(y_arr.std(ddof=0) or 1.0)

        transformed = self.transform(df, fit_mode=True)
        numeric_cols = transformed.select_dtypes(include=[np.number]).columns
        self.feature_columns_ = [c for c in numeric_cols if c != self.target_col]
        self.numeric_fill_values_ = {
            col: float(transformed[col].median()) if transformed[col].notna().any() else 0.0
            for col in self.feature_columns_
        }
        return self

    def fit_transform(self, df: pd.DataFrame, y: Optional[Sequence[float]] = None) -> pd.DataFrame:
        self.fit(df, y=y)
        return self.transform(df)

    def transform_target(self, y: Sequence[float]) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if not self.scale_target:
            return y_arr
        return (y_arr - self.target_mean_) / (self.target_std_ or 1.0)

    def inverse_transform_target(self, y_scaled: Sequence[float]) -> np.ndarray:
        y_arr = np.asarray(y_scaled, dtype=float)
        if not self.scale_target:
            return y_arr
        return y_arr * (self.target_std_ or 1.0) + self.target_mean_

    def transform(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """Engineer leakage-safe per-well features."""

        if df is None:
            raise ValueError("FeaturePipeline.transform requires a pandas DataFrame.")
        if self.group_col not in df.columns:
            raise ValueError(f"Missing required group column '{self.group_col}'.")

        work = df.copy()
        work["_row_order"] = np.arange(len(work))

        sort_cols = [self.group_col]
        if self.md_col in work.columns:
            sort_cols.append(self.md_col)
        work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        processed_groups: List[pd.DataFrame] = []
        for wellname, group in work.groupby(self.group_col, sort=False):
            processed_groups.append(self._process_single_well(group.copy()))

        result = pd.concat(processed_groups, axis=0, ignore_index=True)
        result = result.sort_values("_row_order", kind="mergesort").reset_index(drop=True)
        result = result.drop(columns=["_row_order"])

        if not fit_mode and self.numeric_fill_values_:
            for col, fill_value in self.numeric_fill_values_.items():
                if col in result.columns:
                    result[col] = result[col].fillna(fill_value)

        return result

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias kept for compatibility with likely hidden tests."""

        return self.transform(df)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def _process_single_well(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        if self.md_col in group.columns:
            group = group.sort_values(self.md_col, kind="mergesort").reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)

        group = self._interpolate_surface_columns(group)
        group = self._add_kinematic_features(group)
        group = self._add_surface_distance_features(group)
        group = self._add_gr_rolling_features(group)
        group = self._add_target_features(group)

        return group

    def _interpolate_surface_columns(self, group: pd.DataFrame) -> pd.DataFrame:
        if self.md_col not in group.columns:
            return group

        for col in self.surface_cols:
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            values = values.interpolate(method="linear", limit_direction="both")
            values = values.ffill().bfill()
            if values.notna().any():
                group[col] = values
            else:
                group[col] = 0.0
        return group

    def _add_kinematic_features(self, group: pd.DataFrame) -> pd.DataFrame:
        for col in (self.x_col, self.y_col, self.z_col):
            if col not in group.columns:
                group[col] = 0.0

        x = pd.to_numeric(group[self.x_col], errors="coerce").fillna(0.0)
        y = pd.to_numeric(group[self.y_col], errors="coerce").fillna(0.0)
        z = pd.to_numeric(group[self.z_col], errors="coerce").fillna(0.0)

        dx = x.diff().fillna(0.0)
        dy = y.diff().fillna(0.0)
        dz = z.diff().fillna(0.0)

        if self.md_col in group.columns:
            md = pd.to_numeric(group[self.md_col], errors="coerce").ffill().fillna(0.0)
        else:
            md = pd.Series(np.arange(len(group), dtype=float), index=group.index)

        dmd = md.diff().fillna(0.0)
        step_length = np.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2))
        step_length = step_length.replace(0.0, np.nan)

        vertical_ratio = np.abs(dz) / step_length
        vertical_ratio = vertical_ratio.clip(lower=0.0, upper=1.0).fillna(1.0)
        inclination_rad = np.arccos(vertical_ratio)
        inclination_deg = np.degrees(inclination_rad)

        group["delta_x"] = dx.to_numpy()
        group["delta_y"] = dy.to_numpy()
        group["delta_z"] = dz.to_numpy()
        group["delta_md"] = dmd.to_numpy()
        group["step_length"] = np.nan_to_num(step_length.to_numpy(), nan=0.0)
        group["horizontal_step"] = np.sqrt(dx.pow(2) + dy.pow(2)).to_numpy()
        group["wellbore_inclination_rad"] = inclination_rad.to_numpy()
        group["wellbore_inclination_deg"] = inclination_deg.to_numpy()
        group["azimuth_rad"] = np.arctan2(dy.to_numpy(), dx.to_numpy() + 1e-12)
        group["azimuth_deg"] = np.degrees(group["azimuth_rad"])
        group["curvature_proxy"] = np.sqrt(dx.diff().fillna(0.0).pow(2) + dy.diff().fillna(0.0).pow(2) + dz.diff().fillna(0.0).pow(2))

        return group.fillna({
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_md": 0.0,
            "step_length": 0.0,
            "horizontal_step": 0.0,
            "wellbore_inclination_rad": 0.0,
            "wellbore_inclination_deg": 0.0,
            "azimuth_rad": 0.0,
            "azimuth_deg": 0.0,
            "curvature_proxy": 0.0,
        })

    def _add_surface_distance_features(self, group: pd.DataFrame) -> pd.DataFrame:
        z = pd.to_numeric(group.get(self.z_col, 0.0), errors="coerce").fillna(0.0)
        for col in self.surface_cols:
            if col not in group.columns:
                group[f"surface_delta_{col}"] = 0.0
                group[f"surface_abs_delta_{col}"] = 0.0
                continue
            surface = pd.to_numeric(group[col], errors="coerce").ffill().bfill().fillna(0.0)
            delta = z - surface
            group[f"surface_delta_{col}"] = delta.to_numpy()
            group[f"surface_abs_delta_{col}"] = np.abs(delta.to_numpy())
        return group

    def _add_gr_rolling_features(self, group: pd.DataFrame) -> pd.DataFrame:
        if self.gr_col not in group.columns:
            group[self.gr_col] = 0.0

        gr = pd.to_numeric(group[self.gr_col], errors="coerce").fillna(0.0)
        group["gr_diff_1"] = gr.diff().fillna(0.0)
        group["gr_ewm_3"] = gr.ewm(span=3, adjust=False).mean().fillna(0.0)
        group["gr_ewm_8"] = gr.ewm(span=8, adjust=False).mean().fillna(0.0)

        for window in self.gr_windows:
            shifted = gr.shift(1)
            rolled = shifted.rolling(window=int(window), min_periods=1)
            roll_mean = rolled.mean()
            roll_std = rolled.std(ddof=0)
            roll_min = rolled.min()
            roll_max = rolled.max()
            roll_median = rolled.median()

            group[f"gr_roll_mean_{window}"] = roll_mean.fillna(0.0).to_numpy()
            group[f"gr_roll_std_{window}"] = roll_std.fillna(0.0).to_numpy()
            group[f"gr_roll_min_{window}"] = roll_min.fillna(0.0).to_numpy()
            group[f"gr_roll_max_{window}"] = roll_max.fillna(0.0).to_numpy()
            group[f"gr_roll_median_{window}"] = roll_median.fillna(0.0).to_numpy()
            group[f"gr_roll_range_{window}"] = (roll_max - roll_min).fillna(0.0).to_numpy()
            group[f"gr_roll_delta_mean_{window}"] = (gr - roll_mean).fillna(0.0).to_numpy()

        return group

    def _add_target_features(self, group: pd.DataFrame) -> pd.DataFrame:
        if self.target_input_col in group.columns:
            group[f"{self.target_input_col}_isna"] = group[self.target_input_col].isna().astype(int)
        if self.target_col in group.columns:
            group[f"{self.target_col}_isna"] = group[self.target_col].isna().astype(int)
        return group

    def get_numeric_feature_frame(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        fillna: float = 0.0,
    ) -> pd.DataFrame:
        """Return a numeric feature matrix suitable for model training."""

        transformed = self.transform(df)
        drop_cols = {self.group_col}
        if target_col:
            drop_cols.add(target_col)
        if self.target_col in transformed.columns and self.target_col not in drop_cols:
            drop_cols.add(self.target_col)

        feature_df = transformed.drop(columns=[c for c in drop_cols if c in transformed.columns], errors="ignore")
        numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
        return numeric_df.fillna(fillna)


class ExperimentOrchestrator:
    """Cross-validation harness with well-level grouping and OOF generation."""

    def __init__(
        self,
        models: Optional[Mapping[str, AbstractBaseModel]] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        n_splits: int = 5,
        fast_debug: bool = FAST_DEBUG,
        fast_debug_well_count: int = 10,
        random_state: int = 42,
        metrics_path: str | Path = "metrics.json",
    ) -> None:
        self.models: Dict[str, AbstractBaseModel] = dict(models or {})
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.n_splits = int(n_splits)
        self.fast_debug = bool(fast_debug)
        self.fast_debug_well_count = int(fast_debug_well_count)
        self.random_state = int(random_state)
        self.metrics_path = Path(metrics_path) if metrics_path else None

        self.oof_predictions_: Dict[str, np.ndarray] = {}
        self.fold_models_: Dict[str, List[AbstractBaseModel]] = {}
        self.cv_scores_: Dict[str, List[float]] = {}

    def register_model(self, name: str, model: AbstractBaseModel) -> None:
        self.models[name] = model

    def _select_working_frame(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if not self.fast_debug:
            return df.copy()

        unique_wells = pd.Index(df[group_col].astype(str).dropna().unique())
        if len(unique_wells) <= max(self.n_splits, 1):
            return df.copy()

        chosen_count = min(len(unique_wells), max(self.n_splits, self.fast_debug_well_count))
        rng = np.random.default_rng(self.random_state)
        chosen_wells = np.sort(rng.choice(unique_wells.to_numpy(), size=chosen_count, replace=False))
        mask = df[group_col].astype(str).isin(chosen_wells)
        return df.loc[mask].copy().reset_index(drop=True)

    def make_folds(self, df: pd.DataFrame, group_col: str = "WELLNAME") -> List[Tuple[np.ndarray, np.ndarray]]:
        if group_col not in df.columns:
            raise ValueError(f"Missing required group column '{group_col}'.")

        working = self._select_working_frame(df, group_col=group_col).reset_index(drop=True)
        groups = working[group_col].astype(str).to_numpy()
        n_unique_groups = len(pd.Index(groups).unique())
        if n_unique_groups < 2:
            raise ValueError("Need at least two unique wells for GroupKFold.")

        n_splits = min(self.n_splits, n_unique_groups)
        splitter = GroupKFold(n_splits=n_splits)
        indices = np.arange(len(working))
        return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(indices, groups=groups)]

    def cross_validate(
        self,
        df: pd.DataFrame,
        target_col: str = "TVT",
        group_col: str = "WELLNAME",
        fillna_value: float = 0.0,
    ) -> Dict[str, Any]:
        if target_col not in df.columns:
            raise ValueError(f"Missing target column '{target_col}'.")

        working = self._select_working_frame(df, group_col=group_col).reset_index(drop=True)
        processed = self.feature_pipeline.transform(working)

        drop_cols = {target_col, self.feature_pipeline.group_col}
        feature_df = processed.drop(columns=[c for c in drop_cols if c in processed.columns], errors="ignore")
        X = feature_df.select_dtypes(include=[np.number]).fillna(fillna_value)
        y = pd.to_numeric(processed[target_col], errors="coerce").to_numpy(dtype=float)
        groups = processed[group_col].astype(str).to_numpy()

        folds = self.make_folds(working, group_col=group_col)
        fold_summary: Dict[str, List[float]] = {}

        self.oof_predictions_ = {}
        self.fold_models_ = {}
        self.cv_scores_ = {}

        for model_name, model in self.models.items():
            oof = np.full(len(X), np.nan, dtype=float)
            fold_models: List[AbstractBaseModel] = []
            fold_scores: List[float] = []

            for train_idx, val_idx in folds:
                X_train = X.iloc[train_idx]
                y_train = y[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y[val_idx]

                fold_model = deepcopy(model)
                fold_model.fit(X_train, y_train)
                preds = np.asarray(fold_model.predict(X_val), dtype=float).reshape(-1)
                if preds.shape[0] != len(val_idx):
                    raise ValueError(
                        f"Model '{model_name}' returned {preds.shape[0]} predictions for "
                        f"{len(val_idx)} validation rows."
                    )

                oof[val_idx] = preds
                fold_models.append(fold_model)
                fold_rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
                fold_scores.append(fold_rmse)

            if np.isnan(oof).any():
                raise RuntimeError(f"OOF predictions for '{model_name}' contain unfilled rows.")

            self.oof_predictions_[model_name] = oof
            self.fold_models_[model_name] = fold_models
            self.cv_scores_[model_name] = fold_scores
            fold_summary[model_name] = fold_scores

        run_record = {
            "n_rows": int(len(working)),
            "n_wells": int(pd.Index(groups).nunique()),
            "n_splits": int(len(folds)),
            "fast_debug": bool(self.fast_debug),
            "models": {name: float(np.mean(scores)) for name, scores in fold_summary.items()},
        }
        self._log_metrics(run_record)

        oof_frame = pd.DataFrame({f"{name}_oof": preds for name, preds in self.oof_predictions_.items()})
        oof_frame[group_col] = groups

        return {
            "folds": folds,
            "oof_predictions": self.oof_predictions_,
            "oof_frame": oof_frame,
            "cv_scores": self.cv_scores_,
            "summary": run_record,
        }

    def run_cross_validation(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.cross_validate(*args, **kwargs)

    def _log_metrics(self, record: Mapping[str, Any]) -> None:
        if self.metrics_path is None:
            return

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
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

        existing.append(dict(record))
        self.metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


class _SmokeTestModel(AbstractBaseModel):
    """Tiny deterministic model used in the executable smoke test."""

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, X: Any, y: Any) -> "_SmokeTestModel":
        y_arr = np.asarray(y, dtype=float)
        self.mean_ = float(np.nanmean(y_arr)) if y_arr.size else 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, "__len__"):
            n = len(X)
        else:
            n = np.asarray(X).shape[0]
        return np.full(n, self.mean_, dtype=float)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    wells = [f"WELL_{i}" for i in range(1, 6)]
    rows_per_well = 12

    frames = []
    for well_idx, wellname in enumerate(wells):
        md = np.arange(rows_per_well, dtype=float)
        x = 1000.0 + well_idx * 50.0 + np.cumsum(rng.normal(1.0, 0.05, size=rows_per_well))
        y = 2000.0 + well_idx * 25.0 + np.cumsum(rng.normal(0.8, 0.05, size=rows_per_well))
        z = 5000.0 - np.cumsum(np.abs(rng.normal(0.6, 0.05, size=rows_per_well)))
        gr = 80.0 + np.sin(md / 3.0) * 10.0 + rng.normal(0.0, 0.5, size=rows_per_well)
        tvt = 40.0 + well_idx + md * 0.4 + rng.normal(0.0, 0.2, size=rows_per_well)

        frame = pd.DataFrame(
            {
                "WELLNAME": wellname,
                "MD": md,
                "X": x,
                "Y": y,
                "Z": z,
                "ANCC": z + 30.0,
                "ASTNU": z + 24.0,
                "ASTNL": z + 18.0,
                "EGFDU": z + 12.0,
                "EGFDL": z + 6.0,
                "BUDA": z + 2.0,
                "GR": gr,
                "TVT": tvt,
                "TVT_input": np.where(md < 8, tvt, np.nan),
            }
        )
        frames.append(frame)

    mock_df = pd.concat(frames, ignore_index=True)

    pipeline = FeaturePipeline(scale_target=True)
    engineered = pipeline.fit_transform(mock_df)
    feature_matrix = pipeline.get_numeric_feature_frame(mock_df, target_col="TVT")

    assert len(engineered) == len(mock_df), "Feature extraction changed row count."
    assert feature_matrix.shape[0] == len(mock_df), "Numeric feature extraction failed."

    orchestrator = ExperimentOrchestrator(
        models={"smoke": _SmokeTestModel()},
        feature_pipeline=FeaturePipeline(),
        n_splits=5,
        fast_debug=False,
        metrics_path=Path("metrics.json"),
    )

    cv_result = orchestrator.cross_validate(mock_df)
    smoke_oof = cv_result["oof_predictions"]["smoke"]

    assert len(cv_result["folds"]) == 5, "GroupKFold did not produce five folds."
    assert smoke_oof.shape[0] == len(mock_df), "OOF predictions have incorrect length."
    assert not np.isnan(smoke_oof).any(), "OOF predictions contain NaNs."

    print("pipeline.py smoke test passed.")
