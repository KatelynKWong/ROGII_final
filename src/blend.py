from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

try:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - notebook execution shim
    ROOT = Path(os.getcwd()).resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - notebook flattening shim
    from src.pipeline import FeaturePipeline, log_family_training_complete, log_family_training_start
    from src.models_baselines import BaselineEnsembleModel
    from src.models_kernels import KernelMachineModel
    from src.models_linear import LinearEnsembleModel
    from src.models_sequences import DeepSequenceModel
    from src.models_spatial import SpatialNeighborModel
    from src.models_tabnet import DeepTabularModel
    from src.models_trees import TreeEnsembleModel
except ModuleNotFoundError:  # pragma: no cover - flattened notebook execution shim
    required_names = {
        "FeaturePipeline",
        "log_family_training_start",
        "log_family_training_complete",
        "BaselineEnsembleModel",
        "KernelMachineModel",
        "LinearEnsembleModel",
        "DeepSequenceModel",
        "SpatialNeighborModel",
        "DeepTabularModel",
        "TreeEnsembleModel",
    }
    missing = sorted(name for name in required_names if name not in globals())
    if missing:
        raise
    FeaturePipeline = globals()["FeaturePipeline"]
    log_family_training_start = globals()["log_family_training_start"]
    log_family_training_complete = globals()["log_family_training_complete"]
    BaselineEnsembleModel = globals()["BaselineEnsembleModel"]
    KernelMachineModel = globals()["KernelMachineModel"]
    LinearEnsembleModel = globals()["LinearEnsembleModel"]
    DeepSequenceModel = globals()["DeepSequenceModel"]
    SpatialNeighborModel = globals()["SpatialNeighborModel"]
    DeepTabularModel = globals()["DeepTabularModel"]
    TreeEnsembleModel = globals()["TreeEnsembleModel"]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


ARTIFACT_DIR = ROOT / "artifacts" / "blend"
OOF_CACHE_PATH = ARTIFACT_DIR / "oof_cache.npz"
OOF_META_PATH = ARTIFACT_DIR / "oof_cache.json"
WEIGHTS_PATH = ARTIFACT_DIR / "meta_weights.json"
SUBMISSION_PATH = ROOT / "submission.csv"
PIPELINE_CONFIG_PATH = ROOT / "pipeline_config.json"

DEFAULT_PIPELINE_CONFIG: Dict[str, Any] = {
    "data": {
        "train_root": "/kaggle/input/competitions/rogii-wellbore-geology-prediction/train",
        "test_root": "/kaggle/input/competitions/rogii-wellbore-geology-prediction/test",
        "submission_path": "submission.csv",
        "max_wells": None,
        "row_cap": None,
    },
    "environment": {
        "is_notebook_runtime": True,
    },
    "active_families": {
        "tree_models": True,
        "sequence_models": False,
        "linear_models": True,
        "spatial_models": False,
        "kernel_models": True,
        "tabular_models": True,
        "baseline_models": False,
    },
    "sub_models": {
        "run_heavy_baseline_trees": False,
    },
}

FAMILY_CONFIG_FLAGS = {
    "tree": "tree_models",
    "sequence": "sequence_models",
    "linear": "linear_models",
    "spatial": "spatial_models",
    "kernels": "kernel_models",
    "tabular": "tabular_models",
    "baseline": "baseline_models",
}


@dataclass(frozen=True)
class PeerSpec:
    family: str
    backend: str
    display_name: str


PEER_SPECS: Tuple[PeerSpec, ...] = (
    PeerSpec("tree", "lightgbm", "tree_lightgbm"),
    PeerSpec("tree", "catboost", "tree_catboost"),
    PeerSpec("tree", "xgboost", "tree_xgboost"),
    PeerSpec("sequence", "sequence", "sequence_bilstm"),
    PeerSpec("spatial", "knn_5", "spatial_knn_5"),
    PeerSpec("spatial", "knn_15", "spatial_knn_15"),
    PeerSpec("spatial", "knn_30", "spatial_knn_30"),
    PeerSpec("kernels", "svr_rbf", "kernels_svr_rbf"),
    PeerSpec("kernels", "svr_linear", "kernels_svr_linear"),
    PeerSpec("tabular", "tabular_mlp", "tabular_mlp"),
    PeerSpec("linear", "ridge", "linear_ridge"),
    PeerSpec("linear", "lasso", "linear_lasso"),
    PeerSpec("linear", "elasticnet", "linear_elasticnet"),
    PeerSpec("baseline", "rf", "baseline_rf"),
    PeerSpec("baseline", "et", "baseline_et"),
    PeerSpec("baseline", "hist", "baseline_hist"),
)


def _deep_merge_dict(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: str | Path = PIPELINE_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path

    loaded: Dict[str, Any] = {}
    if path.exists():
        loaded_data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded_data, dict):
            loaded = loaded_data
        else:
            raise ValueError(f"Pipeline config must be a JSON object: {path}")

    return _deep_merge_dict(DEFAULT_PIPELINE_CONFIG, loaded)


def _family_is_active(active_families: Mapping[str, Any], family_key: str) -> bool:
    config_flag = FAMILY_CONFIG_FLAGS[family_key]
    return bool(active_families.get(config_flag, DEFAULT_PIPELINE_CONFIG["active_families"][config_flag]))


def build_active_peer_specs(
    active_families: Mapping[str, Any],
    sub_models: Mapping[str, Any],
) -> Tuple[PeerSpec, ...]:
    active_specs: List[PeerSpec] = []
    run_heavy_baseline_trees = bool(
        sub_models.get(
            "run_heavy_baseline_trees",
            DEFAULT_PIPELINE_CONFIG["sub_models"]["run_heavy_baseline_trees"],
        )
    )

    for spec in PEER_SPECS:
        if not _family_is_active(active_families, spec.family):
            continue
        if spec.family == "baseline" and not run_heavy_baseline_trees and spec.display_name in {"baseline_rf", "baseline_et"}:
            continue
        active_specs.append(spec)

    return tuple(active_specs)


class MetaBlender:
    """Constrained simplex optimizer for blending peer OOF predictions."""

    def __init__(self, peer_names: Optional[Sequence[str]] = None) -> None:
        self.peer_names_: List[str] = list(peer_names or [])
        self.weights_: Optional[np.ndarray] = None
        self.weight_map_: Dict[str, float] = {}
        self.ensemble_oof_rmse_: Optional[float] = None
        self.ensemble_oof_rmse_original_: Optional[float] = None
        self.optimizer_result_: Any = None

    def fit(self, oof_matrix: np.ndarray, target_scaled: np.ndarray, peer_names: Optional[Sequence[str]] = None) -> "MetaBlender":
        matrix = np.asarray(oof_matrix, dtype=float)
        target = np.asarray(target_scaled, dtype=float).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("oof_matrix must be two-dimensional.")
        if matrix.shape[0] != len(target):
            raise ValueError("oof_matrix and target_scaled must have the same number of rows.")

        if peer_names is not None:
            self.peer_names_ = list(peer_names)
        if not self.peer_names_:
            self.peer_names_ = [f"peer_{idx}" for idx in range(matrix.shape[1])]
        if len(self.peer_names_) != matrix.shape[1]:
            raise ValueError("peer_names length must match the number of OOF columns.")

        def objective(weights: np.ndarray) -> float:
            blended = matrix @ weights
            return float(np.sqrt(np.mean((target - blended) ** 2)))

        x0 = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=float)
        bounds = [(0.0, 1.0)] * matrix.shape[1]
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )

        weights = np.asarray(result.x if result.success else x0, dtype=float).reshape(-1)
        weights = np.clip(weights, 0.0, 1.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = x0.copy()
            weight_sum = float(weights.sum())
        weights = weights / weight_sum

        self.weights_ = weights
        self.weight_map_ = {name: float(weight) for name, weight in zip(self.peer_names_, weights)}
        self.ensemble_oof_rmse_ = float(objective(weights))
        self.optimizer_result_ = result
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("MetaBlender must be fit before calling predict.")
        return np.asarray(matrix, dtype=float) @ self.weights_

    def to_dict(self) -> Dict[str, float]:
        if not self.weight_map_:
            raise RuntimeError("MetaBlender must be fit before calling to_dict.")
        return dict(self.weight_map_)


def load_competition_frames(root: str | Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    pipeline = FeaturePipeline()
    return pipeline.load_directory(root)


def _extract_typewell_reference_signal(typewell: pd.DataFrame) -> np.ndarray:
    if typewell is None or typewell.empty:
        return np.asarray([], dtype=float)

    preferred_columns = ("GR", "TVT_input", "TVT")
    for column in preferred_columns:
        if column in typewell.columns:
            signal = pd.to_numeric(typewell[column], errors="coerce")
            break
    else:
        numeric = typewell.select_dtypes(include=[np.number])
        if numeric.empty:
            return np.asarray([], dtype=float)
        signal = pd.to_numeric(numeric.iloc[:, 0], errors="coerce")

    signal = signal.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
    return signal.to_numpy(dtype=float)


def _normalized_window_corr(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float).reshape(-1)
    right_arr = np.asarray(right, dtype=float).reshape(-1)
    length = min(len(left_arr), len(right_arr))
    if length < 2:
        return 0.0
    left_arr = left_arr[-length:]
    right_arr = right_arr[-length:]
    left_std = float(np.nanstd(left_arr))
    right_std = float(np.nanstd(right_arr))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    left_centered = left_arr - np.nanmean(left_arr)
    right_centered = right_arr - np.nanmean(right_arr)
    denom = float(np.sqrt(np.sum(left_centered ** 2) * np.sum(right_centered ** 2)))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(left_centered * right_centered) / denom)


def _add_typewell_reference_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    window_sizes: Sequence[int] = FeaturePipeline.DEFAULT_GR_WINDOWS,
) -> pd.DataFrame:
    work = horizontal.copy().reset_index(drop=True)
    if "GR" in work.columns:
        gr = pd.to_numeric(work["GR"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        gr = np.zeros(len(work), dtype=float)

    ref_signal = _extract_typewell_reference_signal(typewell)
    if len(work) == 0:
        for window in window_sizes:
            work[f"gr_typewell_forward_corr_{window}"] = pd.Series(dtype=float)
            work[f"gr_typewell_reverse_corr_{window}"] = pd.Series(dtype=float)
            work[f"gr_typewell_corr_gap_{window}"] = pd.Series(dtype=float)
        return work

    if ref_signal.size == 0:
        for window in window_sizes:
            work[f"gr_typewell_forward_corr_{window}"] = 0.0
            work[f"gr_typewell_reverse_corr_{window}"] = 0.0
            work[f"gr_typewell_corr_gap_{window}"] = 0.0
        return work

    ref_reverse = ref_signal[::-1]
    n_rows = len(work)
    denom = max(n_rows - 1, 1)

    for window in window_sizes:
        window = int(window)
        forward_corrs: List[float] = []
        reverse_corrs: List[float] = []
        for idx in range(n_rows):
            h_window = gr[max(0, idx - window + 1) : idx + 1]
            ref_idx = int(round((idx / denom) * max(len(ref_signal) - 1, 0)))
            forward_slice = ref_signal[max(0, ref_idx - len(h_window) + 1) : ref_idx + 1]
            reverse_slice = ref_reverse[max(0, ref_idx - len(h_window) + 1) : ref_idx + 1]
            forward_corrs.append(_normalized_window_corr(h_window, forward_slice))
            reverse_corrs.append(_normalized_window_corr(h_window, reverse_slice))
        work[f"gr_typewell_forward_corr_{window}"] = np.asarray(forward_corrs, dtype=float)
        work[f"gr_typewell_reverse_corr_{window}"] = np.asarray(reverse_corrs, dtype=float)
        work[f"gr_typewell_corr_gap_{window}"] = work[f"gr_typewell_forward_corr_{window}"] - work[f"gr_typewell_reverse_corr_{window}"]

    return work


def build_horizontal_frame(frames: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for wellname, bundle in frames.items():
        horizontal = bundle["horizontal"].copy()
        horizontal["WELLNAME"] = wellname
        horizontal = _add_typewell_reference_features(horizontal, bundle.get("typewell", pd.DataFrame()))
        pieces.append(horizontal)
    if not pieces:
        raise ValueError("No horizontal well files were found.")
    return pd.concat(pieces, ignore_index=True)


def select_fast_debug_frame(df: pd.DataFrame, max_wells: Optional[int], row_cap: Optional[int]) -> pd.DataFrame:
    work = df.copy()
    if max_wells is not None:
        chosen = list(dict.fromkeys(work["WELLNAME"].astype(str).tolist()))[: int(max_wells)]
        work = work[work["WELLNAME"].astype(str).isin(chosen)].copy()
    if row_cap is not None:
        capped: List[pd.DataFrame] = []
        for _, group in work.groupby("WELLNAME", sort=False):
            capped.append(group.head(int(row_cap)))
        work = pd.concat(capped, ignore_index=True) if capped else work.iloc[0:0].copy()
    return work.reset_index(drop=True)


def _make_light_tree_model(random_state: int = 42) -> TreeEnsembleModel:
    model = TreeEnsembleModel(metrics_path=None, random_state=random_state)

    def _light_specs(self: TreeEnsembleModel) -> Dict[str, Any]:
        specs = TreeEnsembleModel._build_backend_specs(self)
        specs["lightgbm"].params.update(
            {
                "n_estimators": 40,
                "learning_rate": 0.05,
                "num_leaves": 21,
            }
        )
        specs["catboost"].params.update(
            {
                "iterations": 50,
                "learning_rate": 0.05,
                "depth": 5,
            }
        )
        specs["xgboost"].params.update(
            {
                "n_estimators": 60,
                "learning_rate": 0.05,
                "max_depth": 3,
            }
        )
        return specs

    # Keep the training path lightweight for the interactive verification run.
    import types

    model._build_backend_specs = types.MethodType(_light_specs, model)  # type: ignore[assignment]
    return model


def _make_baseline_model(active_backends: Sequence[str]) -> BaselineEnsembleModel:
    model = BaselineEnsembleModel(metrics_path=None)
    model.BACKEND_ORDER = tuple(active_backends)  # type: ignore[assignment]
    return model


def fit_family_models(
    train_df: pd.DataFrame,
    active_families: Mapping[str, Any],
    sub_models: Mapping[str, Any],
) -> Dict[str, Any]:
    family_models: Dict[str, Any] = {}
    run_heavy_baseline_trees = bool(
        sub_models.get(
            "run_heavy_baseline_trees",
            DEFAULT_PIPELINE_CONFIG["sub_models"]["run_heavy_baseline_trees"],
        )
    )
    baseline_backends = ("rf", "et", "hist") if run_heavy_baseline_trees else ("hist",)

    family_steps = [
        ("Family A", "tree", "tree_models", lambda: _make_light_tree_model().fit(train_df, train_df["TVT"].to_numpy())),
        (
            "Family B",
            "sequence",
            "sequence_models",
            lambda: DeepSequenceModel(
                metrics_path=None,
                sequence_length=8,
                hidden_size=16,
                epochs=2,
                batch_size=64,
                learning_rate=1e-3,
            ).fit(train_df, train_df["TVT"].to_numpy()),
        ),
        ("Family D", "spatial", "spatial_models", lambda: SpatialNeighborModel(metrics_path=None).fit(train_df, train_df["TVT"].to_numpy())),
        ("Family E", "kernels", "kernel_models", lambda: KernelMachineModel(metrics_path=None).fit(train_df, train_df["TVT"].to_numpy())),
        (
            "Family F",
            "tabular",
            "tabular_models",
            lambda: DeepTabularModel(
                metrics_path=None,
                hidden_dims=(64, 32),
                epochs=2,
                batch_size=2048,
                learning_rate=1e-3,
            ).fit(train_df, train_df["TVT"].to_numpy()),
        ),
        ("Family C", "linear", "linear_models", lambda: LinearEnsembleModel(metrics_path=None).fit(train_df, train_df["TVT"].to_numpy())),
        ("Family G", "baseline", "baseline_models", lambda: _make_baseline_model(baseline_backends).fit(train_df, train_df["TVT"].to_numpy())),
    ]

    selected_steps = [
        (family_label, family_key, trainer)
        for family_label, family_key, config_flag, trainer in family_steps
        if bool(active_families.get(config_flag, DEFAULT_PIPELINE_CONFIG["active_families"][config_flag]))
    ]

    if not selected_steps:
        raise RuntimeError("No model families are active in pipeline_config.json.")

    with tqdm(total=len(selected_steps), desc=f"Training {len(selected_steps)} model families", dynamic_ncols=True) as family_bar:
        for family_label, family_key, trainer in selected_steps:
            family_bar.set_description(f"Training {family_label}")
            log_family_training_start(family_label, family_key)
            family_models[family_key] = trainer()
            log_family_training_complete(family_label)
            family_bar.update(1)

    return family_models


def collect_peer_oof_matrix(family_models: Dict[str, Any], peer_specs: Sequence[PeerSpec]) -> Tuple[np.ndarray, List[str]]:
    columns: List[np.ndarray] = []
    names: List[str] = []

    if not peer_specs:
        raise RuntimeError("No active peer specs are available for blending.")

    with tqdm(total=len(peer_specs), desc=f"Collecting {len(peer_specs)} peer OOF columns", dynamic_ncols=True) as peer_bar:
        for spec in peer_specs:
            peer_bar.set_description(f"Collecting {spec.display_name}")
            if spec.family == "tree":
                model = family_models["tree"]
                columns.append(np.asarray(model.scaled_oof_predictions_[spec.backend], dtype=float))
            elif spec.family == "sequence":
                model = family_models["sequence"]
                columns.append(np.asarray(model.oof_predictions_sequence_scaled, dtype=float))
            elif spec.family == "spatial":
                model = family_models["spatial"]
                columns.append(np.asarray(model.scaled_oof_predictions_[spec.backend], dtype=float))
            elif spec.family == "kernels":
                model = family_models["kernels"]
                columns.append(np.asarray(model.scaled_oof_predictions_[spec.backend], dtype=float))
            elif spec.family == "tabular":
                model = family_models["tabular"]
                columns.append(np.asarray(model.scaled_oof_predictions_tabular_mlp, dtype=float))
            elif spec.family == "linear":
                model = family_models["linear"]
                columns.append(np.asarray(model.scaled_oof_predictions_[spec.backend], dtype=float))
            elif spec.family == "baseline":
                model = family_models["baseline"]
                columns.append(np.asarray(model.scaled_oof_predictions_[spec.backend], dtype=float))
            else:
                raise KeyError(f"Unknown family '{spec.family}'.")
            names.append(spec.display_name)
            peer_bar.update(1)

    matrix = np.column_stack(columns)
    return matrix, names


def save_oof_cache(cache_dir: str | Path, oof_matrix: np.ndarray, target_scaled: np.ndarray, peer_names: Sequence[str]) -> None:
    cache_path = _resolve_path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path / "oof_cache.npz",
        oof_matrix=np.asarray(oof_matrix, dtype=float),
        target_scaled=np.asarray(target_scaled, dtype=float),
        peer_names=np.asarray(list(peer_names), dtype=object),
    )
    (cache_path / "oof_cache.json").write_text(
        json.dumps({"peer_names": list(peer_names)}, indent=2, sort_keys=True)
    )


def load_oof_cache(cache_dir: str | Path) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    cache_file = _resolve_path(cache_dir) / "oof_cache.npz"
    if not cache_file.exists():
        return None
    data = np.load(cache_file, allow_pickle=True)
    peer_names = [str(name) for name in data["peer_names"].tolist()]
    return np.asarray(data["oof_matrix"], dtype=float), np.asarray(data["target_scaled"], dtype=float), peer_names


def predict_family_scaled(model: Any, family_name: str, df: pd.DataFrame) -> np.ndarray:
    if family_name == "tree":
        columns: List[np.ndarray] = []
        feature_frame = model._build_feature_frame(df)
        for backend in model.BACKEND_ORDER:
            pred_scaled = np.asarray(model.full_models_[backend].predict(feature_frame), dtype=float).reshape(-1)
            columns.append(pred_scaled)
        return np.column_stack(columns)
    if family_name == "sequence":
        pipeline = model.full_model_["pipeline"]
        feature_columns = model.full_model_["feature_columns"]
        feature_frame = model._build_numeric_features(pipeline, df, reference_columns=feature_columns)
        seq, _, order = model._build_causal_sequences(
            train_df=df,
            feature_frame=feature_frame,
            target=None,
            pipeline=pipeline,
            row_ids=np.arange(len(df)),
        )
        pred_scaled = model.full_model_["backend"].predict(seq)
        ordered = np.empty(len(df), dtype=float)
        ordered[order] = pred_scaled
        return ordered[:, None]
    if family_name in {"spatial", "kernels", "linear", "baseline"}:
        columns: List[np.ndarray] = []
        for backend in model.BACKEND_ORDER:
            bundle = model.full_models_[backend]
            feature_frame = model._build_feature_frame(
                df,
                pipeline=bundle["pipeline"],
                reference_columns=bundle["feature_columns"],
            )
            pred_scaled = np.asarray(bundle["model"].predict(feature_frame), dtype=float).reshape(-1)
            columns.append(pred_scaled)
        return np.column_stack(columns)
    if family_name == "tabular":
        bundle = model.full_model_
        feature_frame = model._build_numeric_features(
            bundle["pipeline"],
            df,
            reference_columns=bundle["feature_columns"],
        )
        scaled = bundle["feature_scaler"].transform(feature_frame)
        pred_scaled = bundle["backend"].predict(scaled)
        return pred_scaled[:, None]
    raise KeyError(f"Unknown family '{family_name}'.")


def collect_test_peer_matrix(family_models: Dict[str, Any], test_df: pd.DataFrame, peer_specs: Sequence[PeerSpec]) -> np.ndarray:
    columns: List[np.ndarray] = []

    if not peer_specs:
        raise RuntimeError("No active peer specs are available for test prediction.")

    with tqdm(total=len(peer_specs), desc=f"Collecting {len(peer_specs)} peer test columns", dynamic_ncols=True) as peer_bar:
        for spec in peer_specs:
            peer_bar.set_description(f"Collecting {spec.display_name}")
            if spec.family == "tree":
                model = family_models["tree"]
                pred = predict_family_scaled(model, "tree", test_df)[:, [list(model.BACKEND_ORDER).index(spec.backend)]]
            elif spec.family == "sequence":
                model = family_models["sequence"]
                pred = predict_family_scaled(model, "sequence", test_df)
            elif spec.family == "spatial":
                model = family_models["spatial"]
                pred = predict_family_scaled(model, "spatial", test_df)[:, [list(model.BACKEND_ORDER).index(spec.backend)]]
            elif spec.family == "kernels":
                model = family_models["kernels"]
                pred = predict_family_scaled(model, "kernels", test_df)[:, [list(model.BACKEND_ORDER).index(spec.backend)]]
            elif spec.family == "tabular":
                model = family_models["tabular"]
                pred = predict_family_scaled(model, "tabular", test_df)
            elif spec.family == "linear":
                model = family_models["linear"]
                pred = predict_family_scaled(model, "linear", test_df)[:, [list(model.BACKEND_ORDER).index(spec.backend)]]
            elif spec.family == "baseline":
                model = family_models["baseline"]
                pred = predict_family_scaled(model, "baseline", test_df)[:, [list(model.BACKEND_ORDER).index(spec.backend)]]
            else:
                raise KeyError(f"Unknown family '{spec.family}'.")

            columns.append(np.asarray(pred, dtype=float).reshape(-1))
            peer_bar.update(1)

    return np.column_stack(columns)


def build_submission_from_predictions(
    test_frames: Dict[str, Dict[str, pd.DataFrame]],
    predictions_by_id: Dict[str, float],
    sample_submission_path: str | Path = ROOT / "data" / "sample_submission.csv",
) -> pd.DataFrame:
    sample = pd.read_csv(_resolve_path(sample_submission_path))
    missing = [identifier for identifier in sample["id"].tolist() if identifier not in predictions_by_id]
    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} submission ids.")
    submission = sample.copy()
    submission["tvt"] = submission["id"].map(predictions_by_id).astype(float)
    if submission["tvt"].isna().any():
        raise RuntimeError("Submission contains NaNs after mapping predictions.")
    if list(submission.columns) != ["id", "tvt"]:
        raise RuntimeError("Submission format is invalid.")
    return submission


def append_metrics(record: Dict[str, Any], metrics_path: str | Path = ROOT / "results/metrics.json") -> None:
    path = _resolve_path(metrics_path)
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, list):
                existing = loaded
            elif isinstance(loaded, dict):
                existing = [loaded]
        except json.JSONDecodeError:
            existing = []
    existing.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def run_blending(
    train_root: str | Path = ROOT / "data" / "train",
    test_root: str | Path = ROOT / "data" / "test",
    cache_dir: str | Path = ARTIFACT_DIR,
    submission_path: str | Path = SUBMISSION_PATH,
    sample_submission_path: str | Path = ROOT / "data" / "sample_submission.csv",
    max_wells: Optional[int] = None,
    row_cap: Optional[int] = None,
    force_recompute_cache: bool = False,
    active_families: Optional[Mapping[str, Any]] = None,
    sub_models: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    active_families = dict(DEFAULT_PIPELINE_CONFIG["active_families"] if active_families is None else active_families)
    sub_models = dict(DEFAULT_PIPELINE_CONFIG["sub_models"] if sub_models is None else sub_models)
    peer_specs = build_active_peer_specs(active_families, sub_models)
    expected_peer_names = [spec.display_name for spec in peer_specs]

    train_root = _resolve_path(train_root)
    test_root = _resolve_path(test_root)
    cache_dir = _resolve_path(cache_dir)
    submission_path = _resolve_path(submission_path)
    sample_submission_path = _resolve_path(sample_submission_path)

    train_frames = load_competition_frames(train_root)
    test_frames = load_competition_frames(test_root)

    train_df = build_horizontal_frame(train_frames)
    train_df = select_fast_debug_frame(train_df, max_wells=max_wells, row_cap=row_cap)

    target_pipeline = FeaturePipeline(scale_target=True)
    target_pipeline.fit(train_df, y=train_df["TVT"].to_numpy())
    target_scaled = target_pipeline.transform_target(train_df["TVT"].to_numpy())

    cached = None if force_recompute_cache else load_oof_cache(cache_dir)
    if cached is None:
        family_models = fit_family_models(train_df, active_families=active_families, sub_models=sub_models)
        oof_matrix, peer_names = collect_peer_oof_matrix(family_models, peer_specs)
        save_oof_cache(cache_dir, oof_matrix, target_scaled, peer_names)
    else:
        oof_matrix, cached_target_scaled, peer_names = cached
        if len(cached_target_scaled) == len(target_scaled):
            target_scaled = cached_target_scaled
        family_models = fit_family_models(train_df, active_families=active_families, sub_models=sub_models)
        cache_matches = (
            oof_matrix.shape[0] == len(train_df)
            and oof_matrix.shape[1] == len(expected_peer_names)
            and peer_names == expected_peer_names
        )
        if not cache_matches:
            oof_matrix, peer_names = collect_peer_oof_matrix(family_models, peer_specs)
            save_oof_cache(cache_dir, oof_matrix, target_scaled, peer_names)

    blender = MetaBlender()
    blender.fit(oof_matrix, target_scaled, peer_names=peer_names)

    test_df = build_horizontal_frame(test_frames)
    test_oof_matrix = collect_test_peer_matrix(family_models, test_df, peer_specs)
    blended_scaled = blender.predict(test_oof_matrix)
    blended = target_pipeline.inverse_transform_target(blended_scaled)

    predictions_by_id: Dict[str, float] = {}
    for wellname, bundle in test_frames.items():
        horizontal = bundle["horizontal"].copy().reset_index(drop=True)
        mask = horizontal["TVT_input"].isna().to_numpy()
        if not mask.any():
            continue
        well_ids = [f"{wellname}_{idx}" for idx in horizontal.index[mask]]
        well_preds = blended[horizontal.index[mask]]
        predictions_by_id.update({identifier: float(pred) for identifier, pred in zip(well_ids, well_preds)})

    submission = build_submission_from_predictions(test_frames, predictions_by_id, sample_submission_path=sample_submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    metrics_record = {
        "model_family": "MetaBlender",
        "peer_names": list(peer_names),
        "peer_weights": blender.to_dict(),
        "optimized_ensemble_rmse_scaled": blender.ensemble_oof_rmse_,
        "n_peers": len(peer_names),
        "train_rows": int(len(train_df)),
        "train_wells": int(train_df["WELLNAME"].nunique()),
        "submission_path": str(Path(submission_path).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
    }
    append_metrics(metrics_record, metrics_path=ROOT / "results/metrics.json")

    return {
        "family_models": family_models,
        "target_pipeline": target_pipeline,
        "blender": blender,
        "submission": submission,
        "metrics": metrics_record,
    }


def parse_args(argv: Optional[Sequence[str]] = None, config: Optional[Mapping[str, Any]] = None) -> argparse.Namespace:
    pipeline_config = load_pipeline_config() if config is None else config
    data_config = pipeline_config["data"]
    parser = argparse.ArgumentParser(description="Optimize stacked ensemble weights and write submission.csv.")
    parser.add_argument("--train-root", default=str(data_config["train_root"]))
    parser.add_argument("--test-root", default=str(data_config["test_root"]))
    parser.add_argument("--cache-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--submission-path", default=str(data_config["submission_path"]))
    parser.add_argument("--sample-submission-path", default=str(ROOT / "data" / "sample_submission.csv"))
    parser.add_argument("--max-wells", type=int, default=data_config["max_wells"] if data_config["max_wells"] is None else int(data_config["max_wells"]))
    parser.add_argument("--row-cap", type=int, default=data_config["row_cap"] if data_config["row_cap"] is None else int(data_config["row_cap"]))
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument("--full", action="store_true", help="Disable the interactive fast-debug limits.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    pipeline_config = load_pipeline_config()
    args = parse_args(argv, config=pipeline_config)
    result = run_blending(
        train_root=args.train_root,
        test_root=args.test_root,
        cache_dir=args.cache_dir,
        submission_path=args.submission_path,
        sample_submission_path=args.sample_submission_path,
        max_wells=None if args.full else args.max_wells,
        row_cap=None if args.full else args.row_cap,
        force_recompute_cache=args.force_recompute_cache,
        active_families=pipeline_config["active_families"],
        sub_models=pipeline_config["sub_models"],
    )

    summary = {
        "submission_rows": int(len(result["submission"])),
        "ensemble_rmse_scaled": result["blender"].ensemble_oof_rmse_,
        "weights": result["blender"].to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
