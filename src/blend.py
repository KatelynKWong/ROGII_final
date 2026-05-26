from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

if __package__ in {None, ""}:  # pragma: no cover - direct execution shim
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.pipeline import FeaturePipeline
from src.models_baselines import BaselineEnsembleModel
from src.models_kernels import KernelMachineModel
from src.models_linear import LinearEnsembleModel
from src.models_sequences import DeepSequenceModel
from src.models_spatial import SpatialNeighborModel
from src.models_tabnet import DeepTabularModel
from src.models_trees import TreeEnsembleModel


ARTIFACT_DIR = Path("artifacts/blend")
OOF_CACHE_PATH = ARTIFACT_DIR / "oof_cache.npz"
OOF_META_PATH = ARTIFACT_DIR / "oof_cache.json"
WEIGHTS_PATH = ARTIFACT_DIR / "meta_weights.json"
SUBMISSION_PATH = Path("submission.csv")


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


def build_horizontal_frame(frames: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for wellname, bundle in frames.items():
        horizontal = bundle["horizontal"].copy()
        horizontal["WELLNAME"] = wellname
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


def fit_family_models(train_df: pd.DataFrame) -> Dict[str, Any]:
    family_models: Dict[str, Any] = {}

    tree_model = _make_light_tree_model()
    tree_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["tree"] = tree_model

    sequence_model = DeepSequenceModel(
        metrics_path=None,
        sequence_length=8,
        hidden_size=16,
        epochs=2,
        batch_size=64,
        learning_rate=1e-3,
    )
    sequence_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["sequence"] = sequence_model

    spatial_model = SpatialNeighborModel(metrics_path=None)
    spatial_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["spatial"] = spatial_model

    kernel_model = KernelMachineModel(metrics_path=None)
    kernel_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["kernels"] = kernel_model

    tabular_model = DeepTabularModel(
        metrics_path=None,
        hidden_dims=(64, 32),
        epochs=2,
        batch_size=64,
        learning_rate=1e-3,
    )
    tabular_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["tabular"] = tabular_model

    linear_model = LinearEnsembleModel(metrics_path=None)
    linear_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["linear"] = linear_model

    baseline_model = BaselineEnsembleModel(metrics_path=None)
    baseline_model.fit(train_df, train_df["TVT"].to_numpy())
    family_models["baseline"] = baseline_model

    return family_models


def collect_peer_oof_matrix(family_models: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    columns: List[np.ndarray] = []
    names: List[str] = []

    tree_model = family_models["tree"]
    for backend in tree_model.BACKEND_ORDER:
        columns.append(np.asarray(tree_model.scaled_oof_predictions_[backend], dtype=float))
        names.append(f"tree_{backend}")

    sequence_model = family_models["sequence"]
    columns.append(np.asarray(sequence_model.oof_predictions_sequence_scaled, dtype=float))
    names.append("sequence_bilstm")

    spatial_model = family_models["spatial"]
    for backend in spatial_model.BACKEND_ORDER:
        columns.append(np.asarray(spatial_model.scaled_oof_predictions_[backend], dtype=float))
        names.append(f"spatial_{backend}")

    kernel_model = family_models["kernels"]
    for backend in kernel_model.BACKEND_ORDER:
        columns.append(np.asarray(kernel_model.scaled_oof_predictions_[backend], dtype=float))
        names.append(f"kernels_{backend}")

    tabular_model = family_models["tabular"]
    columns.append(np.asarray(tabular_model.scaled_oof_predictions_tabular_mlp, dtype=float))
    names.append("tabular_mlp")

    linear_model = family_models["linear"]
    for backend in linear_model.BACKEND_ORDER:
        columns.append(np.asarray(linear_model.scaled_oof_predictions_[backend], dtype=float))
        names.append(f"linear_{backend}")

    baseline_model = family_models["baseline"]
    for backend in baseline_model.BACKEND_ORDER:
        columns.append(np.asarray(baseline_model.scaled_oof_predictions_[backend], dtype=float))
        names.append(f"baseline_{backend}")

    matrix = np.column_stack(columns)
    return matrix, names


def save_oof_cache(cache_dir: str | Path, oof_matrix: np.ndarray, target_scaled: np.ndarray, peer_names: Sequence[str]) -> None:
    cache_path = Path(cache_dir)
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
    cache_file = Path(cache_dir) / "oof_cache.npz"
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


def collect_test_peer_matrix(family_models: Dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    columns: List[np.ndarray] = []

    tree_model = family_models["tree"]
    tree_preds = predict_family_scaled(tree_model, "tree", test_df)
    for idx in range(tree_preds.shape[1]):
        columns.append(tree_preds[:, idx])

    sequence_model = family_models["sequence"]
    columns.append(predict_family_scaled(sequence_model, "sequence", test_df).reshape(-1))

    spatial_model = family_models["spatial"]
    spatial_preds = predict_family_scaled(spatial_model, "spatial", test_df)
    for idx in range(spatial_preds.shape[1]):
        columns.append(spatial_preds[:, idx])

    kernel_model = family_models["kernels"]
    kernel_preds = predict_family_scaled(kernel_model, "kernels", test_df)
    for idx in range(kernel_preds.shape[1]):
        columns.append(kernel_preds[:, idx])

    tabular_model = family_models["tabular"]
    columns.append(predict_family_scaled(tabular_model, "tabular", test_df).reshape(-1))

    linear_model = family_models["linear"]
    linear_preds = predict_family_scaled(linear_model, "linear", test_df)
    for idx in range(linear_preds.shape[1]):
        columns.append(linear_preds[:, idx])

    baseline_model = family_models["baseline"]
    baseline_preds = predict_family_scaled(baseline_model, "baseline", test_df)
    for idx in range(baseline_preds.shape[1]):
        columns.append(baseline_preds[:, idx])

    return np.column_stack(columns)


def build_submission_from_predictions(
    test_frames: Dict[str, Dict[str, pd.DataFrame]],
    predictions_by_id: Dict[str, float],
    sample_submission_path: str | Path = "data/sample_submission.csv",
) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission_path)
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


def append_metrics(record: Dict[str, Any], metrics_path: str | Path = "metrics.json") -> None:
    path = Path(metrics_path)
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
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def run_blending(
    train_root: str | Path = "data/train",
    test_root: str | Path = "data/test",
    cache_dir: str | Path = ARTIFACT_DIR,
    submission_path: str | Path = SUBMISSION_PATH,
    sample_submission_path: str | Path = "data/sample_submission.csv",
    max_wells: Optional[int] = 6,
    row_cap: Optional[int] = 500,
    force_recompute_cache: bool = False,
) -> Dict[str, Any]:
    train_frames = load_competition_frames(train_root)
    test_frames = load_competition_frames(test_root)

    train_df = build_horizontal_frame(train_frames)
    train_df = select_fast_debug_frame(train_df, max_wells=max_wells, row_cap=row_cap)

    target_pipeline = FeaturePipeline(scale_target=True)
    target_pipeline.fit(train_df, y=train_df["TVT"].to_numpy())
    target_scaled = target_pipeline.transform_target(train_df["TVT"].to_numpy())

    cached = None if force_recompute_cache else load_oof_cache(cache_dir)
    if cached is None:
        family_models = fit_family_models(train_df)
        oof_matrix, peer_names = collect_peer_oof_matrix(family_models)
        save_oof_cache(cache_dir, oof_matrix, target_scaled, peer_names)
    else:
        oof_matrix, cached_target_scaled, peer_names = cached
        if len(cached_target_scaled) == len(target_scaled):
            target_scaled = cached_target_scaled
        family_models = fit_family_models(train_df)
        if oof_matrix.shape[0] != len(train_df):
            oof_matrix, peer_names = collect_peer_oof_matrix(family_models)
            save_oof_cache(cache_dir, oof_matrix, target_scaled, peer_names)

    blender = MetaBlender()
    blender.fit(oof_matrix, target_scaled, peer_names=peer_names)

    test_df = build_horizontal_frame(test_frames)
    test_oof_matrix = collect_test_peer_matrix(family_models, test_df)
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
    Path(submission_path).parent.mkdir(parents=True, exist_ok=True)
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
    append_metrics(metrics_record, metrics_path="metrics.json")

    return {
        "family_models": family_models,
        "target_pipeline": target_pipeline,
        "blender": blender,
        "submission": submission,
        "metrics": metrics_record,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize stacked ensemble weights and write submission.csv.")
    parser.add_argument("--train-root", default="data/train")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--cache-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--submission-path", default=str(SUBMISSION_PATH))
    parser.add_argument("--sample-submission-path", default="data/sample_submission.csv")
    parser.add_argument("--max-wells", type=int, default=6)
    parser.add_argument("--row-cap", type=int, default=500)
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument("--full", action="store_true", help="Disable the interactive fast-debug limits.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_blending(
        train_root=args.train_root,
        test_root=args.test_root,
        cache_dir=args.cache_dir,
        submission_path=args.submission_path,
        sample_submission_path=args.sample_submission_path,
        max_wells=None if args.full else args.max_wells,
        row_cap=None if args.full else args.row_cap,
        force_recompute_cache=args.force_recompute_cache,
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
