from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline import AbstractBaseModel, ExperimentOrchestrator, FeaturePipeline


class _TestModel(AbstractBaseModel):
    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self.mean_ = float(np.nanmean(y_arr)) if y_arr.size else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def _mock_frame(n_wells: int = 5, rows_per_well: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    frames = []
    for well_idx in range(n_wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 4000.0 - np.cumsum(np.abs(rng.normal(0.4, 0.02, size=rows_per_well)))
        frame = pd.DataFrame(
            {
                "WELLNAME": f"WELL_{well_idx}",
                "MD": md,
                "X": 100.0 + well_idx + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Y": 200.0 + well_idx + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Z": z,
                "ANCC": z + 20.0,
                "ASTNU": z + 15.0,
                "ASTNL": z + 10.0,
                "EGFDU": z + 8.0,
                "EGFDL": z + 4.0,
                "BUDA": z + 2.0,
                "GR": 70.0 + rng.normal(0.0, 1.0, size=rows_per_well),
                "TVT": 30.0 + well_idx + md * 0.25,
                "TVT_input": np.where(md < 4, 1.0, np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class PipelineSmokeTests(unittest.TestCase):
    def test_feature_pipeline_preserves_rows(self) -> None:
        df = _mock_frame()
        pipeline = FeaturePipeline()
        engineered = pipeline.fit_transform(df)

        self.assertEqual(len(engineered), len(df))
        self.assertIn("wellbore_inclination_deg", engineered.columns)
        self.assertIn("gr_roll_mean_5", engineered.columns)

    def test_groupkfold_compiles(self) -> None:
        df = _mock_frame()
        orchestrator = ExperimentOrchestrator(models={"test": _TestModel()}, metrics_path="")
        result = orchestrator.cross_validate(df)

        self.assertEqual(len(result["folds"]), 5)
        self.assertEqual(result["oof_predictions"]["test"].shape[0], len(df))
        self.assertFalse(np.isnan(result["oof_predictions"]["test"]).any())


if __name__ == "__main__":
    unittest.main()
