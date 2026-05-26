from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models_trees import TreeEnsembleModel


def _mock_tree_frame(n_wells: int = 5, rows_per_well: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(321)
    frames = []
    for well_idx in range(n_wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 3000.0 - np.cumsum(np.abs(rng.normal(0.35, 0.02, size=rows_per_well)))
        frame = pd.DataFrame(
            {
                "WELLNAME": f"WELL_{well_idx}",
                "MD": md,
                "X": 1000.0 + well_idx * 40.0 + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Y": 1200.0 + well_idx * 20.0 + np.cumsum(rng.normal(0.8, 0.1, size=rows_per_well)),
                "Z": z,
                "ANCC": z + 22.0,
                "ASTNU": z + 18.0,
                "ASTNL": z + 14.0,
                "EGFDU": z + 10.0,
                "EGFDL": z + 6.0,
                "BUDA": z + 3.0,
                "GR": 72.0 + np.sin(md / 2.0) * 3.0 + rng.normal(0.0, 0.4, size=rows_per_well),
                "TVT": 15.0 + well_idx + md * 0.35 + rng.normal(0.0, 0.1, size=rows_per_well),
                "TVT_input": np.where(md < 5, 0.5, np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TreeModelTests(unittest.TestCase):
    def test_fit_produces_oof_predictions(self) -> None:
        df = _mock_tree_frame()
        model = TreeEnsembleModel(metrics_path=None)
        model.fit(df, df["TVT"].to_numpy())

        self.assertEqual(set(model.oof_predictions_.keys()), set(TreeEnsembleModel.BACKEND_ORDER))
        self.assertEqual(set(model.scaled_oof_predictions_.keys()), set(TreeEnsembleModel.BACKEND_ORDER))

        for backend in TreeEnsembleModel.BACKEND_ORDER:
            self.assertEqual(len(model.oof_predictions_[backend]), len(df))
            self.assertEqual(len(model.scaled_oof_predictions_[backend]), len(df))
            self.assertFalse(np.isnan(model.oof_predictions_[backend]).any())
            self.assertFalse(np.isnan(model.scaled_oof_predictions_[backend]).any())
            self.assertEqual(len(model.fold_scores_[backend]), 5)

        self.assertGreaterEqual(len(model.hyperparameter_log_), 15)

    def test_predict_matches_input_length(self) -> None:
        df = _mock_tree_frame()
        model = TreeEnsembleModel(metrics_path=None)
        model.fit(df, df["TVT"].to_numpy())
        preds = model.predict(df)
        all_preds = model.predict_all(df)

        self.assertEqual(len(preds), len(df))
        self.assertEqual(set(all_preds.keys()), set(TreeEnsembleModel.BACKEND_ORDER))
        for backend_pred in all_preds.values():
            self.assertEqual(len(backend_pred), len(df))
            self.assertFalse(np.isnan(backend_pred).any())
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main()
