from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models_linear import LinearEnsembleModel


def _mock_linear_frame(n_wells: int = 5, rows_per_well: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(909)
    frames = []
    for well_idx in range(n_wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 3800.0 - np.cumsum(np.abs(rng.normal(0.32, 0.02, size=rows_per_well)))
        frame = pd.DataFrame(
            {
                "WELLNAME": f"WELL_{well_idx}",
                "MD": md,
                "X": 900.0 + well_idx * 35.0 + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Y": 1100.0 + well_idx * 18.0 + np.cumsum(rng.normal(0.8, 0.1, size=rows_per_well)),
                "Z": z,
                "ANCC": z + 21.0,
                "ASTNU": z + 16.0,
                "ASTNL": z + 11.0,
                "EGFDU": z + 9.0,
                "EGFDL": z + 5.0,
                "BUDA": z + 2.0,
                "GR": 66.0 + np.sin(md / 2.0) * 3.5 + rng.normal(0.0, 0.4, size=rows_per_well),
                "TVT": 16.0 + well_idx + md * 0.3 + rng.normal(0.0, 0.08, size=rows_per_well),
                "TVT_input": np.where(md < 4, 0.8, np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class LinearModelTests(unittest.TestCase):
    def test_fit_produces_three_oof_arrays(self) -> None:
        df = _mock_linear_frame()
        model = LinearEnsembleModel(metrics_path=None)
        model.fit(df, df["TVT"].to_numpy())

        self.assertEqual(len(model.oof_predictions_ridge), len(df))
        self.assertEqual(len(model.oof_predictions_lasso), len(df))
        self.assertEqual(len(model.oof_predictions_elasticnet), len(df))
        self.assertFalse(np.isnan(model.oof_predictions_ridge).any())
        self.assertFalse(np.isnan(model.oof_predictions_lasso).any())
        self.assertFalse(np.isnan(model.oof_predictions_elasticnet).any())
        self.assertEqual(len(model.fold_scores_["ridge"]), 5)
        self.assertEqual(len(model.fold_scores_["lasso"]), 5)
        self.assertEqual(len(model.fold_scores_["elasticnet"]), 5)

    def test_predict_matches_length(self) -> None:
        df = _mock_linear_frame()
        model = LinearEnsembleModel(metrics_path=None)
        model.fit(df, df["TVT"].to_numpy())
        preds = model.predict(df)
        all_preds = model.predict_all(df)

        self.assertEqual(len(preds), len(df))
        self.assertEqual(set(all_preds.keys()), {"ridge", "lasso", "elasticnet"})
        for backend_pred in all_preds.values():
            self.assertEqual(len(backend_pred), len(df))
            self.assertFalse(np.isnan(backend_pred).any())
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main()
