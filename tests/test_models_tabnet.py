from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models_tabnet import DeepTabularModel


def _mock_tabular_frame(n_wells: int = 5, rows_per_well: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    frames = []
    for well_idx in range(n_wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 4300.0 - np.cumsum(np.abs(rng.normal(0.36, 0.02, size=rows_per_well)))
        frame = pd.DataFrame(
            {
                "WELLNAME": f"WELL_{well_idx}",
                "MD": md,
                "X": 1000.0 + well_idx * 33.0 + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Y": 1500.0 + well_idx * 19.0 + np.cumsum(rng.normal(0.8, 0.1, size=rows_per_well)),
                "Z": z,
                "ANCC": z + 19.0,
                "ASTNU": z + 15.0,
                "ASTNL": z + 11.0,
                "EGFDU": z + 8.0,
                "EGFDL": z + 5.0,
                "BUDA": z + 2.0,
                "GR": 70.0 + np.sin(md / 2.0) * 3.5 + rng.normal(0.0, 0.4, size=rows_per_well),
                "TVT": 17.0 + well_idx + md * 0.28 + rng.normal(0.0, 0.08, size=rows_per_well),
                "TVT_input": np.where(md < 4, 0.8, np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TabularModelTests(unittest.TestCase):
    def test_fit_produces_tabular_oof(self) -> None:
        df = _mock_tabular_frame()
        model = DeepTabularModel(metrics_path=None, epochs=5)
        model.fit(df, df["TVT"].to_numpy())

        self.assertIsNotNone(model.oof_predictions_tabular_mlp)
        self.assertEqual(len(model.oof_predictions_tabular_mlp), len(df))
        self.assertEqual(len(model.fold_scores_), 5)
        self.assertFalse(np.isnan(model.oof_predictions_tabular_mlp).any())
        self.assertFalse(np.isnan(model.scaled_oof_predictions_tabular_mlp).any())
        self.assertEqual(len(model.predict_oof()), len(df))
        self.assertEqual(model.full_model_["backend_state"]["backend"], "tabular_mlp")

    def test_predict_matches_length(self) -> None:
        df = _mock_tabular_frame()
        model = DeepTabularModel(metrics_path=None, epochs=5)
        model.fit(df, df["TVT"].to_numpy())
        preds = model.predict(df)

        self.assertEqual(len(preds), len(df))
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main()
