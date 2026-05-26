from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models_sequences import DeepSequenceModel


def _mock_sequence_frame(n_wells: int = 5, rows_per_well: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(777)
    frames = []
    for well_idx in range(n_wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 4500.0 - np.cumsum(np.abs(rng.normal(0.42, 0.02, size=rows_per_well)))
        frame = pd.DataFrame(
            {
                "WELLNAME": f"WELL_{well_idx}",
                "MD": md,
                "X": 800.0 + well_idx * 30.0 + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Y": 1200.0 + well_idx * 15.0 + np.cumsum(rng.normal(0.8, 0.1, size=rows_per_well)),
                "Z": z,
                "ANCC": z + 20.0,
                "ASTNU": z + 15.0,
                "ASTNL": z + 10.0,
                "EGFDU": z + 8.0,
                "EGFDL": z + 4.0,
                "BUDA": z + 2.0,
                "GR": 68.0 + np.sin(md / 2.0) * 4.0 + rng.normal(0.0, 0.5, size=rows_per_well),
                "TVT": 18.0 + well_idx + md * 0.25 + rng.normal(0.0, 0.1, size=rows_per_well),
                "TVT_input": np.where(md < 4, 1.0, np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class SequenceModelTests(unittest.TestCase):
    def test_fit_produces_sequence_oof(self) -> None:
        df = _mock_sequence_frame()
        model = DeepSequenceModel(metrics_path=None, sequence_length=6)
        model.fit(df, df["TVT"].to_numpy())

        self.assertIsNotNone(model.oof_predictions_sequence)
        self.assertEqual(len(model.oof_predictions_sequence), len(df))
        self.assertEqual(len(model.fold_scores_), 5)
        self.assertFalse(np.isnan(model.oof_predictions_sequence).any())
        self.assertEqual(len(model.predict_oof()), len(df))
        self.assertEqual(model.full_model_["backend_state"]["backend"], "torch_bilstm")

    def test_predict_matches_length(self) -> None:
        df = _mock_sequence_frame()
        model = DeepSequenceModel(metrics_path=None, sequence_length=6)
        model.fit(df, df["TVT"].to_numpy())
        preds = model.predict(df)

        self.assertEqual(len(preds), len(df))
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main()
