from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models_kernels import KernelMachineModel


def _mock_kernel_frame(n_wells: int = 5, rows_per_well: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(818)
    frames = []
    for well_idx in range(n_wells):
        md = np.arange(rows_per_well, dtype=float)
        z = 3600.0 - np.cumsum(np.abs(rng.normal(0.34, 0.02, size=rows_per_well)))
        frame = pd.DataFrame(
            {
                "WELLNAME": f"WELL_{well_idx}",
                "MD": md,
                "X": 950.0 + well_idx * 36.0 + np.cumsum(rng.normal(1.0, 0.1, size=rows_per_well)),
                "Y": 1180.0 + well_idx * 17.0 + np.cumsum(rng.normal(0.8, 0.1, size=rows_per_well)),
                "Z": z,
                "ANCC": z + 19.0,
                "ASTNU": z + 15.0,
                "ASTNL": z + 11.0,
                "EGFDU": z + 8.0,
                "EGFDL": z + 5.0,
                "BUDA": z + 2.0,
                "GR": 67.0 + np.sin(md / 2.0) * 3.2 + rng.normal(0.0, 0.4, size=rows_per_well),
                "TVT": 15.0 + well_idx + md * 0.31 + rng.normal(0.0, 0.08, size=rows_per_well),
                "TVT_input": np.where(md < 4, 0.7, np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class KernelModelTests(unittest.TestCase):
    def test_fit_produces_svr_oof_vectors(self) -> None:
        df = _mock_kernel_frame()
        model = KernelMachineModel(metrics_path=None)
        model.fit(df, df["TVT"].to_numpy())

        self.assertEqual(set(model.oof_predictions_.keys()), set(KernelMachineModel.BACKEND_ORDER))
        self.assertEqual(set(model.scaled_oof_predictions_.keys()), set(KernelMachineModel.BACKEND_ORDER))
        self.assertEqual(len(model.oof_predictions_svr_rbf), len(df))
        self.assertEqual(len(model.oof_predictions_svr_linear), len(df))

        for backend in KernelMachineModel.BACKEND_ORDER:
            self.assertEqual(len(model.oof_predictions_[backend]), len(df))
            self.assertEqual(len(model.scaled_oof_predictions_[backend]), len(df))
            self.assertFalse(np.isnan(model.oof_predictions_[backend]).any())
            self.assertFalse(np.isnan(model.scaled_oof_predictions_[backend]).any())
            self.assertEqual(len(model.fold_scores_[backend]), 5)

    def test_predict_matches_length(self) -> None:
        df = _mock_kernel_frame()
        model = KernelMachineModel(metrics_path=None)
        model.fit(df, df["TVT"].to_numpy())
        preds = model.predict(df)
        all_preds = model.predict_all(df)

        self.assertEqual(len(preds), len(df))
        self.assertEqual(set(all_preds.keys()), set(KernelMachineModel.BACKEND_ORDER))
        for backend_pred in all_preds.values():
            self.assertEqual(len(backend_pred), len(df))
            self.assertFalse(np.isnan(backend_pred).any())
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main()
