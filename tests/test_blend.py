from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.blend import MetaBlender, load_oof_cache, save_oof_cache


class BlendTests(unittest.TestCase):
    def test_meta_blender_simplex_constraints(self) -> None:
        rng = np.random.default_rng(99)
        target = rng.normal(size=200)
        oof = np.column_stack(
            [
                target + rng.normal(scale=0.2, size=200),
                target + rng.normal(scale=0.3, size=200),
                target + rng.normal(scale=0.4, size=200),
            ]
        )

        blender = MetaBlender(peer_names=["a", "b", "c"])
        blender.fit(oof, target)

        self.assertIsNotNone(blender.weights_)
        self.assertAlmostEqual(float(blender.weights_.sum()), 1.0, places=6)
        self.assertTrue(np.all(blender.weights_ >= -1e-12))
        self.assertTrue(np.all(blender.weights_ <= 1.0 + 1e-12))
        self.assertIsNotNone(blender.ensemble_oof_rmse_)
        self.assertLess(blender.ensemble_oof_rmse_, 0.6)

    def test_oof_cache_roundtrip(self) -> None:
        oof = np.arange(12, dtype=float).reshape(4, 3)
        target = np.arange(4, dtype=float)
        peer_names = ["p1", "p2", "p3"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_oof_cache(tmpdir, oof, target, peer_names)
            loaded = load_oof_cache(tmpdir)

            self.assertIsNotNone(loaded)
            loaded_oof, loaded_target, loaded_names = loaded
            self.assertEqual(loaded_oof.shape, oof.shape)
            self.assertTrue(np.array_equal(loaded_oof, oof))
            self.assertTrue(np.array_equal(loaded_target, target))
            self.assertEqual(loaded_names, peer_names)


if __name__ == "__main__":
    unittest.main()
