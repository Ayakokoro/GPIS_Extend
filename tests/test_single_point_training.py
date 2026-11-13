"""Contract checks for native scene-derived single-point residual training."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import json
from pathlib import Path
import unittest
from unittest import mock
import uuid

import numpy as np
from numpy.testing import assert_allclose
import torch

from tools.single_point_training.model import (
    FEATURE_NAMES, SCHEMA, SinglePointResidualNet, bounded_log_correction,
)
from tools.single_point_training.train import load_dataset, source_file_metadata, training_normalization


@contextmanager
def dataset_directory():
    # Python 3.10's Windows 0700 temporary-directory ACL excludes the sandbox
    # principal. Inherit the workspace ACL instead, and remove only our files.
    parent = Path(__file__).resolve().parents[1] / "outputs" / "single_point_training_tests"
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / uuid.uuid4().hex
    root.mkdir()
    try:
        yield root
    finally:
        for name in ("manifest.json", "samples.csv", "samples.partial.csv"):
            (root / name).unlink(missing_ok=True)
        root.rmdir()


def make_dataset(root: Path, mutate=None):
    manifest = {"schema": SCHEMA, "status": "complete", "rows": 6, "split_counts": [2, 2, 2],
                "sigma": .02, "length_scale": .2, "feature_names": list(FEATURE_NAMES),
                "source": {"scene": "synthetic-test.json"},
                "views": [{"view_id": split, "split": split} for split in range(3)]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for split in range(3):
        for query in range(2):
            z_u = .2 + .1*query
            row = {"view_id": split, "ray_id": split, "split": split,
                   "z_x": float(query + 10*split), "z_u": z_u, "z_v": -.5 + .1*query,
                   "rho_xu": 0., "rho_xv": 0., "rho_uv": 0.,
                   "log_alpha_base": -1., "log_alpha_history": -1.,
                   "log_p_history": float(torch.special.log_ndtr(torch.tensor(z_u, dtype=torch.float64))), "log_correction": 0.}
            rows.append(row)
    if mutate:
        mutate(rows)
    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BoundedNetworkTests(unittest.TestCase):
    def test_asinh_transform_preserves_large_inputs_without_clipping(self):
        model = SinglePointResidualNet(feature_transform="asinh_z")
        with torch.no_grad():
            for layer in model.network:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
                    layer.weight[0, 0] = 1
        features = torch.zeros(3, 6)
        features[:, 0] = torch.tensor([2., 20., 60000.])
        assert_allclose(model.raw_residual(features).detach().numpy(),
                        np.arcsinh([2., 20., 60000.]), rtol=2e-7)

    def test_initial_network_is_identity_including_probability_tails(self):
        torch.manual_seed(1)
        model = SinglePointResidualNet()
        features = torch.randn(6, 6)
        logp = torch.tensor([-1000., -20., -1., -1e-8, -1e-100, 0.], dtype=torch.float64)
        actual = model(features, logp)
        assert_allclose(actual.detach().numpy(), 0, atol=2e-16, rtol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1313)

    def test_transform_obeys_bound_for_both_signs_and_extreme_residuals(self):
        residual = torch.tensor([-1000., -20., -1., 0., 1., 20., 1000.], dtype=torch.float64)
        for probability in (1e-30, .1, .5, 1-1e-12, 1.):
            logp = torch.full_like(residual, np.log(probability))
            estimate = bounded_log_correction(residual, logp)
            self.assertTrue(torch.isfinite(estimate).all())
            self.assertTrue((estimate <= -logp + 1e-14).all())
            self.assertTrue((estimate[1:] >= estimate[:-1]).all())
            if 0 < probability < 1:
                self.assertLess(float(estimate[2]), 0)
                self.assertGreater(float(estimate[4]), 0)

    def test_probability_one_has_finite_zero_network_gradient(self):
        residual = torch.tensor([3.], requires_grad=True, dtype=torch.float64)
        estimate = bounded_log_correction(residual, torch.zeros(1, dtype=torch.float64))
        estimate.sum().backward()
        assert_allclose(estimate.detach().numpy(), [0.])
        assert_allclose(residual.grad.numpy(), [0.])

    def test_input_values_are_not_clipped_to_a_training_range(self):
        model = SinglePointResidualNet()
        with torch.no_grad():
            for layer in model.network:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
                    layer.weight[0, 0] = 1
        features = torch.zeros(2, 6)
        features[:, 0] = torch.tensor([2., 20.])
        assert_allclose(model.raw_residual(features).detach().numpy(), [2., 20.])


class DatasetContractTests(unittest.TestCase):
    def test_asinh_normalization_uses_transformed_training_features(self):
        with dataset_directory() as root:
            make_dataset(root)
            data = load_dataset(root)
            mean, scale = training_normalization(data, "asinh_z")
            self.assertAlmostEqual(mean[0], np.arcsinh(1.) / 2)
            self.assertAlmostEqual(scale[0], np.arcsinh(1.) / 2)
            assert_allclose(mean[3:], 0)
            assert_allclose(scale[3:], 1)

    def test_normalization_uses_training_views_only_and_preserves_correlations(self):
        with dataset_directory() as root:
            make_dataset(root)
            data = load_dataset(root)
            mean, scale = training_normalization(data)
            self.assertAlmostEqual(mean[0], .5)
            self.assertAlmostEqual(scale[0], .5)
            assert_allclose(mean[3:], 0)
            assert_allclose(scale[3:], 1)
            self.assertGreater(data["features"][4, 0], mean[0] + 3*scale[0])

    def check_rejected(self, mutate, pattern):
        with dataset_directory() as root:
            make_dataset(root, mutate)
            with self.assertRaisesRegex(ValueError, pattern):
                load_dataset(root)

    def test_view_split_leakage_is_rejected(self):
        self.check_rejected(lambda rows: rows[2].update(view_id=0), "view_id leakage")

    def test_ray_split_leakage_is_rejected(self):
        self.check_rejected(lambda rows: rows[2].update(ray_id=0), "ray_id leakage")

    def test_inconsistent_teacher_labels_are_rejected(self):
        self.check_rejected(lambda rows: rows[0].update(log_correction=.1), "two hazard labels")

    def test_teacher_exceeding_envelope_is_rejected(self):
        self.check_rejected(lambda rows: rows[0].update(log_correction=2., log_alpha_history=1.), "envelope")

    def test_history_probability_uses_the_history_feature(self):
        self.check_rejected(lambda rows: rows[0].update(log_p_history=-2.), "Gaussian history feature")

    def test_negative_history_correlation_is_rejected(self):
        self.check_rejected(lambda rows: rows[0].update(rho_xu=-.1), "negative current/history")

    def test_incomplete_manifest_is_rejected_even_with_final_csv(self):
        with dataset_directory() as root:
            make_dataset(root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["status"] = "generating"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "status=complete"):
                load_dataset(root)

    def test_truncated_csv_is_rejected_by_manifest_row_count(self):
        with dataset_directory() as root:
            make_dataset(root)
            lines = (root / "samples.csv").read_text().splitlines()
            (root / "samples.csv").write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "row count"):
                load_dataset(root)

    def test_stale_partial_export_is_rejected(self):
        with dataset_directory() as root:
            make_dataset(root)
            (root / "samples.partial.csv").write_text("partial")
            with self.assertRaisesRegex(ValueError, "partial.csv"):
                load_dataset(root)

    def test_manifest_view_assignments_are_checked(self):
        with dataset_directory() as root:
            make_dataset(root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["views"][1]["split"] = 2
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "view assignments"):
                load_dataset(root)

    def test_same_source_and_asset_is_hashed_once(self):
        with dataset_directory() as root:
            make_dataset(root)
            path = str((root / "samples.csv").resolve())
            with mock.patch("tools.single_point_training.train.sha256", return_value="a"*64) as digest:
                info = source_file_metadata({"source": {"scene": path, "sdf_asset": path}})
            self.assertEqual(digest.call_count, 1)
            self.assertEqual(info["files"]["scene"]["sha256"], "a"*64)
            self.assertEqual(info["files"]["scene"], info["files"]["sdf_asset"])
            self.assertFalse(info["generation_time_identity_certified"])


if __name__ == "__main__":
    unittest.main()
