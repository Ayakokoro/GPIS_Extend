"""Independent checks for the offline Gaussian-orthant learning pipeline.

Run with ``python -B -m unittest discover -s tests -p test_gaussian_orthant_nn.py -v``.
Only numpy/scipy/torch and the standard-library test runner are required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gaussian_orthant_nn import gaussian as gg


def _full_gradient_schur_oracle(positions, values, origin_mean, mean_gradient,
                              observed_gradient, direction, sigma, length):
    """Condition a physical 3-D SE GP on [F(0), Gx(0), Gy(0), Gz(0)].

    This deliberately assembles the full observation matrix, independently
    of the implementation's dimensionless one-dimensional simplification.
    """
    points = np.asarray(positions)[:, None] * np.asarray(direction)[None, :]
    displacement = points[:, None, :] - points[None, :, :]
    prior = sigma ** 2 * np.exp(-np.sum(displacement ** 2, axis=2) / (2 * length ** 2))
    origin_cross = sigma ** 2 * np.exp(-np.sum(points ** 2, axis=1) / (2 * length ** 2))
    cross = np.column_stack((origin_cross, origin_cross[:, None] * points / length ** 2))
    observed_cov = np.diag([sigma ** 2] + [sigma ** 2 / length ** 2] * 3)
    innovation = np.r_[-origin_mean, np.asarray(observed_gradient) - np.asarray(mean_gradient)]
    mean = np.asarray(values) + cross @ np.linalg.solve(observed_cov, innovation)
    covariance = prior - cross @ np.linalg.solve(observed_cov, cross.T)
    return mean, covariance


def _bivariate_quad(a, b, rho):
    scale = np.sqrt(1 - rho ** 2)
    value, error = quad(
        lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi) * ndtr((b - rho * x) / scale),
        -np.inf, a, epsabs=1e-15, epsrel=2e-11, limit=250,
    )
    return value, error


class GaussianStatisticsTests(unittest.TestCase):
    def test_prior_is_the_unconditioned_se_kernel(self):
        t = np.array([0.0, 0.23, 0.8, 1.9, 3.7])
        sigma, length = 0.73, 1.27
        actual = gg.se_covariance(t, length_scale=length, sigma=sigma)
        expected = np.array([
            [sigma ** 2 * np.exp(-(x - y) ** 2 / (2 * length ** 2)) for y in t]
            for x in t
        ])
        assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
        assert_allclose(np.diag(actual), sigma ** 2, rtol=2e-14)

    def test_surface_statistics_match_full_3d_schur_conditioning(self):
        t = np.array([0.17, 0.49, 0.94, 1.8, 3.6])
        mean_values = np.array([0.2, -0.7, -0.1, 0.9, 0.3])
        direction = np.array([2.0, -1.0, 3.0]) / np.sqrt(14)
        mean_gradient = np.array([0.4, -0.2, 0.6])
        observed_gradient = np.array([-0.1, 0.3, 1.4])
        origin_mean, sigma, length = 0.37, 0.71, 0.83
        expected = _full_gradient_schur_oracle(
            t, mean_values, origin_mean, mean_gradient, observed_gradient,
            direction, sigma, length,
        )
        actual = gg.surface_conditioned_statistics(
            t, mean_values, origin_mean, float(mean_gradient @ direction),
            float(observed_gradient @ direction), length_scale=length, sigma=sigma,
        )
        assert_allclose(actual[0], expected[0], rtol=2e-12, atol=2e-14)
        assert_allclose(actual[1], expected[1], rtol=2e-10, atol=2e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(actual[1]).min(), -2e-13)

    def test_outward_gradient_has_the_correct_conditioning_sign(self):
        t = np.array([0.1, 0.5, 1.3])
        length = 0.8
        mean, _ = gg.surface_conditioned_statistics(
            t, np.zeros(3), 0.0, -0.3, 0.8, length_scale=length, sigma=0.7,
        )
        assert_allclose(mean, 1.1 * t * np.exp(-0.5 * (t / length) ** 2), rtol=2e-13)
        self.assertTrue(np.all(mean > 0))

    def test_standardization_is_invariant_to_positive_variable_scales(self):
        mean = np.array([-0.8, 0.2, 1.1])
        covariance = np.array([[0.7, 0.13, -0.08], [0.13, 1.7, 0.21], [-0.08, 0.21, 2.4]])
        scales = np.array([0.013, 17.0, 0.51])
        z, corr = gg.standardized_statistics(mean, covariance)
        z2, corr2 = gg.standardized_statistics(mean * scales, covariance * np.outer(scales, scales))
        assert_allclose(z2, z, rtol=3e-14, atol=3e-14)
        assert_allclose(corr2, corr, rtol=3e-14, atol=3e-14)


class OrthantLabelTests(unittest.TestCase):
    def _assert_usable(self, result, tolerance):
        self.assertEqual(result.status, "ok", result)
        self.assertTrue(np.isfinite(result.log_probability))
        self.assertGreaterEqual(result.error, 0)
        self.assertGreaterEqual(result.relative_error, 0)
        self.assertLessEqual(result.relative_error, tolerance * (1 + 1e-10))

    def test_one_dimension_keeps_rare_log_probability(self):
        for z in (-12.0, -2.0, 0.0, 3.0):
            with self.subTest(z=z):
                result = gg.orthant_qmc(np.array([z]), np.eye(1), seed=71)
                self._assert_usable(result, 5e-4)
                self.assertAlmostEqual(result.log_probability, float(log_ndtr(z)), delta=2e-13)
                assert_allclose(result.probability, ndtr(z), rtol=2e-13, atol=0)

    def test_independent_probability_is_an_exact_product_including_tails(self):
        for z in (np.array([-0.7, 0.2, 1.4, -0.3]), np.array([-9.0, -8.0, -7.0])):
            with self.subTest(z=z.tolist()):
                result = gg.orthant_qmc(z, np.eye(len(z)), seed=32)
                self._assert_usable(result, 5e-4)
                expected_log = float(np.sum(log_ndtr(z)))
                self.assertAlmostEqual(result.log_probability, expected_log, delta=4e-13)
                assert_allclose(result.probability, np.exp(expected_log), rtol=4e-13, atol=0)

    def test_zero_mean_bivariate_labels_match_arcsine_identity(self):
        for rho in (-0.85, -0.35, 0.0, 0.6, 0.97):
            with self.subTest(rho=rho):
                tolerance = 8e-4
                result = gg.orthant_qmc(
                    np.zeros(2), np.array([[1.0, rho], [rho, 1.0]]), seed=413,
                    relative_tolerance=tolerance, initial_power=8, max_power=14,
                )
                self._assert_usable(result, tolerance)
                expected = 0.25 + np.arcsin(rho) / (2 * np.pi)
                self.assertAlmostEqual(result.probability, expected,
                                       delta=max(6 * result.error, 2e-7 * expected))

    def test_nonzero_correlated_labels_match_independent_quadrature(self):
        for a, b, rho in ((0.26356, -0.98664, 0.59434), (-2.0, -1.0, -0.6)):
            with self.subTest(a=a, b=b, rho=rho):
                expected, oracle_error = _bivariate_quad(a, b, rho)
                tolerance = 1e-3
                result = gg.orthant_qmc(
                    np.array([a, b]), np.array([[1.0, rho], [rho, 1.0]]), seed=189,
                    relative_tolerance=tolerance, initial_power=8, max_power=15,
                )
                self._assert_usable(result, tolerance)
                self.assertAlmostEqual(result.probability, expected,
                                       delta=max(6 * result.error + oracle_error, expected * 3e-7))
                self.assertAlmostEqual(result.log_probability, np.log(expected),
                                       delta=max(6 * result.relative_error, 3e-7))

    def test_five_correlated_points_match_order_statistic_identity(self):
        # Xi = Zi - Z0: all five Xi>0 iff Z0 is the minimum of six iid normals.
        correlation = np.full((5, 5), 0.5)
        np.fill_diagonal(correlation, 1.0)
        tolerance = 8e-4
        result = gg.orthant_qmc(
            np.zeros(5), correlation, seed=151, relative_tolerance=tolerance,
            initial_power=8, max_power=15,
        )
        self._assert_usable(result, tolerance)
        self.assertAlmostEqual(result.probability, 1 / 6,
                               delta=max(6 * result.error, 2e-7))

    def test_five_dimensional_rare_log_label_matches_common_factor_integral(self):
        # This oracle integrates a scalar common factor, not the Genz transform.
        z = np.array([-4.0, -3.5, -3.0, -2.8, -4.2])
        rho = 0.6
        correlation = np.full((5, 5), rho)
        np.fill_diagonal(correlation, 1.0)
        expected, oracle_error = quad(
            lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
            * np.prod(ndtr((z + np.sqrt(rho) * x) / np.sqrt(1 - rho))),
            -12, 12, epsabs=1e-16, epsrel=2e-10, limit=250,
        )
        tolerance = 1e-3
        result = gg.orthant_qmc(
            z, correlation, seed=153, relative_tolerance=tolerance,
            initial_power=8, max_power=15,
        )
        self._assert_usable(result, tolerance)
        self.assertLess(expected, 1e-5)
        self.assertAlmostEqual(result.probability, expected,
                               delta=max(6 * result.error + oracle_error, expected * 3e-7))
        self.assertAlmostEqual(result.log_probability, np.log(expected),
                               delta=max(6 * result.relative_error, 3e-7))

    def test_positive_origin_is_joint_probability_ratio_not_equality_conditioning(self):
        rho = np.exp(-0.5)
        result = gg.positive_origin_probability(
            np.zeros(2), np.array([[1.0, rho], [rho, 1.0]]), seed=519,
            relative_tolerance=8e-4, initial_power=8, max_power=14,
        )
        self._assert_usable(result, 8e-4)
        expected = 0.5 + np.arcsin(rho) / np.pi
        self.assertAlmostEqual(result.probability, expected, delta=max(6 * result.error, 2e-7))
        # Conditioning F0=0 would instead leave a centered scalar Gaussian.
        self.assertGreater(abs(result.probability - 0.5), 0.1)

    def test_positive_origin_ratio_stays_stable_for_rare_independent_origin(self):
        result = gg.positive_origin_probability(np.array([-12.0, 0.3]), np.eye(2), seed=39)
        self._assert_usable(result, 5e-4)
        self.assertAlmostEqual(result.log_probability, float(log_ndtr(0.3)), delta=3e-13)
        assert_allclose(result.probability, ndtr(0.3), rtol=3e-13)


class DatasetIsolationTests(unittest.TestCase):
    def test_family_split_is_deterministic_and_shared_by_variants(self):
        split = [gg.family_split(i, seed=813) for i in range(2000)]
        self.assertEqual(split, [gg.family_split(i, seed=813) for i in range(2000)])
        self.assertEqual(set(split), {0, 1, 2})
        counts = np.bincount(split, minlength=3) / len(split)
        assert_allclose(counts, [6 / 7, 1 / 14, 1 / 14], atol=0.035, rtol=0)
        for group in range(80):
            self.assertEqual(len({gg.family_split(group, seed=813) for _ in range(4)}), 1)

    def test_generated_dataset_has_no_family_leakage_or_unqualified_labels(self):
        with tempfile.TemporaryDirectory(prefix="gaussian-orthant-test-") as folder:
            path = Path(folder) / "fixture.npz"
            command = [
                sys.executable, "-B", "-m", "tools.gaussian_orthant_nn.generate", "--output", str(path),
                "--groups", "20", "--seed", "17", "--initial-power", "8", "--max-power", "12",
                "--relative-tolerance", "0.002",
            ]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       timeout=120, env=environment)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with np.load(path, allow_pickle=False) as data:
                required = {"z", "corr", "dim", "logp", "group_id", "split", "mode",
                            "probability", "label_error", "label_relative_error", "label_samples", "positions"}
                self.assertTrue(required.issubset(data.files), required - set(data.files))
                self.assertGreater(len(data["dim"]), 10)
                self.assertTrue(np.isfinite(data["logp"]).all())
                self.assertTrue(np.isfinite(data["label_relative_error"]).all())
                self.assertTrue(np.all(data["label_error"] >= 0))
                self.assertTrue(np.all(data["label_relative_error"] <= 0.002 * (1 + 1e-6)))
                self.assertTrue(set(np.unique(data["split"])).issubset({0, 1, 2}))
                paired = 0
                for group in np.unique(data["group_id"]):
                    rows = data["group_id"] == group
                    self.assertEqual(len(np.unique(data["split"][rows])), 1)
                    paired += {0, 1}.issubset(set(data["mode"][rows]))
                self.assertGreater(paired, 0, "The smoke dataset must exercise paired prior/surface families")
                assert_allclose(data["probability"], np.exp(data["logp"]), rtol=3e-6, atol=0)
            with (path.parent / "generation.json").open(encoding="utf-8") as stream:
                self.assertIsInstance(json.load(stream), dict)


class NetworkFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tools.gaussian_orthant_nn import infer, model
        cls.torch, cls.infer, cls.model = torch, infer, model

    def test_inference_standardization_removes_physical_scales(self):
        mean = np.array([0.2, -0.7, 1.0])
        covariance = np.array([[1.3, 0.1, -0.2], [0.1, 0.8, 0.15], [-0.2, 0.15, 1.7]])
        scales = np.array([0.02, 8.0, 0.4])
        a = self.infer.prepare_gaussian(mean, covariance)
        b = self.infer.prepare_gaussian(mean * scales, covariance * np.outer(scales, scales))
        self.assertEqual(a.dimension, b.dimension)
        assert_allclose(a.z, b.z, rtol=2e-6, atol=2e-7)
        assert_allclose(a.corr, b.corr, rtol=2e-6, atol=2e-7)

    def test_feature_sorting_permutes_correlations_and_ignores_padding(self):
        torch = self.torch
        z = torch.tensor([[0.2, -0.6, 0.9, 99.0, -88.0]])
        corr = torch.eye(5).unsqueeze(0)
        corr[0, :3, :3] = torch.tensor([[1.0, 0.2, -0.3], [0.2, 1.0, 0.1], [-0.3, 0.1, 1.0]])
        dim = torch.tensor([3])
        expected = self.model.make_features(z, corr, dim)
        permutation = torch.tensor([2, 0, 1, 3, 4])
        observed = self.model.make_features(z[:, permutation], corr[:, permutation][:, :, permutation], dim)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        altered_z, altered_corr = z.clone(), corr.clone()
        altered_z[:, 3:] = torch.tensor([[-321.0, 456.0]])
        altered_corr[:, 3:, :] = 0.7
        altered_corr[:, :, 3:] = 0.7
        torch.testing.assert_close(self.model.make_features(altered_z, altered_corr, dim), expected, rtol=0, atol=0)

    def test_network_preserves_exact_scalar_probability_and_marginal_upper_bound(self):
        torch = self.torch
        torch.manual_seed(348)
        network = self.model.OrthantNet(max_dim=5, width=16, depth=2).eval()
        z = torch.tensor([[-9.0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0], [-0.3, 0.8, 1.2, 0, 0]])
        corr = torch.eye(5).repeat(3, 1, 1)
        dim = torch.tensor([1, 1, 3])
        with torch.no_grad():
            logp = network(z, corr, dim).cpu().numpy()
        assert_allclose(logp[:2], log_ndtr(np.array([-9.0, 0.4])), rtol=1e-6, atol=3e-6)
        self.assertTrue(np.isfinite(logp).all())
        self.assertLessEqual(logp[2], float(log_ndtr(-0.3)) + 1e-6)

    def test_network_respects_frechet_lower_bound_without_a_rare_probability_floor(self):
        torch = self.torch
        network = self.model.OrthantNet(max_dim=5, width=16, depth=2).eval()
        z = torch.tensor([[3.0, 2.5, 4.0, 2.8, 3.2],
                          [-0.4, 0.3, 2.0, 0.0, 0.0],
                          [-9.0, -8.0, -7.0, 0.0, 0.0]])
        dim = torch.tensor([5, 3, 3])
        corr = torch.eye(5).repeat(3, 1, 1)
        with torch.no_grad():
            logp = network(z, corr, dim).cpu().numpy()
        self.assertTrue(np.isfinite(logp).all())
        for index, n in enumerate(dim.tolist()):
            locations = z[index, :n].cpu().numpy().astype(np.float64)
            lower = max(0.0, 1.0 - np.sum(ndtr(-locations)))
            upper = np.min(ndtr(locations))
            self.assertGreaterEqual(np.exp(float(logp[index])), lower - 2e-7)
            self.assertLessEqual(np.exp(float(logp[index])), upper + 2e-7)
        self.assertLess(logp[2], -60.0, "The Frechet bound must not impose a tiny-P floor")


class CudaLabelTests(unittest.TestCase):
    def test_cuda_matches_cpu_log_tails_deterministic_pivots_and_scalar_oracle(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        if os.environ.get("GAUSSIAN_ORTHANT_SKIP_CUDA") == "1":
            self.skipTest("CUDA test deferred while another process uses the GPU")
        if not hasattr(gg, "_genz_replicates_cuda"):
            self.skipTest("This checkout has no CUDA label backend")

        rho = 0.6
        corr = np.full((5, 5), rho)
        np.fill_diagonal(corr, 1.0)
        means = np.array([[-40.0, -39.0, -38.0, -37.0, -36.0],
                          [-4.0, -3.5, -3.0, -2.8, -4.2]])
        factors = np.repeat(np.linalg.cholesky(corr)[None], 2, axis=0)
        cpu = gg._genz_replicates(means, factors, 9, 4, 151)
        gpu = gg._genz_replicates_cuda(means, factors, 9, 4, 151)
        self.assertTrue(np.isfinite(cpu).all())
        self.assertTrue(np.isfinite(gpu).all())
        self.assertTrue(np.all(cpu[:, 0] < -700), "Fixture must exercise inverse-log-CDF tails")
        assert_allclose(gpu, cpu, rtol=2e-12, atol=2e-10)

        # Direct backend calls also exercise zero diagonal factors which the
        # public API may otherwise resolve through an analytic special case.
        deterministic_means = np.array([[0.2, -0.3], [0.2, 0.0], [0.2, 1.0]])
        deterministic_factors = np.array([[[1.0, 0.0], [1.0, 0.0]],
                                          [[1.0, 0.0], [0.0, 0.0]],
                                          [[1.0, 0.0], [0.0, 0.0]]])
        cpu = gg._genz_replicates(deterministic_means, deterministic_factors, 8, 4, 153)
        gpu = gg._genz_replicates_cuda(deterministic_means, deterministic_factors, 8, 4, 153)
        assert_allclose(gpu, cpu, rtol=2e-12, atol=2e-10)
        self.assertTrue(np.isneginf(gpu[:, 1]).all())

        expected, oracle_error = quad(
            lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
            * np.prod(ndtr((means[1] + np.sqrt(rho) * x) / np.sqrt(1 - rho))),
            -12, 12, epsabs=1e-16, epsrel=2e-10, limit=250,
        )
        observed = gg.orthant_qmc(means[1], corr, seed=153, backend="cuda",
                                  relative_tolerance=1e-3, initial_power=8, max_power=15)
        self.assertEqual(observed.status, "ok")
        self.assertAlmostEqual(observed.probability, expected,
                               delta=max(6 * observed.error + oracle_error, expected * 3e-7))


if __name__ == "__main__":
    unittest.main()
