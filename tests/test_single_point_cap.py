"""Checks that a deployment cap does not alter the learned correction law."""
from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
import torch

from tools.single_point_training.model import (
    MODEL_SCHEMA,
    SinglePointResidualNet,
    bounded_log_correction,
    cap_log_correction,
    validate_correction_cap,
)
from tools.single_point_training.train import export_model, parser


class InferenceCapTests(unittest.TestCase):
    def test_cap_preserves_attenuation_and_identity(self):
        values = torch.tensor([-1000., -10., -.1, 0., .1, math.log(2), 10.], dtype=torch.float64)
        before = values.clone()
        capped = cap_log_correction(values, 2.)
        assert_allclose(capped.numpy(), np.minimum(values.numpy(), math.log(2)), rtol=0, atol=0)
        self.assertTrue(torch.equal(values, before), "deployment must not mutate training predictions")
        self.assertTrue(torch.equal(capped[:4], values[:4]))

    def test_none_disables_cap_without_changing_positive_tail(self):
        values = torch.tensor([-5., 0., 10., 1000.], dtype=torch.float64)
        self.assertTrue(torch.equal(cap_log_correction(values, None), values))
        self.assertIsNone(validate_correction_cap(None))

    def test_unit_cap_allows_only_decreased_or_unchanged_extinction(self):
        values = torch.tensor([-2., 0., .1, 20.], dtype=torch.float64)
        assert_allclose(cap_log_correction(values, 1.).numpy(), [-2., 0., 0., 0.])

    def test_invalid_caps_cannot_erase_the_analytic_baseline(self):
        for cap in (-1., 0., .99, float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(cap=cap), self.assertRaises((TypeError, ValueError)):
                validate_correction_cap(cap)

    def test_cap_composes_with_the_analytic_history_envelope(self):
        residual = torch.tensor([-20., 0., 5., 1000.], dtype=torch.float64)
        for probability in (.001, .1, .8, 1.):
            logp = torch.full_like(residual, math.log(probability))
            unbounded_by_user = bounded_log_correction(residual, logp)
            used = cap_log_correction(unbounded_by_user, 2.)
            bound = min(2., 1/probability)
            self.assertTrue((used <= math.log(bound)+1e-14).all())

    def test_network_forward_and_gradients_remain_uncapped(self):
        model = SinglePointResidualNet()
        with torch.no_grad():
            model.network[-1].bias.fill_(3.)
        features = torch.zeros(1, 6)
        logp = torch.tensor([math.log(.1)], dtype=torch.float64)
        correction = model(features, logp)
        self.assertGreater(float(correction.detach()), math.log(2))
        correction.sum().backward()
        self.assertGreater(float(model.network[-1].bias.grad[0]), 0.)
        assert_allclose(cap_log_correction(correction.detach(), 2.).numpy(), [math.log(2)])

    def test_cli_default_and_explicit_disable(self):
        required = ["--data", "data", "--output", "model"]
        self.assertEqual(parser().parse_args(required).correction_cap, 2.)
        self.assertIsNone(parser().parse_args(required+["--correction-cap", "none"]).correction_cap)
        self.assertEqual(parser().parse_args(required+["--correction-cap", "1.5"]).correction_cap, 1.5)

    def test_inference_helper_applies_cap_without_gradients_or_parameter_mutation(self):
        model = SinglePointResidualNet()
        with torch.no_grad():
            model.network[-1].bias.fill_(3.)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        features = torch.zeros(1, 6)
        logp = torch.tensor([math.log(.1)], dtype=torch.float64)
        result = model.inference_log_correction(features, logp)
        self.assertFalse(result.requires_grad)
        assert_allclose(result.numpy(), [math.log(2)])
        self.assertGreater(float(model.inference_log_correction(features, logp, None)), math.log(2))
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))

    def test_export_declares_cap_only_in_deployment_and_preserves_weights(self):
        model = SinglePointResidualNet()
        data = {"manifest": {"sigma": .02, "length_scale": .2},
                "features": np.zeros((3, 6)),
                "partitions": [np.array([0]), np.array([1]), np.array([2])]}
        capped = export_model(model, data, {}, correction_cap=2.)
        uncapped = export_model(model, data, {}, correction_cap=None)
        self.assertEqual(capped["schema"], MODEL_SCHEMA)
        self.assertEqual(capped["layers"], uncapped["layers"])
        deployment = capped["deployment"]
        self.assertEqual(deployment["correction_cap"], 2.)
        self.assertEqual(deployment["stage"], "inference_only")
        self.assertIsNone(deployment["lower_cap"])
        self.assertFalse(deployment["applied_to_training_loss"])
        self.assertFalse(deployment["applied_to_checkpoint_selection"])
        self.assertIn("min(1 / p_history, correction_cap)", deployment["sampling_majorant"])
        self.assertIn("log(correction_cap)", capped["output"]["formula"])
        self.assertIsNone(uncapped["deployment"]["correction_cap"])
        self.assertEqual(uncapped["output"]["formula"], "uncapped_log_correction")
        self.assertEqual(uncapped["deployment"]["sampling_majorant"], "M_base / p_history")


if __name__ == "__main__":
    unittest.main()
