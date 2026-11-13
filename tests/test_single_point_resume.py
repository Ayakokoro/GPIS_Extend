"""Resume preserves the optimizer trajectory; legacy weights are explicit warm starts."""
from __future__ import annotations

import copy
import io
import json
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid

import numpy as np
import torch

from test_single_point_training import make_dataset
from tools.single_point_training import train as training


@contextmanager
def experiment():
    parent = Path(__file__).resolve().parents[1] / "outputs/single_point_resume_tests"
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / uuid.uuid4().hex
    root.mkdir()
    data = root / "data"
    data.mkdir()
    def labels(rows):
        for row in rows:
            correction = -.2 - .1 * (row["z_x"] % 10)
            row.update(log_correction=correction, log_alpha_history=-1 + correction)
    make_dataset(data, labels)
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(parent.resolve()):
            raise RuntimeError("refusing cleanup outside the test workspace")
        shutil.rmtree(root)


def run(root, name, epochs, resume=None, extra=()):
    arguments = ["--data", str(root / "data"), "--output", str(root / name),
                 "--device", "cpu", "--epochs", str(epochs), "--width", "4", "--batch-size", "1",
                 "--cpu-threads", "1", "--patience", "0", "--learning-rate", ".01", "--weight-decay", ".001"]
    if resume:
        arguments += ["--resume", str(root / resume)]
    with redirect_stdout(io.StringIO()):
        return training.train(training.parser().parse_args(arguments + list(extra)))


def checkpoint(root, name):
    return torch.load(root / name, map_location="cpu", weights_only=True)


class ResumeTests(unittest.TestCase):
    def assert_state_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_exact_resume_matches_uninterrupted_with_scheduler_plateau(self):
        actual_metrics = training.error_metrics
        def fixed_validation(*args, **kwargs):
            result = actual_metrics(*args, **kwargs)
            result["log_mae"] = 1.0
            return result
        with experiment() as root, mock.patch.object(training, "error_metrics", side_effect=fixed_validation):
            run(root, "full", 14)
            run(root, "prefix", 6)
            report = run(root, "resumed", 8, "prefix/latest.pt")
            full = checkpoint(root, "full/latest.pt")
            resumed = checkpoint(root, "resumed/latest.pt")
            self.assert_state_equal(full["state_dict"], resumed["state_dict"])
            self.assert_state_equal(full["best_state_dict"], resumed["best_state_dict"])
            self.assertEqual(full["resume_state"]["scheduler"], resumed["resume_state"]["scheduler"])
            self.assertLess(full["resume_state"]["optimizer"]["param_groups"][0]["lr"], .01)
            self.assertTrue(torch.equal(full["resume_state"]["random"]["torch_cpu"],
                                        resumed["resume_state"]["random"]["torch_cpu"]))
            self.assertEqual(report["continuation"]["kind"], "exact")
            self.assertEqual((report["starting_epoch"], report["final_epoch"], report["epochs_completed"]), (6, 14, 8))
            a = np.genfromtxt(root / "full/history.csv", names=True, delimiter=",")
            b = np.genfromtxt(root / "resumed/history.csv", names=True, delimiter=",")
            for field in ("epoch", "train_loss", "validation_log_mae", "validation_log_p95", "learning_rate"):
                np.testing.assert_array_equal(a[field][6:], b[field])
            best = checkpoint(root, "resumed/best.pt")
            self.assertEqual(best["resume_state"]["epoch"], best["best_epoch"])
            self.assertIn("optimizer", best["resume_state"])

    def test_legacy_warm_start_keeps_parent_best_without_improvement(self):
        with experiment() as root:
            run(root, "parent", 2)
            legacy = checkpoint(root, "parent/best.pt")
            for key in ("resume_state", "best_resume_state", "best_state_dict", "manifest_sha256", "continuation"):
                legacy.pop(key)
            legacy["best_epoch"] = 295
            torch.save(legacy, root / "parent/legacy.pt")
            actual_metrics = training.error_metrics
            def same_score(*args, **kwargs):
                result = actual_metrics(*args, **kwargs)
                result["log_mae"] = .25
                return result
            with mock.patch.object(training, "error_metrics", side_effect=same_score):
                report = run(root, "warm", 2, "parent/legacy.pt", ["--learning-rate", ".00025"])
            best = checkpoint(root, "warm/best.pt")
            self.assert_state_equal(legacy["state_dict"], best["state_dict"])
            self.assertEqual(best["best_epoch"], 295)
            self.assertEqual((report["starting_epoch"], report["final_epoch"]), (295, 297))
            self.assertEqual(report["continuation"]["kind"], "warm_start")
            self.assertFalse(report["continuation"]["optimizer_restored"])
            self.assertEqual(report["continuation"]["initial_learning_rate"], .00025)

    def test_resume_rejects_changed_dataset(self):
        with experiment() as root:
            run(root, "parent", 1)
            data = root / "data/samples.csv"
            data.write_text(data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset SHA256"):
                run(root, "bad", 1, "parent/best.pt")

    def test_resume_rejects_changed_transform_and_exact_contract(self):
        with experiment() as root:
            run(root, "parent", 1)
            with self.assertRaisesRegex(ValueError, "feature transform"):
                run(root, "transform", 1, "parent/best.pt", ["--feature-transform", "asinh_z"])
            with self.assertRaisesRegex(ValueError, "training contract"):
                run(root, "batch", 1, "parent/latest.pt", ["--batch-size", "2"])

    def test_exact_resume_rejects_changed_manifest(self):
        with experiment() as root:
            run(root, "parent", 1)
            path = root / "data/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["notes"] = "changed sampling description"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                run(root, "bad", 1, "parent/latest.pt")


if __name__ == "__main__":
    unittest.main()
