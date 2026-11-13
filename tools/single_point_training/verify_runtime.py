"""Compare native renderer inference with the trained PyTorch checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .model import FEATURE_NAMES, SinglePointResidualNet
from .train import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, model, output = Path(args.data), Path(args.model), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261113)
    selected, seen = [], 0
    with (data / "samples.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        for row in reader:
            if int(row["split"]) != 2:
                continue
            seen += 1
            if len(selected) < 4096:
                selected.append(row)
            else:
                index = int(rng.integers(seen))
                if index < len(selected):
                    selected[index] = row
    if not selected:
        raise ValueError("no held-out test rows")
    probe = output / "inputs.csv"
    with probe.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(selected)
    native = output / "native.csv"
    subprocess.run([str(Path(args.native_test).resolve()), "--predict",
                    str((model / "model.json").resolve()), str(probe.resolve()),
                    str(native.resolve())], check=True)
    checkpoint = torch.load(model / "best.pt", map_location="cpu", weights_only=True)
    export = json.loads((model / "model.json").read_text(encoding="utf-8"))
    net = SinglePointResidualNet(checkpoint["model_config"]["width"],
                                checkpoint.get("feature_transform", "identity"))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    features = torch.tensor([[float(row[name]) for name in FEATURE_NAMES] for row in selected], dtype=torch.float32)
    logp = torch.tensor([float(row["log_p_history"]) for row in selected], dtype=torch.float64)
    expected = net.inference_log_correction(features, logp, export["deployment"]["correction_cap"]).numpy()
    actual = np.atleast_1d(np.genfromtxt(native, delimiter=",", names=True))["log_correction"]
    error = np.abs(actual - expected)
    passed = bool(np.isfinite(error).all() and np.allclose(actual, expected, rtol=2e-5, atol=2e-5))
    report = {"passed": passed, "count": len(selected), "test_population": seen,
              "maximum_absolute_log_error": float(error.max()), "mean_absolute_log_error": float(error.mean()),
              "relative_tolerance": 2e-5, "absolute_tolerance": 2e-5,
              "model_sha256": sha256(model / "model.json"), "checkpoint_sha256": sha256(model / "best.pt"),
              "native_test_sha256": sha256(Path(args.native_test)), "feature_transform": net.feature_transform}
    write_json(output / "verification.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise AssertionError("native / PyTorch inference disagreement")


if __name__ == "__main__":
    main()
