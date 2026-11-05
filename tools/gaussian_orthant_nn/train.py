"""Reproducible shared Gaussian-orthant training on group-separated NPZ data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

# Required by deterministic CUDA matrix multiplication, before CUDA startup.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy.special import log_ndtr
import torch
from torch.nn import functional as F

try:
    from .model import FEATURE_SPEC, OrthantNet
except ImportError:
    from model import FEATURE_SPEC, OrthantNet


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"z", "corr", "dim", "logp", "group_id", "split"}
        if missing := required.difference(archive.files):
            raise ValueError(f"dataset is missing {sorted(missing)}")
        data = {name: archive[name] for name in archive.files}
    count = len(data["dim"])
    if data["z"].shape != (count, 5) or data["corr"].shape != (count, 5, 5):
        raise ValueError("expected z[N,5], corr[N,5,5]")
    for key in ("dim", "logp", "group_id", "split"):
        if data[key].shape != (count,):
            raise ValueError(f"{key} must have one entry per row")
    if not np.isfinite(data["z"]).all() or not np.isfinite(data["corr"]).all() or not np.isfinite(data["logp"]).all():
        raise ValueError("dataset must contain finite inputs and finite log labels")
    if np.any((data["dim"] < 1) | (data["dim"] > 5)) or np.any(~np.isin(data["split"], [0, 1, 2])):
        raise ValueError("invalid dimension or split label")
    groups = []
    for partition in range(3):
        selected = data["split"] == partition
        if not np.any(selected):
            raise ValueError("train, validation and test must all be nonempty")
        groups.append(np.unique(data["group_id"][selected]))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if np.intersect1d(groups[first], groups[second]).size:
            raise ValueError("profile group leakage across train/validation/test")
    valid = np.arange(5)[None, :] < data["dim"][:, None]
    upper = np.min(np.where(valid, log_ndtr(data["z"]), np.inf), axis=1)
    if np.any(data["logp"] > upper + 1e-4):
        raise ValueError("labels violate the marginal probability upper bound")
    return data


def training_domain(data) -> dict:
    selected = data["split"] == 0
    z, corr, dim = data["z"][selected], data["corr"][selected], data["dim"][selected]
    valid = np.arange(5)[None, :] < dim[:, None]
    minimum_eigenvalue = 1.0
    for n in range(2, 6):
        block = corr[dim == n, :n, :n].astype(np.float64)
        if len(block):
            minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(block).min()))
    return {"minimum_z": float(z[valid].min()), "maximum_z": float(z[valid].max()),
            "minimum_correlation_eigenvalue": max(0.0, minimum_eigenvalue),
            "supported_dimensions": sorted(int(n) for n in np.unique(dim)),
            "general_singular_covariances_supported": False,
            "standardization": "mean / sqrt(diagonal covariance)",
            "provenance_mode_is_not_an_input": True,
            "permutation_invariance": "canonical z ordering; equal-z ties are not strictly invariant"}


def predict(model, z, corr, dim, batch_size=8192):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(z), batch_size):
            result = model(z[start:start + batch_size], corr[start:start + batch_size], dim[start:start + batch_size])
            chunks.append(result.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks)


def error_metrics(prediction, target) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    with np.errstate(over="ignore", under="ignore"):
        relative = np.abs(np.expm1(error))
        probability_error = np.abs(np.exp(prediction) - np.exp(target))
    finite_relative = relative[np.isfinite(relative)]
    return {"count": int(len(error)), "log_mae": float(absolute.mean()),
            "log_rmse": float(np.sqrt(np.mean(error ** 2))),
            "log_p50": float(np.quantile(absolute, .5)), "log_p95": float(np.quantile(absolute, .95)),
            "log_p99": float(np.quantile(absolute, .99)), "log_max": float(absolute.max()),
            "probability_mae": float(probability_error.mean()),
            "relative_error_median": float(np.median(relative)),
            "relative_error_p95": float(np.quantile(finite_relative, .95)) if len(finite_relative) else None,
            "relative_error_nonfinite_count": int((~np.isfinite(relative)).sum())}


def _json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def train(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    data = load_dataset(args.data)
    output = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    domain = training_domain(data)
    model = OrthantNet(width=args.width, depth=args.depth).to(device)
    z = torch.as_tensor(data["z"], dtype=torch.float32, device=device)
    corr = torch.as_tensor(data["corr"], dtype=torch.float32, device=device)
    dim = torch.as_tensor(data["dim"], dtype=torch.long, device=device)
    target = torch.as_tensor(data["logp"], dtype=torch.float32, device=device)
    partitions = [torch.as_tensor(np.flatnonzero(data["split"] == split), device=device) for split in range(3)]
    training = partitions[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, min_lr=1e-5)
    config = vars(args).copy()
    config.update({"device_resolved": str(device), "torch_version": str(torch.__version__),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "feature_spec": FEATURE_SPEC, "model_config": model.configuration(), "domain": domain,
                   "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                   "split_counts": [int(len(indices)) for indices in partitions],
                   "split_group_counts": [int(len(np.unique(data["group_id"][data["split"] == split]))) for split in range(3)],
                   "loss": "SmoothL1(log_probability_error, beta=1); exact n1 bypass"})
    _json(output / "config.json", config)
    best_score, best_epoch = np.inf, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = training[torch.randperm(len(training), device=device)]
        loss_sum = 0.0
        for first in range(0, len(order), args.batch_size):
            indices = order[first:first + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(z[indices], corr[indices], dim[indices])
            loss = F.smooth_l1_loss(prediction, target[indices], beta=1.0)
            if not torch.isfinite(loss):
                raise ArithmeticError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        validation = partitions[1]
        estimated = predict(model, z[validation], corr[validation], dim[validation])
        validation_target = data["logp"][data["split"] == 1]
        metrics = error_metrics(estimated, validation_target)
        score = metrics["log_mae"]
        scheduler.step(score)
        record = {"epoch": epoch, "train_huber": loss_sum / len(training),
                  "validation_log_mae": score, "validation_log_p95": metrics["log_p95"],
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": time.perf_counter() - start_time}
        history.append(record)
        if score < best_score:
            best_score, best_epoch = score, epoch
            state = {"format_version": 1, "model_config": model.configuration(),
                     "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                     "domain": domain, "feature_spec": FEATURE_SPEC, "training_config": config,
                     "best_epoch": epoch, "best_validation_log_mae": score}
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(record), flush=True)
            _json(output / "history.json", history)
        if epoch - best_epoch >= args.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    report = {"best_epoch": best_epoch, "epochs_completed": len(history),
              "training_seconds": time.perf_counter() - start_time, "splits": {}}
    for split, name in enumerate(("train", "validation", "test")):
        indices = partitions[split]
        prediction = predict(model, z[indices], corr[indices], dim[indices])
        mask = data["split"] == split
        truth = data["logp"][mask]
        entry = error_metrics(prediction, truth)
        entry["by_dimension"] = {}
        for n in range(1, 6):
            local = data["dim"][mask] == n
            if local.any():
                entry["by_dimension"][str(n)] = error_metrics(prediction[local], truth[local])
        report["splits"][name] = entry
    _json(output / "history.json", history)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _json(output / "metrics.json", report)
    print(json.dumps({"checkpoint": str(checkpoint), "best_epoch": best_epoch,
                      "test": report["splits"]["test"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="outputs/gaussian-orthant-nn/dataset.npz")
    parser.add_argument("--output-dir", default="outputs/gaussian-orthant-nn")
    parser.add_argument("--checkpoint", default="models/gaussian-orthant-nn/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.cpu_threads) < 1:
        parser.error("epochs, patience, batch size and CPU threads must be positive")
    train(args)


if __name__ == "__main__":
    main()
