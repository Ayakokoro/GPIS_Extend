"""Train a bounded single-point residual from the native Tungsten CSV export.

Example:
    python -m tools.single_point_training.train --data outputs/my-data \
        --output models/my-single-point --device cpu

Data splitting belongs to the native exporter. Complete view and ray groups
must be held out; this consumer never randomly splits rows of the same ray.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from .model import FEATURE_NAMES, MODEL_SCHEMA, SCHEMA, SinglePointResidualNet, validate_correction_cap


REQUIRED_COLUMNS = ("view_id", "ray_id", "split", *FEATURE_NAMES,
                    "log_alpha_base", "log_alpha_history", "log_p_history", "log_correction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_file_metadata(manifest: dict) -> dict:
    """Fingerprint each existing source file once, using bounded read memory.

    These are training-time fingerprints. They do not retroactively prove that
    an unrecorded asset was identical when the native data was generated.
    Missing source files are allowed so exported datasets remain portable.
    """
    source = manifest.get("source", {})
    result = {"captured_at": "training startup", "generation_time_identity_certified": False,
              "files": {}}
    cache = {}
    for key in ("scene", "sdf_asset"):
        value = source.get(key)
        if not value:
            continue
        if not isinstance(value, str):
            raise ValueError(f"source.{key} must be a file path string")
        path = Path(value)
        if not path.is_absolute():
            result["files"][key] = {"path": value, "status": "unresolved_relative_path"}
            continue
        canonical = str(path.resolve())
        if canonical not in cache:
            try:
                before = path.stat()
                if not path.is_file():
                    raise FileNotFoundError("not a regular file")
                digest = sha256(path)
                after = path.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    raise RuntimeError(f"source file changed while hashing: {path}")
                cache[canonical] = {"path": canonical, "status": "hashed", "size_bytes": after.st_size,
                                    "mtime_ns": after.st_mtime_ns, "sha256": digest}
            except OSError as error:
                cache[canonical] = {"path": canonical, "status": "unavailable", "reason": str(error)}
        result["files"][key] = cache[canonical]
    return result


def _integer_column(values: np.ndarray, name: str) -> np.ndarray:
    if np.any(values < 0) or np.any(values > 2**53) or np.any(values != np.floor(values)):
        raise ValueError(f"{name} must contain exactly representable nonnegative integers")
    return values.astype(np.int64)


def load_dataset(directory: str | Path) -> dict:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"expected dataset schema {SCHEMA}")
    if manifest.get("status") != "complete":
        raise ValueError("dataset must have status=complete; inspected, interrupted and camera-only diagnostic exports cannot train")
    if (root / "samples.partial.csv").exists():
        raise ValueError("dataset still contains samples.partial.csv; completion is ambiguous")
    expected_rows = manifest.get("rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 1:
        raise ValueError("completed manifest must declare a positive integer row count")
    expected_splits = manifest.get("split_counts")
    if (not isinstance(expected_splits, list) or len(expected_splits) != 3 or
            any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in expected_splits) or
            sum(expected_splits) != expected_rows):
        raise ValueError("manifest split_counts must contain three positive counts summing to rows")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("manifest feature_names must match the six-feature model order")
    for parameter in ("sigma", "length_scale"):
        value = manifest.get(parameter)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"manifest {parameter} must be one positive finite constant")
    path = root / "samples.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("duplicate CSV column names")
    if missing := set(REQUIRED_COLUMNS).difference(header):
        raise ValueError(f"dataset is missing columns {sorted(missing)}")
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8"))
    if len(rows) == 0:
        raise ValueError("dataset contains no samples")
    if len(rows) != expected_rows:
        raise ValueError("CSV row count does not match the completed manifest")
    data = {name: np.asarray(rows[name]) for name in REQUIRED_COLUMNS}
    if any(not np.isfinite(values).all() for values in data.values()):
        raise ValueError("dataset inputs and log labels must all be finite")
    for name in ("view_id", "ray_id", "split"):
        data[name] = _integer_column(data[name], name)
    if np.any(~np.isin(data["split"], (0, 1, 2))):
        raise ValueError("split must be 0 (train), 1 (validation), or 2 (test)")
    features = np.column_stack([data[name] for name in FEATURE_NAMES])
    if np.any(np.abs(features[:, 3:]) > 1 + 1e-10):
        raise ValueError("correlations must lie in [-1,1]")
    if np.any(data["rho_xu"] < -1e-10):
        raise ValueError("negative current/history correlation invalidates this teacher envelope")
    a, b, c = features[:, 3], features[:, 4], features[:, 5]
    determinant = 1 + 2*a*b*c - a*a - b*b - c*c
    if np.any(determinant < -1e-8):
        raise ValueError("the three input correlations do not define a positive semidefinite Gaussian")
    logp = data["log_p_history"]
    if np.any(logp > 0):
        raise ValueError("history log probability must not be positive")
    expected_logp = torch.special.log_ndtr(torch.from_numpy(data["z_u"])).numpy()
    if not np.allclose(logp, expected_logp, rtol=2e-6, atol=2e-8):
        raise ValueError("history probability disagrees with the Gaussian history feature")
    expected = data["log_alpha_history"] - data["log_alpha_base"]
    if not np.allclose(data["log_correction"], expected, rtol=2e-7, atol=2e-8):
        raise ValueError("log correction disagrees with the two hazard labels")
    if np.any(data["log_correction"] > -logp + 2e-6):
        raise ValueError("teacher correction exceeds the analytic 1/p_history envelope")
    partitions = [np.flatnonzero(data["split"] == partition) for partition in range(3)]
    if any(len(indices) == 0 for indices in partitions):
        raise ValueError("train, validation and test partitions must all be nonempty")
    if [len(indices) for indices in partitions] != expected_splits:
        raise ValueError("CSV split counts do not match the completed manifest")
    for key in ("view_id", "ray_id"):
        groups = [np.unique(data[key][indices]) for indices in partitions]
        for first, second in ((0, 1), (0, 2), (1, 2)):
            if np.intersect1d(groups[first], groups[second]).size:
                raise ValueError(f"{key} leakage across train/validation/test")
    # A ray may have many query rows, but may never belong to different views.
    order = np.argsort(data["ray_id"], kind="stable")
    same_ray = np.diff(data["ray_id"][order]) == 0
    if np.any(same_ray & (np.diff(data["view_id"][order]) != 0)):
        raise ValueError("one ray_id is assigned to multiple views")
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("completed manifest must record generated views")
    view_splits = {}
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("manifest views must be objects")
        view_id, split = view.get("view_id"), view.get("split")
        if (isinstance(view_id, bool) or not isinstance(view_id, int) or view_id < 0 or
                isinstance(split, bool) or not isinstance(split, int) or split not in (0, 1, 2)):
            raise ValueError("manifest views require integer view_id and split")
        if view_id in view_splits:
            raise ValueError("manifest contains duplicate view_id entries")
        view_splits[view_id] = split
    for view_id in np.unique(data["view_id"]):
        selected = data["view_id"] == view_id
        if view_id not in view_splits or np.any(data["split"][selected] != view_splits[view_id]):
            raise ValueError("CSV view assignments disagree with the completed manifest")
    data.update(features=features, manifest=manifest, partitions=partitions,
                dataset_path=path, manifest_path=root / "manifest.json")
    return data


def training_normalization(data: dict, feature_transform: str = "identity") -> tuple[np.ndarray, np.ndarray]:
    training = data["features"][data["partitions"][0]]
    if feature_transform == "asinh_z":
        training = training.copy()
        training[:, :3] = np.arcsinh(training[:, :3])
    mean, scale = np.zeros(6), np.ones(6)
    mean[:3] = training[:, :3].mean(axis=0)
    stddev = training[:, :3].std(axis=0)
    # Constant features stay centered with unit scale. There is no clipping of
    # validation/test values to the training range.
    scale[:3] = np.where(stddev > 1e-8, stddev, 1.0)
    return mean, scale


def domain_metadata(data: dict) -> dict:
    features = data["features"][data["partitions"][0]]
    return {"minimum_features": features.min(axis=0).tolist(),
            "maximum_features": features.max(axis=0).tolist(),
            "feature_order": list(FEATURE_NAMES),
            "input_clipping": False,
            "scope": "One scene and fixed covariance; held-out views do not certify arbitrary new scenes.",
            "training_range_is_not_an_accuracy_certificate": True}


def error_metrics(prediction, target, log_base) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    weights = np.exp(log_base - np.max(log_base))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        relative = np.abs(np.expm1(error))
    finite = np.isfinite(relative)
    finite_relative = relative[finite]
    return {"count": int(len(error)), "log_mae": float(absolute.mean()),
            "log_rmse": float(np.sqrt(np.mean(error**2))),
            "log_p95": float(np.quantile(absolute, .95)),
            "log_p99": float(np.quantile(absolute, .99)), "log_max": float(absolute.max()),
            "base_hazard_weighted_log_mae": float(np.sum(weights*absolute)/np.sum(weights)),
            "relative_error_median": float(np.median(finite_relative)) if len(finite_relative) else None,
            "relative_error_p95": float(np.quantile(finite_relative, .95)) if len(finite_relative) else None,
            "relative_error_nonfinite_count": int((~finite).sum())}


def predict(model, features, logp, indices, batch_size=8192) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start+batch_size]
            chunks.append(model(features[selected], logp[selected]).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def export_model(model: SinglePointResidualNet, data: dict, training: dict,
                 correction_cap: float | None = 2.0) -> dict:
    correction_cap = validate_correction_cap(correction_cap)
    layers = []
    for layer in model.network:
        if isinstance(layer, torch.nn.Linear):
            layers.append({"type": "linear", "weights": layer.weight.detach().cpu().tolist(),
                           "bias": layer.bias.detach().cpu().tolist()})
        else:
            layers.append({"type": "relu"})
    return {"schema": MODEL_SCHEMA, "architecture": model.configuration(),
            "feature_order": list(FEATURE_NAMES),
            "feature_transform": model.feature_transform,
            "feature_mean": model.feature_mean.cpu().tolist(),
            "feature_scale": model.feature_scale.cpu().tolist(),
            "layers": layers, "mlp_dtype": "float32", "probability_transform_dtype": "float64",
            "output": {"quantity": "log_hazard_correction", "raw_quantity": "residual_f",
                       "uncapped_formula": "-logaddexp(log_p_history, log1mexp(log_p_history) - residual_f)",
                       "formula": "uncapped_log_correction" if correction_cap is None else "min(uncapped_log_correction, log(correction_cap))",
                       "hazard": "alpha_base * exp(log_hazard_correction)",
                       "bound": "alpha_base / p_history" if correction_cap is None else "alpha_base * min(1 / p_history, correction_cap)",
                       "empty_history": "use alpha_base without network"},
            "deployment": {"correction_cap": correction_cap, "stage": "inference_only",
                           "lower_cap": None, "applied_to_training_loss": False,
                           "applied_to_checkpoint_selection": False,
                           "sampling_majorant": "M_base / p_history" if correction_cap is None else "M_base * min(1 / p_history, correction_cap)",
                           "majorant_scope": "one fixed history anchor per interval; otherwise use a certified interval enclosure or cap * M_base"},
            "sigma": data["manifest"]["sigma"], "length_scale": data["manifest"]["length_scale"],
            "source_manifest": data["manifest"], "training_domain": domain_metadata(data),
            "training": training,
            "runtime_integration": "global_conditional_neural; distance and transmittance correction only; baseline collision gradient retained"}


def capture_random_state() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": int(numpy_state[2]), "has_gaussian": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["kind"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gaussian"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda = state["torch_cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("exact resume requires the same CUDA device count")
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def continuation_contract(args, device) -> dict:
    return {"batch_size": args.batch_size, "loss_weighting": args.loss_weighting,
            "weight_decay": args.weight_decay, "device": str(device),
            "cpu_threads": args.cpu_threads, "torch_version": str(torch.__version__),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                            if torch.cuda.is_available() else []}


def validate_resume(checkpoint, model, data, dataset_digest, manifest_digest, transform) -> None:
    if checkpoint.get("schema") != MODEL_SCHEMA:
        raise ValueError("resume checkpoint has an unsupported model schema")
    if checkpoint.get("dataset_sha256") != dataset_digest:
        raise ValueError("resume dataset SHA256 mismatch")
    if checkpoint.get("manifest_sha256", manifest_digest) != manifest_digest:
        raise ValueError("resume manifest SHA256 mismatch")
    if checkpoint.get("model_config") != model.configuration():
        raise ValueError("resume architecture mismatch")
    if checkpoint.get("feature_order") != list(FEATURE_NAMES):
        raise ValueError("resume feature order mismatch")
    if checkpoint.get("feature_transform", "identity") != transform:
        raise ValueError("resume feature transform mismatch")
    for name in ("sigma", "length_scale"):
        if checkpoint.get(name) != data["manifest"][name]:
            raise ValueError(f"resume {name} mismatch")
    for key in ("feature_mean", "feature_scale"):
        if not torch.equal(checkpoint["state_dict"][key].cpu(), model.state_dict()[key].cpu()):
            raise ValueError("resume feature normalization mismatch")
    if checkpoint.get("resume_state") and checkpoint.get("manifest_sha256") is None:
        raise ValueError("exact resume requires a manifest hash")


def train(args) -> dict:
    correction_cap = validate_correction_cap(args.correction_cap)
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or args.patience < 0:
        raise ValueError("epochs, batch size and CPU threads must be positive; patience must be nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be nonnegative and finite")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    data = load_dataset(args.data)
    output = Path(args.output)
    if not args.overwrite and any((output / name).exists() for name in ("best.pt", "latest.pt", "model.json", "metrics.json", "config.json", "history.csv")):
        raise FileExistsError("training output already exists; select a new directory or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    feature_transform = getattr(args, "feature_transform", "identity")
    model = SinglePointResidualNet(args.width, feature_transform).to(device)
    mean, scale = training_normalization(data, feature_transform)
    model.set_normalization(mean, scale)
    features = torch.as_tensor(data["features"], dtype=torch.float32, device=device)
    if not torch.isfinite(features).all():
        raise ValueError("features overflow float32; data requires an explicit larger-range representation")
    logp = torch.as_tensor(data["log_p_history"], dtype=torch.float64, device=device)
    targets = torch.as_tensor(data["log_correction"], dtype=torch.float64, device=device)
    partitions = [torch.as_tensor(indices, dtype=torch.long, device=device) for indices in data["partitions"]]
    training = partitions[0]
    loss_weights = torch.ones(len(features), dtype=torch.float64, device=device)
    if args.loss_weighting == "base-hazard":
        training_log_base = data["log_alpha_base"][data["partitions"][0]]
        weights = np.exp(training_log_base - np.max(training_log_base))
        weights /= weights.mean()
        loss_weights[training] = torch.as_tensor(weights, dtype=torch.float64, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=8, min_lr=1e-6)
    dataset_digest, manifest_digest = sha256(data["dataset_path"]), sha256(data["manifest_path"])
    contract = continuation_contract(args, device)
    start_epoch, best_epoch, best_score, best_state, best_resume_state = 0, 0, float("inf"), None, None
    continuation = {"kind": "new", "starting_epoch": 0}
    resume = getattr(args, "resume", None)
    if resume:
        resume_path = Path(resume)
        if resume_path.resolve().parent == output.resolve():
            raise ValueError("resume output must be a new directory, preserving the parent checkpoint")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_resume(checkpoint, model, data, dataset_digest, manifest_digest, feature_transform)
        model.load_state_dict(checkpoint["state_dict"])
        best_state = {name: value.detach().cpu().clone() for name, value in
                      checkpoint.get("best_state_dict", checkpoint["state_dict"]).items()}
        best_epoch = int(checkpoint["best_epoch"])
        current_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        initial_prediction = predict(model, features, logp, partitions[1])
        best_score = error_metrics(initial_prediction, data["log_correction"][data["partitions"][1]],
                                   data["log_alpha_base"][data["partitions"][1]])["log_mae"]
        model.load_state_dict(current_state)
        if checkpoint.get("resume_state"):
            state = checkpoint["resume_state"]
            if state.get("version") != 1 or state.get("contract") != contract:
                raise ValueError("exact resume training contract mismatch (batch/loss/weight decay/device/version/threads)")
            start_epoch = int(state["epoch"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best_resume_state = copy.deepcopy(checkpoint["best_resume_state"])
            if int(best_resume_state["epoch"]) != best_epoch or best_epoch > start_epoch:
                raise ValueError("resume best/current epoch metadata is inconsistent")
            # Evaluation above consumes no random values; restore only after
            # constructing all state, so the next shuffled minibatch is exact.
            restore_random_state(state["random"])
            kind = "exact"
        else:
            start_epoch = best_epoch
            kind = "warm_start"
        continuation = {"kind": kind, "starting_epoch": start_epoch,
                        "parent_checkpoint": str(resume_path.resolve()), "parent_sha256": sha256(resume_path),
                        "parent_best_epoch": best_epoch, "initial_validation_log_mae": best_score,
                        "optimizer_restored": kind == "exact", "scheduler_restored": kind == "exact",
                        "random_state_restored": kind == "exact", "epochs_argument": "additional epochs",
                        "initial_learning_rate": optimizer.param_groups[0]["lr"]}
    config = {**vars(args), "device_resolved": str(device), "torch_version": str(torch.__version__),
              "architecture": model.configuration(), "feature_order": list(FEATURE_NAMES),
              "feature_mean": model.feature_mean.cpu().tolist(), "feature_scale": model.feature_scale.cpu().tolist(),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "dataset_sha256": dataset_digest, "manifest_sha256": manifest_digest,
              "source_files": source_file_metadata(data["manifest"]),
              "sigma": data["manifest"]["sigma"], "length_scale": data["manifest"]["length_scale"],
              "split_counts": [int(len(indices)) for indices in partitions],
              "split_view_counts": [int(len(np.unique(data["view_id"][indices]))) for indices in data["partitions"]],
              "split_ray_counts": [int(len(np.unique(data["ray_id"][indices]))) for indices in data["partitions"]],
              "loss": "SmoothL1(log_hazard_correction error, beta=0.25)",
              "correction_cap_applied_to_training": False,
              "checkpoint_selection": "uncapped validation log MAE",
              "domain": domain_metadata(data), "continuation": continuation}
    write_json(output / "config.json", config)
    def training_state(epoch):
        return {"version": 1, "epoch": epoch, "contract": contract,
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "scheduler": copy.deepcopy(scheduler.state_dict()), "random": capture_random_state()}

    def checkpoint_payload(current_state, resume_state):
        return {"schema": MODEL_SCHEMA, "model_config": model.configuration(), "state_dict": current_state,
                "feature_transform": feature_transform, "feature_order": list(FEATURE_NAMES),
                "sigma": config["sigma"], "length_scale": config["length_scale"],
                "dataset_sha256": dataset_digest, "manifest_sha256": manifest_digest,
                "source_files": config["source_files"], "best_epoch": best_epoch,
                "validation_log_mae": best_score, "best_state_dict": best_state,
                "resume_state": resume_state, "best_resume_state": best_resume_state,
                "continuation": continuation,
                "deployment": {"correction_cap": correction_cap, "stage": "inference_only"}}

    if best_state is not None:
        if best_resume_state is None:
            best_resume_state = training_state(start_epoch)
        save_checkpoint(output / "best.pt", checkpoint_payload(best_state, best_resume_state))
    history = []
    started = time.perf_counter()
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_loss", "validation_log_mae", "validation_log_p95", "learning_rate", "elapsed_seconds"))
        writer.writeheader()
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            model.train()
            order = training[torch.randperm(len(training), device=device)]
            loss_sum = 0.0
            for start in range(0, len(order), args.batch_size):
                selected = order[start:start+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                estimated = model(features[selected], logp[selected])
                losses = F.smooth_l1_loss(estimated, targets[selected], beta=.25, reduction="none")
                loss = (losses*loss_weights[selected]).mean()
                if not torch.isfinite(loss):
                    raise ArithmeticError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(selected)
            validation = data["partitions"][1]
            estimated = predict(model, features, logp, partitions[1])
            if not np.isfinite(estimated).all():
                raise ArithmeticError("nonfinite validation prediction")
            metrics = error_metrics(estimated, data["log_correction"][validation], data["log_alpha_base"][validation])
            score = metrics["log_mae"]
            scheduler.step(score)
            record = {"epoch": epoch, "train_loss": loss_sum / len(training),
                      "validation_log_mae": score, "validation_log_p95": metrics["log_p95"],
                      "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter()-started}
            history.append(record)
            writer.writerow(record)
            stream.flush()
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_resume_state = training_state(epoch)
                save_checkpoint(output / "best.pt", checkpoint_payload(best_state, best_resume_state))
            current_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            save_checkpoint(output / "latest.pt", checkpoint_payload(current_state, training_state(epoch)))
            if epoch == start_epoch + 1 or epoch % 10 == 0 or epoch == start_epoch + args.epochs:
                print(f"epoch {epoch}: train={record['train_loss']:.6g}, validation log MAE={score:.6g}", flush=True)
            if args.patience and epoch - best_epoch >= args.patience:
                break
    model.load_state_dict(best_state)
    report = {"best_epoch": best_epoch, "epochs_completed": len(history), "elapsed_seconds": time.perf_counter()-started,
              "starting_epoch": start_epoch, "final_epoch": history[-1]["epoch"], "continuation": continuation,
              "parameter_count": config["parameter_count"], "partitions": {},
              "deployment": {"correction_cap": correction_cap, "stage": "inference_only",
                             "targets_clipped": False, "checkpoint_selection_capped": False}}
    for name, selected_np, selected in zip(("train", "validation", "test"), data["partitions"], partitions):
        estimated = predict(model, features, logp, selected)
        if not np.isfinite(estimated).all():
            raise ArithmeticError(f"nonfinite {name} prediction")
        report["partitions"][name] = error_metrics(estimated, data["log_correction"][selected_np], data["log_alpha_base"][selected_np])
        report["partitions"][name]["baseline"] = error_metrics(np.zeros(len(selected_np)), data["log_correction"][selected_np], data["log_alpha_base"][selected_np])
        report["partitions"][name]["envelope_violation_count"] = int(np.sum(estimated > -data["log_p_history"][selected_np] + 1e-10))
        domain = config["domain"]
        out_of_range = (data["features"][selected_np] < np.asarray(domain["minimum_features"])) | (data["features"][selected_np] > np.asarray(domain["maximum_features"]))
        report["partitions"][name]["outside_training_feature_range_count"] = int(out_of_range.any(axis=1).sum())
        # Both reports use the ORIGINAL teacher labels. This exposes cap bias
        # separately from learning error and never feeds back into optimization.
        log_cap = np.inf if correction_cap is None else np.log(correction_cap)
        capped = np.minimum(estimated, log_cap)
        cap_report = error_metrics(capped, data["log_correction"][selected_np], data["log_alpha_base"][selected_np])
        cap_report["prediction_clipped_count"] = int(np.sum(estimated > log_cap))
        cap_report["prediction_clipped_fraction"] = float(np.mean(estimated > log_cap))
        cap_report["teacher_above_cap_count"] = int(np.sum(data["log_correction"][selected_np] > log_cap))
        report["partitions"][name]["render_capped"] = cap_report
    write_json(output / "metrics.json", report)
    write_json(output / "model.json", export_model(model, data, {"best_epoch": best_epoch, "dataset_sha256": config["dataset_sha256"],
                                                               "manifest_sha256": config["manifest_sha256"], "source_files": config["source_files"],
                                                               "seed": args.seed}, correction_cap))
    print(f"Saved {output / 'model.json'}; test log MAE={report['partitions']['test']['log_mae']:.6g}", flush=True)
    return report


def parse_correction_cap(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        return validate_correction_cap(float(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True, help="Native export directory containing manifest.json and samples.csv")
    result.add_argument("--output", required=True, help="Directory for checkpoint, JSON weights, metrics and training history")
    result.add_argument("--resume", help="Parent best.pt/latest.pt; legacy weights warm-start, full state resumes exactly; output must be a new directory")
    result.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    result.add_argument("--epochs", type=int, default=100, help="Epochs for this invocation; additional epochs when resuming")
    result.add_argument("--batch-size", type=int, default=4096)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--feature-transform", choices=("identity", "asinh_z"), default="identity",
                        help="Invertible asinh of the three standardized means avoids large near-birth inputs dominating normalization; correlations unchanged")
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--loss-weighting", choices=("uniform", "base-hazard"), default="uniform")
    result.add_argument("--patience", type=int, default=20, help="Validation early-stop patience; zero disables")
    result.add_argument("--seed", type=int, default=20261113)
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--correction-cap", type=parse_correction_cap, default=2.0,
                        help="Inference-only maximum correction multiplier C>=1 (default 2); 'none' disables. Does not change training or checkpoint selection.")
    result.add_argument("--overwrite", action="store_true")
    return result


if __name__ == "__main__":
    train(parser().parse_args())
