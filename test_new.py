#!/usr/bin/env python3
"""Evaluate KVQ on every test-/val- dataset declared in one YAML config."""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

import vqa.datasets as datasets
import vqa.models as models


SAMPLE_TYPES = ("resize", "fragments", "crop", "arp_resize", "arp_fragments")
RESULTS_ROOT = Path("/mnt/data/local-disk2/xyy/KVQ/results")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args():
    parser = argparse.ArgumentParser(description="Test KVQ from a YAML experiment config")
    parser.add_argument("-c", "--config", default="configs/offical-mutil-test.yaml")
    parser.add_argument("--dataset", action="append",
                        help="Only run this data key; repeat for more than one")
    parser.add_argument("--checkpoint", help="Override config test_load_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    if not value:
        raise ValueError("Experiment/dataset name becomes empty after sanitizing")
    return value


def read_mos_file(path, video_root):
    """Read `filename MOS`, `filename,MOS`, or `filename,-1,-1,MOS`."""
    path, root = Path(path), Path(video_root)
    if not path.is_file():
        raise FileNotFoundError("MOS file does not exist: {}".format(path))
    rows, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "," in line:
            fields = [field.strip() for field in
                      next(csv.reader([line], skipinitialspace=True))]
            if len(fields) not in (2, 4):
                raise ValueError(
                    "{}:{}: expected filename,MOS or filename,-1,-1,MOS; "
                    "parsed {} fields: {!r}".format(
                        path, line_number, len(fields), fields
                    )
                )
            filename, mos_text = fields[0], fields[-1]
        else:
            fields = line.rsplit(None, 1)
            if len(fields) != 2:
                raise ValueError("{}:{}: expected filename MOS".format(path, line_number))
            filename, mos_text = fields[0].strip(), fields[1].strip()
        if not filename:
            raise ValueError("{}:{}: empty filename".format(path, line_number))
        try:
            mos = float(mos_text)
        except ValueError:
            raise ValueError("{}:{}: invalid MOS {!r}".format(path, line_number, mos_text))
        if not math.isfinite(mos):
            raise ValueError("{}:{}: MOS must be finite".format(path, line_number))
        video = Path(filename)
        if not video.is_absolute():
            video = root / video
        video = video.resolve()
        if not video.is_file():
            raise FileNotFoundError("{}:{}: missing video {}".format(path, line_number, video))
        identity = str(video)
        if identity in seen:
            raise ValueError("Duplicate video in {}: {}".format(path, video))
        seen.add(identity)
        rows.append({"filename": identity, "mos": mos})
    if len(rows) < 2:
        raise ValueError("{} needs at least two labeled videos".format(path))
    return rows


def write_dataset_annotation(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write("{},-1,-1,{:.12g}\n".format(row["filename"], row["mos"]))


def make_dataset(spec, rows, annotation_path):
    options = dict(spec["args"])
    options["phase"] = "test"
    options["anno_file"] = str(annotation_path.resolve())
    options["data_prefix"] = "/"
    options["get_close_matches"] = False
    dataset = getattr(datasets, spec["type"])(options)
    if len(dataset) != len(rows):
        raise RuntimeError("FusionDataset resolved {} videos, expected {}".format(len(dataset), len(rows)))
    infos = getattr(dataset, "video_infos", None)
    if infos is not None:
        for index, (info, row) in enumerate(zip(infos, rows)):
            actual = str(Path(info["filename"]).resolve())
            if actual != row["filename"] or not math.isclose(
                    float(info["label"]), row["mos"], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("Dataset path/MOS mismatch at index {}: {}".format(index, info))
    return dataset


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_model(config, checkpoint, device):
    model_config = config["model"]
    model = getattr(models, model_config["type"])(**model_config.get("args", {}))
    saved = torch_load(checkpoint)
    state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    if not isinstance(state, dict):
        raise TypeError("Unsupported checkpoint format: {}".format(checkpoint))
    state = {key[7:] if key.startswith("module.") else key: value
             for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    print("Loaded checkpoint: {} ({} tensors)".format(Path(checkpoint).resolve(), len(state)))
    return model.to(device).eval()


def reshape_clips(data, key, tensor):
    batch, channels, frames, height, width = tensor.shape
    clips = int(torch.as_tensor(data["num_clips"][key]).reshape(-1)[0].item())
    if clips < 1 or frames % clips:
        raise ValueError("{}: {} frames cannot be divided into {} clips".format(key, frames, clips))
    return (tensor.reshape(batch, channels, clips, frames // clips, height, width)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(batch * clips, channels, frames // clips, height, width))


def extract_filename(data, fallback):
    for key in ("name", "filename", "video_name"):
        value = data.get(key)
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if isinstance(value, str):
            return value
    return fallback


def compute_metrics(predictions, targets):
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Predictions or MOS contain non-finite values")
    if prediction.std() <= 1e-12 or target.std() <= 1e-12:
        raise ValueError("Predictions or MOS are constant; correlations are undefined")
    calibrated = ((prediction - prediction.mean()) / prediction.std()
                  * target.std() + target.mean())
    return {
        "srcc": float(spearmanr(target, prediction).statistic),
        "plcc": float(pearsonr(target, prediction).statistic),
        "krcc": float(kendalltau(target, prediction).statistic),
        "rmse_raw": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "rmse_calibrated": float(np.sqrt(np.mean((target - calibrated) ** 2))),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "count": int(len(target)),
    }


def evaluate(dataset_name, dataset, rows, model, device, workers, amp, output_dir):
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=False,
                        generator=generator)
    predictions, targets, records = [], [], []
    for index, data in enumerate(tqdm(loader, desc=dataset_name)):
        video = {}
        for key in SAMPLE_TYPES:
            if key in data:
                video[key] = reshape_clips(data, key, data[key].to(device, non_blocking=True))
        if not video:
            raise RuntimeError("{}: no configured video tensor in batch".format(dataset_name))
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                output = model(video)
        scores = output[0] if isinstance(output, (tuple, list)) else output
        prediction = float(scores.detach().float().mean().cpu().item())
        target = float(data["gt_label"].reshape(-1)[0].item())
        predictions.append(prediction)
        targets.append(target)
        records.append({
            "index": index,
            "filename": extract_filename(data, rows[index]["filename"]),
            "mos": target,
            "prediction": prediction,
        })
    metrics = compute_metrics(predictions, targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_video.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "filename", "mos", "prediction"))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def format_metrics(metrics):
    return " ".join("{}={:.4f}".format(key.upper(), value)
                    for key, value in metrics.items())


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    experiment = safe_name(config.get("name") or config_path.stem)
    experiment_dir = RESULTS_ROOT / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "test_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)

    print("Config: {}".format(config_path.resolve()))
    print("Experiment results: {}".format(experiment_dir))
    print("Log: {}".format(log_path))
    if bool(config.get("ddp", False)):
        raise ValueError("test_new.py is single-GPU; set ddp: 0")
    workers = args.num_workers if args.num_workers is not None else int(config.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    checkpoint = args.checkpoint or config.get("test_load_path")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_specs = config.get("data", {})
    selected = args.dataset or [key for key in data_specs if key.startswith(("test-", "val-"))]
    if not selected:
        raise ValueError("Config contains no test-* or val-* datasets")
    unknown = set(selected) - set(data_specs)
    if unknown:
        raise KeyError("Unknown dataset keys: {}".format(sorted(unknown)))

    print("Datasets: {}".format(selected))
    model = make_model(config, checkpoint, device)
    summary = {}
    for key in selected:
        spec = data_specs[key]
        options = spec.get("args", {})
        mos_file = options.get("anno_file")
        video_root = options.get("data_prefix")
        if not mos_file or not video_root:
            raise ValueError("{} requires args.anno_file and args.data_prefix".format(key))
        rows = read_mos_file(mos_file, video_root)
        dataset_dir = experiment_dir / safe_name(key)
        annotation = dataset_dir / "resolved_labels.txt"
        write_dataset_annotation(annotation, rows)
        dataset = make_dataset(spec, rows, annotation)
        metrics = evaluate(key, dataset, rows, model, device, workers,
                           not args.no_amp, dataset_dir)
        summary[key] = metrics
        print("{}: {}".format(key, format_metrics(metrics)), flush=True)
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Finished. Summary: {}".format(experiment_dir / "summary.json"))


if __name__ == "__main__":
    main()
