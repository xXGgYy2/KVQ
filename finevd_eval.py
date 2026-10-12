#!/usr/bin/env python3
"""Evaluate KVQ on FineVD using the MOS ``overall`` column.

Run from the KVQ repository root. The script writes:
  results/finevd/per_video.csv
  results/finevd/metrics.csv
  results/finevd/finevd_results.xlsx (when openpyxl is installed)

Only model outputs 0, 1, 2 and 3 are retained. A missing/non-numeric output is
written as NaN and is excluded from that branch's correlation metrics.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

import vqa.datasets as kvq_datasets
import vqa.models as kvq_models


VIDEO_EXTENSIONS = {
    ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".y4m"
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KVQ on FineVD.")
    parser.add_argument(
        "--video-dir",
        default="/mnt/data/local-disk2/xyy/datasets/FineVD/videos_all_videos",
    )
    parser.add_argument(
        "--mos-csv",
        default="/mnt/data/local-disk2/xyy/datasets/FineVD/FineVD/MOS_train.csv",
    )
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--checkpoint", default="weights/KVQ.pth")
    parser.add_argument("--output-dir", default="results/finevd")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip videos without an overall MOS instead of stopping.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N videos (0 means all; useful for a smoke test).",
    )
    return parser.parse_args()


def video_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"FineVD video directory does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No video files found under: {root}")
    return files


def load_mos(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"MOS CSV has no header: {csv_path}")
        columns = {column.strip().lower(): column for column in reader.fieldnames}
        name_column = columns.get("video_name") or columns.get("filename") or columns.get("video")
        overall_column = columns.get("overall")
        if not name_column or not overall_column:
            raise ValueError(
                f"Expected video_name and overall columns in {csv_path}; "
                f"found {reader.fieldnames}"
            )
        result: dict[str, float] = {}
        for row in reader:
            name = (row.get(name_column) or "").strip()
            value = (row.get(overall_column) or "").strip()
            if not name or not value:
                continue
            try:
                result[Path(name).name.lower()] = float(value)
            except ValueError:
                print(f"[WARN] ignoring non-numeric overall MOS: {name!r}={value!r}")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_annotation(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            # FusionDataset's annotation format is filename, ignored_1, ignored_2, MOS.
            handle.write(f"{row['filename']}, -1, -1, {float(row['mos']):.12g}\n")


def make_dataset(config: dict[str, Any], root: Path, annotation: Path):
    data_config = config.get("data", {})
    template = next(
        (value for key, value in data_config.items() if key.startswith("val")),
        next(iter(data_config.values()), None),
    )
    if template is None:
        raise ValueError("The config has no data section.")
    options = dict(template.get("args", {}))
    options["phase"] = "test"
    options["data_prefix"] = str(root.resolve())
    options["anno_file"] = str(annotation.resolve())
    dataset_class = getattr(kvq_datasets, template["type"])
    try:
        dataset = dataset_class(options)
    except TypeError as positional_error:
        try:
            dataset = dataset_class(**options)
        except TypeError:
            raise positional_error
    return dataset, options.get("sample_types", {})


def make_model(config: dict[str, Any], checkpoint: Path, device: torch.device):
    model_config = config["model"]
    model_class = getattr(kvq_models, model_config["type"])
    model_args = model_config.get("args", {})
    try:
        model = model_class(**model_args)
    except TypeError as keyword_error:
        try:
            model = model_class(model_args)
        except TypeError:
            raise keyword_error

    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch
        saved = torch.load(checkpoint, map_location="cpu")
    state_dict = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        unwrapped = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(unwrapped, strict=True)
    return model.to(device).eval()


def model_call(model: torch.nn.Module, inputs: dict[str, torch.Tensor]):
    signature = inspect.signature(model.forward)
    kwargs = {}
    for name, value in {
        "cross_flag": False,
        "epoch": 0,
        "inference": True,
    }.items():
        if name in signature.parameters:
            kwargs[name] = value
    return model(inputs, **kwargs)


def batch_filename(batch: Any, fallback: str) -> str:
    """Use a dataset-provided name when available; otherwise annotation order."""
    if not isinstance(batch, dict):
        return fallback
    for key in ("name", "filename", "video_name", "video_names"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if torch.is_tensor(value) and value.numel():
            return str(value.flatten()[0].item())
        if isinstance(value, str):
            return value
    return fallback


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return math.nan
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value, dtype=np.float64).mean())
    if isinstance(value, (float, int, np.number)):
        return float(value)
    return math.nan


def output_0_to_3(raw: Any) -> list[float]:
    """Return the first four numeric model outputs in branch order.

    The released KVQ checkpoint commonly returns a five-item tuple where raw
    item 2 is ``None`` and raw items 0, 1, 3 and 4 are the four scalar branches.
    Those four values are exposed as output0..output3 in the result table.
    """
    if isinstance(raw, dict):
        values = []
        for key in sorted(raw, key=str):
            value = scalar(raw[key])
            if np.isfinite(value):
                values.append(value)
            if len(values) == 4:
                break
        return values + [math.nan] * (4 - len(values))
    elif isinstance(raw, (tuple, list)):
        values = []
        for value in raw:
            value = scalar(value)
            if np.isfinite(value):
                values.append(value)
            if len(values) == 4:
                break
        return values + [math.nan] * (4 - len(values))
    else:
        value = scalar(raw)
        return [value, math.nan, math.nan, math.nan]


def metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index in range(4):
        prediction_key = f"output{index}"
        pairs = [
            (float(row[prediction_key]), float(row["mos"]))
            for row in rows
            if np.isfinite(float(row[prediction_key])) and np.isfinite(float(row["mos"]))
        ]
        if len(pairs) < 2:
            srcc = krcc = plcc = math.nan
        else:
            prediction = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
            mos = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            srcc = float(spearmanr(prediction, mos).statistic) if np.ptp(prediction) and np.ptp(mos) else math.nan
            krcc = float(kendalltau(prediction, mos).statistic) if np.ptp(prediction) and np.ptp(mos) else math.nan
            plcc = float(pearsonr(prediction, mos).statistic) if np.ptp(prediction) and np.ptp(mos) else math.nan
        result.append({"branch": prediction_key, "n": len(pairs), "PLCC": plcc, "SRCC": srcc, "KRCC": krcc})
    return result


def write_xlsx(path: Path, rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        print("[WARN] openpyxl is not installed; CSV files are complete.")
        return
    workbook = Workbook()
    video_sheet = workbook.active
    video_sheet.title = "per_video"
    video_fields = ["filename", "mos", "output0", "output1", "output2", "output3"]
    video_sheet.append(video_fields)
    for row in rows:
        video_sheet.append([row.get(field, "") for field in video_fields])
    metric_sheet = workbook.create_sheet("metrics")
    metric_fields = ["branch", "n", "PLCC", "SRCC", "KRCC"]
    metric_sheet.append(metric_fields)
    for row in metric_rows:
        metric_sheet.append([row.get(field, "") for field in metric_fields])
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = cell.font.copy(bold=True)
    workbook.save(path)


def main() -> int:
    cli = args()
    root = Path(cli.video_dir)
    mos_path = Path(cli.mos_csv)
    config_path = Path(cli.config)
    checkpoint = Path(cli.checkpoint)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    videos = video_files(root)
    mos = load_mos(mos_path)
    rows: list[dict[str, Any]] = []
    missing = []
    for video in videos:
        score = mos.get(video.name.lower())
        row = {"filename": video.relative_to(root).as_posix(), "mos": score}
        if score is None:
            missing.append(row["filename"])
        else:
            rows.append(row)
    print(f"[FineVD] videos={len(videos)}, labels={len(rows)}, missing={len(missing)}")
    if missing:
        print(f"[FineVD] first missing labels: {missing[:10]}")
        if not cli.allow_missing:
            raise RuntimeError(
                "Some video names were not found in MOS_train.csv. "
                "Check the filename spelling or use --allow-missing."
            )
    if cli.limit > 0:
        rows = rows[: cli.limit]
        print(f"[FineVD] smoke-test limit={len(rows)}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    annotation = output / "finevd_kvq_labels.txt"
    make_annotation(annotation, rows)
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    model = make_model(config, checkpoint, device)
    dataset, sample_types = make_dataset(config, root, annotation)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
    )

    completed: list[dict[str, Any]] = []
    for index, batch in enumerate(tqdm(loader, desc="FineVD")):
        if index >= len(rows):
            break
        video_batch = batch.get("video", batch) if isinstance(batch, dict) else batch
        inputs = {
            name: video_batch[name].to(device, non_blocking=True)
            for name in sample_types
            if isinstance(video_batch, dict)
            and name in video_batch
            and torch.is_tensor(video_batch[name])
        }
        if not inputs:
            keys = list(video_batch) if isinstance(video_batch, dict) else type(video_batch)
            raise RuntimeError(f"No configured video tensors found in batch; keys={keys}")
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(not cli.no_amp and device.type == "cuda")):
                raw = model_call(model, inputs)
        values = output_0_to_3(raw)
        actual_filename = batch_filename(batch, rows[index]["filename"])
        actual_mos = mos.get(Path(actual_filename).name.lower(), rows[index]["mos"])
        completed.append({"filename": actual_filename, "mos": actual_mos, **{
            f"output{branch}": values[branch] for branch in range(4)
        }})
        del batch, inputs, raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = ["filename", "mos", "output0", "output1", "output2", "output3"]
    metric_rows = metrics(completed)
    write_csv(output / "per_video.csv", completed, fields)
    write_csv(output / "metrics.csv", metric_rows, ["branch", "n", "PLCC", "SRCC", "KRCC"])
    write_xlsx(output / "finevd_results.xlsx", completed, metric_rows)
    print(f"Results written to: {output.resolve()}")
    print(f"Per-video table: {output / 'per_video.csv'}")
    print(f"Metric table: {output / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
