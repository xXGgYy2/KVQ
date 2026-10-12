"""Multi-GPU DDP fine-tuning for KVQ on fixed VDPVE or mixed-data splits.

Launch this file with torchrun. ``--batch-size`` is the per-GPU batch size.
PLCC and ranking losses are calculated over the full batch from every GPU.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import uuid
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from scipy.stats import kendalltau, pearsonr, spearmanr


TRAINER_NAME = "kvq_vdpve_ddp_v3"
SAMPLE_TYPES = ("resize", "fragments", "crop", "arp_resize", "arp_fragments")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def gather_rng(world_size):
    states = [None] * world_size
    dist.all_gather_object(states, capture_rng())
    return states


@contextmanager
def evaluation_seed(seed):
    previous = capture_rng()
    seed_everything(seed)
    try:
        yield
    finally:
        restore_rng(previous)


class SeededDataset(Dataset):
    """Give each video a repeatable sampling seed, independent of worker order."""

    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Dataset workers use CPU RNGs only. Do not initialize CUDA in a worker.
        previous = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        seed = (self.seed + index) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        try:
            data = self.dataset[index]
            if isinstance(data, dict):
                # Ensure exported predictions remain traceable even when the
                # underlying dataset does not expose a filename in __getitem__.
                infos = getattr(self.dataset, "video_infos", None)
                if infos is not None and isinstance(infos[index], dict):
                    data = dict(data)
                    data["audit_filename"] = str(infos[index].get("filename", ""))
            return data
        finally:
            random.setstate(previous[0])
            np.random.set_state(previous[1])
            torch.set_rng_state(previous[2])


def annotation_content_id(value, video_path):
    """Ignore unknown IDs; recover only the known VDPVE naming convention."""
    if value not in ("", "-1"):
        return value
    path = Path(video_path)
    if "vdpve" in {part.lower() for part in path.parts}:
        match = re.fullmatch(r"([A-Za-z]\d{4})_\d+", path.stem)
        if match:
            return match.group(1).upper()
    return None


def validate_annotation_file(anno_file, video_root):
    anno_path = Path(anno_file)
    root = Path(video_root)
    if not anno_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {anno_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")

    rows = []
    seen = set()
    missing = []
    with anno_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, fields in enumerate(csv.reader(handle), 1):
            if not fields:
                continue
            fields = [field.strip() for field in fields]
            if len(fields) != 4:
                raise ValueError(
                    f"{anno_path}:{line_number} must contain four comma-separated fields"
                )
            try:
                mos = float(fields[-1])
            except ValueError as error:
                raise ValueError(
                    f"{anno_path}:{line_number} has an invalid MOS: {fields[-1]!r}"
                ) from error
            if not math.isfinite(mos):
                raise ValueError(f"{anno_path}:{line_number}: MOS must be finite")
            video_path = Path(fields[0])
            if not video_path.is_absolute():
                video_path = root / video_path
            video_path = video_path.resolve()
            if str(video_path) in seen:
                raise ValueError(f"Duplicate video in {anno_path}: {video_path}")
            seen.add(str(video_path))
            if not video_path.is_file() and len(missing) < 10:
                missing.append(str(video_path))
            rows.append({"filename": str(video_path),
                         "content_id": annotation_content_id(fields[1], video_path), "mos": mos})

    if not rows:
        raise ValueError(f"Annotation file is empty: {anno_path}")
    if missing:
        sample = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Videos referenced by {anno_path} were not found (first {len(missing)}):\n  {sample}"
        )
    return rows


def overlapping_rows(left, right):
    videos = {r["filename"] for r in left} & {r["filename"] for r in right}
    # -1 is missing metadata, never evidence that two videos share content.
    left_ids = {r["content_id"] for r in left if r["content_id"] not in (None, "", "-1")}
    right_ids = {r["content_id"] for r in right if r["content_id"] not in (None, "", "-1")}
    return videos, left_ids & right_ids


def check_split_overlap(split_rows):
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        videos, contents = overlapping_rows(split_rows[left], split_rows[right])
        if videos or contents:
            raise ValueError(
                f"Split leakage between {left}/{right}: {len(videos)} shared videos, "
                f"{len(contents)} shared content IDs. "
                f"Examples: videos={sorted(videos)[:5]}, contents={sorted(contents)[:5]}. "
                "this script does not re-split or silently change annotations."
            )


def check_prior_exposure(split_rows, previous_rows, source):
    for name in ("val", "test"):
        videos, contents = overlapping_rows(split_rows[name], previous_rows)
        if videos or contents:
            raise ValueError(
                f"Prior-checkpoint leakage: new {name} overlaps {source}: "
                f"{len(videos)} videos, {len(contents)} content IDs. "
                f"Examples: videos={sorted(videos)[:5]}, contents={sorted(contents)[:5]}. "
                "Use held-out videos/content for a fair comparison of initial checkpoints."
            )


def validate_dataset_metadata(dataset, rows, name):
    """Check that FusionDataset actually resolved the exact paths and MOS."""
    infos = getattr(dataset, "video_infos", None)
    if infos is None or len(infos) != len(rows):
        raise ValueError(f"{name}: FusionDataset must expose matching video_infos for path/MOS audit")
    for index, (info, row) in enumerate(zip(infos, rows)):
        if not isinstance(info, dict) or "filename" not in info or "label" not in info:
            raise ValueError(f"{name}[{index}]: expected video_infos with filename and label")
        path = str(Path(info["filename"]).resolve())
        mos = float(info["label"])
        if path != row["filename"] or not math.isclose(mos, row["mos"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{name}[{index}]: FusionDataset path/MOS differs from annotation: {info}")


def dataset_args(template, phase, anno_file, video_root):
    args = copy.deepcopy(template)
    args["phase"] = phase
    args["anno_file"] = anno_file
    args["data_prefix"] = video_root
    args["get_close_matches"] = False
    return args


def build_datasets(config, cli_args):
    import vqa.datasets as datasets
    if "train" not in config.get("data", {}):
        raise KeyError("config data.train is required as the FusionDataset template")
    dataset_type = config["data"]["train"]["type"]
    template = config["data"]["train"]["args"]
    dataset_class = getattr(datasets, dataset_type)
    split_specs = {
        "train": ("train", cli_args.train_anno),
        "val": ("test", cli_args.val_anno),
        "test": ("test", cli_args.test_anno),
    }
    return {
        name: dataset_class(dataset_args(template, phase, anno, cli_args.video_root))
        for name, (phase, anno) in split_specs.items()
    }


def rank_loss(prediction, target):
    batch_size = prediction.shape[0]
    if batch_size < 2:
        raise ValueError("Ranking/PLCC training requires batch_size >= 2")
    pairwise = torch.relu(
        (prediction - prediction.t()) * torch.sign(target.t() - target)
    )
    return pairwise.sum() / (batch_size * (batch_size - 1) * (1 + pairwise.max()))


def plcc_loss(prediction, target):
    pred_std, pred_mean = torch.std_mean(prediction, unbiased=False)
    target_std, target_mean = torch.std_mean(target, unbiased=False)
    prediction = (prediction - pred_mean) / (pred_std + 1e-8)
    target = (target - target_mean) / (target_std + 1e-8)
    loss_0 = torch.nn.functional.mse_loss(prediction, target) / 4
    rho = torch.mean(prediction * target)
    loss_1 = torch.nn.functional.mse_loss(rho * prediction, target) / 4
    return (loss_0 + loss_1) / 2


def lpc_loss(scores, patch_scores):
    batch_size = scores.shape[0]
    scores = scores.reshape(batch_size, -1)
    patch_scores = patch_scores.reshape(batch_size, -1)
    scores = (scores - scores.mean(1, keepdim=True)) / (
        scores.std(1, keepdim=True) + 1e-8
    )
    patch_scores = (patch_scores - patch_scores.mean(1, keepdim=True)) / (
        patch_scores.std(1, keepdim=True) + 1e-8
    )
    return 1 - torch.nn.functional.cosine_similarity(scores, patch_scores).mean()


def make_patch_video(fragments, spatial_h, spatial_w, time_grid):
    batch, channels, frames, height, width = fragments.shape
    if frames % time_grid or height % spatial_h or width % spatial_w:
        raise ValueError(
            "LPC requires fragments T/H/W to be divisible by "
            f"{time_grid}/{spatial_h}/{spatial_w}; got {frames}/{height}/{width}"
        )
    return (
        fragments.reshape(
            batch,
            channels,
            time_grid,
            frames // time_grid,
            spatial_h,
            height // spatial_h,
            spatial_w,
            width // spatial_w,
        )
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(
            batch * time_grid * spatial_h * spatial_w,
            channels,
            frames // time_grid,
            height // spatial_h,
            width // spatial_w,
        )
    )


def reshape_patch_scores(patch_scores, local_scores, batch, spatial_h, spatial_w, time_grid):
    local_t, local_h, local_w = local_scores.shape[-3:]
    if local_t % time_grid or local_h % spatial_h or local_w % spatial_w:
        raise ValueError(
            "Model local-score shape is incompatible with the configured LPC grids: "
            f"{local_t}/{local_h}/{local_w}"
        )
    return (
        patch_scores.reshape(
            batch,
            time_grid,
            spatial_h,
            spatial_w,
            local_t // time_grid,
            local_h // spatial_h,
            local_w // spatial_w,
        )
        .permute(0, 1, 4, 2, 5, 3, 6)
        .reshape(batch, local_t, local_h, local_w)
    )


def extract_video(data, device):
    return {
        key: data[key].to(device, non_blocking=True)
        for key in SAMPLE_TYPES
        if key in data
    }


def load_initial_weights(model, checkpoint_path, device):
    checkpoint = torch_load(checkpoint_path, "cpu")
    source = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(source, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    source = {key.removeprefix("module."): value for key, value in source.items()}
    # This trainer fine-tunes a complete KVQ model, not a partial Swin backbone.
    model.load_state_dict(source, strict=True)
    print(f"Loaded complete KVQ checkpoint: {checkpoint_path} ({len(source)} tensors)")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_value, value in zip(ema_model.parameters(), model.parameters()):
        ema_value.mul_(decay).add_(value.detach(), alpha=1 - decay)
    for ema_value, value in zip(ema_model.buffers(), model.buffers()):
        ema_value.copy_(value.detach())


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def reshape_validation_clips(data, key, tensor):
    batch, channels, frames, height, width = tensor.shape
    num_clips_value = data["num_clips"][key]
    num_clips = int(torch.as_tensor(num_clips_value).reshape(-1)[0].item())
    if num_clips < 1 or frames % num_clips:
        raise ValueError(f"{key}: {frames} frames cannot be split into {num_clips} clips")
    return (
        tensor.reshape(
            batch, channels, num_clips, frames // num_clips, height, width
        )
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * num_clips, channels, frames // num_clips, height, width)
    )


@torch.no_grad()
def evaluate(loader, model, device, description, prediction_path=None):
    model.eval()
    predictions = []
    targets = []
    records = []
    if loader.generator is not None:
        loader.generator.manual_seed(loader.dataset.seed)
    for data in tqdm(loader, desc=description):
        video = {}
        for key in SAMPLE_TYPES:
            if key in data:
                tensor = data[key].to(device, non_blocking=True)
                video[key] = reshape_validation_clips(data, key, tensor)
        output = model(video)
        scores = output[0] if isinstance(output, (tuple, list)) else output
        predictions.append(float(scores.float().mean().cpu()))
        targets.append(float(data["gt_label"].item()))
        filename = data.get("audit_filename", data.get("name", data.get("filename", "")))
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        records.append({"index": len(records), "filename": str(filename),
                        "mos": targets[-1], "prediction": predictions[-1]})

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if len(target) < 2:
        raise ValueError("Evaluation requires at least two videos")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError(f"{description}: non-finite predictions or labels")
    if target.std() <= 1e-12:
        raise ValueError(f"{description}: MOS labels are constant")
    if prediction_path is not None:
        write_json(prediction_path, records)
    prediction_std = prediction.std()
    constant = prediction_std <= 1e-12
    calibrated = (np.full_like(prediction, target.mean()) if constant else
                  (prediction - prediction.mean()) / prediction_std * target.std() + target.mean())
    return {
        "srcc": None if constant else float(spearmanr(target, prediction).statistic),
        "plcc": None if constant else float(pearsonr(target, prediction).statistic),
        "krcc": None if constant else float(kendalltau(target, prediction).statistic),
        "rmse_raw": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "rmse_calibrated": float(np.sqrt(np.mean((target - calibrated) ** 2))),
        "prediction_std": float(prediction_std),
        "prediction_mean": float(prediction.mean()),
        "count": len(target),
    }


def format_metrics(metrics):
    return " ".join(f"{key.upper()}={value:.4f}" if value is not None else
                    f"{key.upper()}=undefined" for key, value in metrics.items())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def metric_score(metrics):
    values = [metrics.get("srcc"), metrics.get("plcc")]
    return sum(values) if all(v is not None and math.isfinite(v) for v in values) else -float("inf")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune KVQ on fixed four-column video annotations with DDP")
    parser.add_argument("-c", "--config", default="configs/kvq.yaml")
    parser.add_argument(
        "--train-anno",
        default="/mnt/data/local-disk2/xyy/FAST-VQA-and-FasterVQA/annotations/vdpve/train.txt",
    )
    parser.add_argument(
        "--val-anno",
        default="/mnt/data/local-disk2/xyy/FAST-VQA-and-FasterVQA/annotations/vdpve/val.txt",
    )
    parser.add_argument(
        "--test-anno",
        default="/mnt/data/local-disk2/xyy/FAST-VQA-and-FasterVQA/annotations/vdpve/test.txt",
    )
    parser.add_argument(
        "--video-root", default="/mnt/data/local-disk2/xyy/datasets/VDPVE/train"
    )
    parser.add_argument("--prior-exposure-anno", action="append", default=[],
                        help="Previous checkpoint's train/selection-val annotations; reject new val/test overlap. Repeatable.")
    parser.add_argument("--prior-video-root",
                        default="/mnt/data/local-disk2/xyy/datasets/VDPVE/train")
    parser.add_argument("--output-dir", default="results/KVQ_VDPVE_DDP_v3")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size on each GPU (defaults to config batch_size)",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Head learning rate; deliberately lower than the pretraining config",
    )
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume", help="Resume a checkpoint made by this DDP script")
    parser.add_argument(
        "--init-checkpoint",
        default="weights/KVQ.pth",
        help="Initial complete KVQ checkpoint; strict loading (ignored with --resume)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpc-weight", type=float, default=0.02)
    parser.add_argument("--lpc-time-grid", type=int, default=16)
    parser.add_argument(
        "--eval-both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate both raw and EMA weights at each validation point",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Require two consecutive successful DDP updates; save no weights")
    parser.add_argument("--smoke-max-batches", type=int, default=20,
                        help="Smoke test attempt limit, including skipped AMP updates (default: 20)")
    parser.add_argument("--static-graph", action=argparse.BooleanOptionalAction, default=True,
                        help="Support repeated activation checkpointing in KVQ's fixed training graph")
    return parser.parse_args()


def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Use torchrun to launch this script")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=180))
    return local_rank, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_train_loader(dataset, args, rank, world_size, epoch):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    sampler.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank * 100000 + epoch)
    loader = DataLoader(
        SeededDataset(dataset, args.seed + epoch * 100000),
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader


def make_eval_loader(dataset, args, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SeededDataset(dataset, seed),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def gather_global_batch(local_tensor, rank, world_size):
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    with torch.no_grad():
        dist.all_gather(gathered, local_tensor.detach())
    # Remote predictions are constants on this rank. The local slot retains its
    # graph; scaling the global loss by world_size compensates for DDP averaging.
    gathered[rank] = local_tensor
    return torch.cat(gathered, dim=0)


def build_optimizer(model, args, config):
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "backbone" in name else other).append(parameter)
    weight_decay = (
        float(config["optimizer"].get("wd", 0.0))
        if args.weight_decay is None
        else args.weight_decay
    )
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.lr * args.backbone_lr_mult},
            {"params": other, "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, warmup_steps, total_steps):
    def multiplier(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def make_payload(
    model, ema_model, selected_model, selected_weights, optimizer, scheduler,
    scaler, epoch, batch_in_epoch, global_step, best_score, run_config,
    world_size, metrics=None, rng_states=None,
):
    return {
        "state_dict": selected_model.state_dict(),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "selected_weights": selected_weights,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        # An epoch-ending step checkpoint still needs validation on resume.
        "next_epoch": max(epoch, 0),
        "next_batch": batch_in_epoch,
        "global_step": global_step,
        "best_score": best_score,
        "validation_metrics": metrics,
        "config": run_config,
        "run_id": run_config["run_id"],
        "fingerprint": run_config["fingerprint"],
        "trainer": TRAINER_NAME,
        "world_size": world_size,
        "rng_states": rng_states,
    }


def check_checkpoint(checkpoint, run_config):
    if checkpoint.get("trainer") != TRAINER_NAME:
        raise ValueError("Only v3 checkpoints can be resumed; start anew from weights/KVQ.pth")
    if checkpoint.get("fingerprint") != run_config["fingerprint"]:
        raise ValueError("Resume configuration or annotations changed. Keep GPU count, batch, "
                         "epochs, optimizer, sampling and precision settings identical.")
    if not checkpoint.get("run_id"):
        raise ValueError("Checkpoint has no experiment ID")


def resume_cursor(checkpoint, steps_per_epoch, epochs):
    epoch, batch = int(checkpoint["next_epoch"]), int(checkpoint["next_batch"])
    if not 0 <= epoch <= epochs or not 0 <= batch <= steps_per_epoch:
        raise ValueError("Invalid checkpoint cursor")
    if epoch == epochs and batch:
        raise ValueError("Completed checkpoint cannot have a pending batch")
    return epoch, batch


def rank_zero_call(action):
    """Propagate rank-zero evaluation/save failures before other ranks continue."""
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = (True, action())
        except Exception as error:
            result[0] = (False, f"{type(error).__name__}: {error}")
    dist.broadcast_object_list(result, src=0)
    ok, value = result[0]
    if not ok:
        raise RuntimeError(value)
    return value


def advance_smoke_test(consecutive_successes, skipped, attempts, max_batches):
    """Only consecutive successful optimizer updates pass the bounded test."""
    if max_batches < 2 or not 1 <= attempts <= max_batches:
        raise ValueError("Smoke test needs max_batches >= 2 and attempts within the limit")
    consecutive_successes = 0 if skipped else consecutive_successes + 1
    if consecutive_successes >= 2:
        return consecutive_successes, "passed"
    if attempts >= max_batches:
        return consecutive_successes, "failed"
    return consecutive_successes, "running"


def train_one_epoch(
    loader, ddp_model, ema_model, optimizer, scheduler, scaler, device, rank,
    world_size, epoch, start_batch, global_step, args, fragment_config,
    total_steps, best_score, run_config,
):
    ddp_model.train()
    seed_everything(args.seed + rank * 100000 + epoch)
    spatial_h = int(fragment_config.get("fragments_h", 14))
    spatial_w = int(fragment_config.get("fragments_w", 14))
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    progress = tqdm(loader, desc=f"Train epoch {epoch + 1}", disable=rank != 0)
    resume_rng = getattr(args, "_resume_rng", None)
    args._resume_rng = None
    initial_step = global_step
    attempts, consecutive_successes = 0, 0
    smoke_status = "running"
    smoke_limit = min(args.smoke_max_batches, len(loader) - start_batch) if args.smoke_test else 0

    for batch_index, data in enumerate(progress):
        if batch_index < start_batch:
            continue
        if resume_rng is not None:
            restore_rng(resume_rng)
            resume_rng = None
        video = extract_video(data, device)
        target = data["gt_label"].float().to(device, non_blocking=True).reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            if args.lpc:
                fragments = video["fragments"]
                patch_video = make_patch_video(
                    fragments, spatial_h, spatial_w, args.lpc_time_grid
                )
                outputs = ddp_model(
                    video, epoch=epoch, inference=False, reduce_scores=True,
                    patch_flag=True, patch_video=patch_video,
                )
                if len(outputs) != 6:
                    raise RuntimeError(f"KVQ returned {len(outputs)} values; expected 6")
                scores, _, _, local_scores, _, patch_scores = outputs
            else:
                outputs = ddp_model(video, epoch=epoch, inference=False, reduce_scores=True)
                scores = outputs[0]

        # Reductions and statistical losses must happen in FP32, outside autocast.
        prediction = scores.float().mean((-3, -2, -1))
        if prediction.numel() != target.numel():
            raise ValueError(f"Expected one prediction per video: {prediction.shape}, {target.shape}")
        prediction = prediction.reshape(-1, 1)
        global_prediction = gather_global_batch(prediction, rank, world_size)
        global_target = gather_global_batch(target, rank, world_size)
        correlation = plcc_loss(global_prediction, global_target)
        ordering = rank_loss(global_prediction, global_target)
        if args.lpc:
            patch_scores = reshape_patch_scores(
                patch_scores, local_scores, target.shape[0],
                spatial_h, spatial_w, args.lpc_time_grid,
            )
            locality = lpc_loss(local_scores.float(), patch_scores.float())
        else:
            locality = prediction.new_zeros(())
        quality = correlation + 0.5 * ordering
        loss = world_size * quality + args.lpc_weight * locality

        valid = torch.isfinite(loss).to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise FloatingPointError(f"Non-finite loss on a rank at epoch {epoch + 1}, batch {batch_index}")

        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp_model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
        )
        finite_grad = torch.isfinite(grad_norm).to(torch.int32)
        dist.all_reduce(finite_grad, op=dist.ReduceOp.MIN)
        skipped = not bool(finite_grad.item())
        if skipped:
            diagnostic = (
                f"[rank{rank}] Skipped update: epoch={epoch + 1}, batch={batch_index + 1}, "
                f"step={global_step}, reason=nonfinite_grad_norm_on_at_least_one_rank, "
                f"local_grad_norm={grad_norm.item():.6g}, amp={args.amp}"
            )
            if not args.amp:
                raise FloatingPointError(diagnostic + "; non-finite gradients in FP32 training")
            scaler.update(new_scale=scale_before * 0.5)
            print(f"{diagnostic}, scale={scale_before:.6g}->{scaler.get_scale():.6g}", flush=True)
        else:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if ema_model is not None:
                update_ema(ema_model, ddp_model.module)
        values = torch.stack([
            quality.detach() + args.lpc_weight * locality.detach(),
            correlation.detach(), ordering.detach(), locality.detach(),
            global_prediction.detach().std(unbiased=False),
            prediction.new_tensor(float(skipped)), prediction.new_tensor(1.0),
        ])
        totals += values.to(torch.float64)
        if rank == 0:
            progress.set_postfix(
                plcc_loss=f"{correlation.item():.4f}", rank_loss=f"{ordering.item():.4f}",
                lpc=f"{locality.item():.4f}", pred_std=f"{values[4].item():.3g}",
                lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
                step=f"{global_step}/{total_steps}",
            )

        if not args.smoke_test and not skipped and global_step % args.checkpoint_every == 0:
            rng_states = gather_rng(world_size)
            def save_step():
                selected = ema_model if ema_model is not None else ddp_model.module
                payload = make_payload(
                    ddp_model.module, ema_model, selected,
                    "ema" if ema_model is not None else "raw", optimizer, scheduler,
                    scaler, epoch, batch_index + 1, global_step, best_score,
                    run_config, world_size, rng_states=rng_states,
                )
                step_path = Path(args.output_dir) / "checkpoints" / f"step_{global_step:09d}.pth"
                save_checkpoint(step_path, payload)
                save_checkpoint(Path(args.output_dir) / "last.pth", payload)
                print(f"Saved checkpoint: {step_path}", flush=True)
            rank_zero_call(save_step)
        attempts += 1
        if args.smoke_test:
            consecutive_successes, smoke_status = advance_smoke_test(
                consecutive_successes, skipped, attempts, smoke_limit
            )
            if rank == 0:
                print(
                    f"Smoke test: batch={attempts}/{smoke_limit}, "
                    f"successful_updates={global_step - initial_step}, "
                    f"consecutive={consecutive_successes}/2, skipped={skipped}, "
                    f"rank0_grad_norm={grad_norm.item():.6g}, "
                    f"scale={scale_before:.6g}->{scaler.get_scale():.6g}, status={smoke_status}",
                    flush=True,
                )
            if smoke_status != "running":
                break

    if resume_rng is not None:
        restore_rng(resume_rng)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(1.0, totals[6].item())
    averages = (totals[:5] / count).tolist()
    metrics = dict(zip(("loss", "plcc_loss", "rank_loss", "lpc_loss", "prediction_std"), averages))
    metrics["skipped_updates"] = int(totals[5].item() / world_size)
    metrics["successful_updates"] = global_step - initial_step
    if args.smoke_test:
        metrics.update(smoke_status=smoke_status, attempted_batches=attempts,
                       consecutive_successes=consecutive_successes, final_scale=scaler.get_scale())
    return global_step, metrics


def evaluate_variants(raw, ema, loader, device, label, args):
    def action():
        results = {}
        candidates = {"raw": raw}
        if ema is not None:
            candidates = {"ema": ema, **(candidates if args.eval_both else {})}
        for name, model in candidates.items():
            with evaluation_seed(loader.dataset.seed):
                results[name] = evaluate(
                    loader, model, device, f"{label} ({name})",
                    Path(args.output_dir) / "predictions" / f"{label}_{name}.json",
                )
            print(f"{label} {name}: {format_metrics(results[name])}", flush=True)
        return results
    return rank_zero_call(action)


def choose_best_variant(metrics_by_variant, raw_model, ema_model):
    variant = max(metrics_by_variant, key=lambda name: metric_score(metrics_by_variant[name]))
    return variant, ema_model if variant == "ema" else raw_model, metric_score(metrics_by_variant[variant])


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    local_rank, rank, world_size, device = init_distributed()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    args.batch_size = int(config["batch_size"]) if args.batch_size is None else args.batch_size
    args.num_workers = 2 if args.num_workers is None else args.num_workers
    epochs = int(config["num_epochs"]) if args.epochs is None else args.epochs
    global_batch = args.batch_size * world_size
    if args.smoke_test and (args.resume or args.check_only):
        raise ValueError("Use --smoke-test for a fresh test, separately from --resume/--check-only")
    if args.smoke_max_batches < 2:
        raise ValueError("smoke-max-batches must be at least 2")
    if world_size < 2 or args.batch_size < 1 or global_batch < 8:
        raise ValueError("Use at least 2 GPUs and global batch >= 8 (for example 4 GPUs x 2).")
    if epochs < 1 or args.num_workers < 0 or args.checkpoint_every < 1 or args.eval_every < 1:
        raise ValueError("epochs, checkpoint-every and eval-every must be positive; workers >= 0")
    if not 0 <= args.warmup_epochs < epochs:
        raise ValueError("warmup-epochs must be >= 0 and less than epochs")
    if args.lr <= 0 or args.backbone_lr_mult <= 0 or args.lpc_weight < 0 or args.lpc_time_grid < 1:
        raise ValueError("Invalid learning rate or LPC settings")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()

    split_paths = {"train": args.train_anno, "val": args.val_anno, "test": args.test_anno}
    def inspect_data():
        rows = {name: validate_annotation_file(path, args.video_root) for name, path in split_paths.items()}
        check_split_overlap(rows)
        prior_audit = []
        for path in args.prior_exposure_anno:
            previous = validate_annotation_file(path, args.prior_video_root)
            check_prior_exposure(rows, previous, path)
            prior_audit.append({"path": str(Path(path).resolve()), "sha256": file_hash(path),
                                "video_root": str(Path(args.prior_video_root).resolve()), "count": len(previous)})
        unknown = {name: sum(row["content_id"] is None for row in values) for name, values in rows.items()}
        if any(unknown.values()):
            print(f"Content IDs unavailable: {unknown}. Exact file overlap is checked; "
                  "unknown source-content relationships cannot be verified. VDPVE IDs are "
                  "inferred only for paths under VDPVE matching A0000_00-style filenames.", flush=True)
        for name in ("val", "test"):
            if len(rows[name]) < 2:
                raise ValueError(f"{name} needs at least two videos")
        result = {"counts": {k: len(v) for k, v in rows.items()},
                  "hashes": {k: file_hash(v) for k, v in split_paths.items()}}
        if prior_audit:
            result["prior_exposure_audit"] = prior_audit
        return result
    data_info = rank_zero_call(inspect_data)
    split_datasets = build_datasets(config, args)
    for name, dataset in split_datasets.items():
        if len(dataset) != data_info["counts"][name]:
            raise ValueError(f"{name}: dataset size differs from annotations; check FusionDataset")
    def audit_dataset_paths():
        for name, dataset in split_datasets.items():
            rows = validate_annotation_file(split_paths[name], args.video_root)
            validate_dataset_metadata(dataset, rows, name)
        print("FusionDataset exact video paths and MOS verified for all splits.", flush=True)
    rank_zero_call(audit_dataset_paths)
    steps_per_epoch = len(make_train_loader(split_datasets["train"], args, rank, world_size, 0))
    if steps_per_epoch == 0 or (args.smoke_test and steps_per_epoch < 2):
        raise ValueError("Not enough videos for the requested global batch")
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    weight_decay = float(config["optimizer"].get("wd", 0.0)) if args.weight_decay is None else args.weight_decay
    if not all(math.isfinite(v) for v in (args.lr, args.backbone_lr_mult, weight_decay,
                                         args.grad_clip, args.lpc_weight, args.warmup_epochs)):
        raise ValueError("Numeric configuration must be finite")
    if weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("weight-decay and grad-clip must be nonnegative")

    effective_data = {
        name: {"type": config["data"]["train"]["type"],
               "args": dataset_args(config["data"]["train"]["args"],
                                    "train" if name == "train" else "test",
                                    str(Path(path).resolve()), str(Path(args.video_root).resolve()))}
        for name, path in split_paths.items()
    }
    contract = {
        "model": config["model"], "data": effective_data, "annotations": data_info,
        "world_size": world_size, "batch_size": args.batch_size, "epochs": epochs,
        "lr": args.lr, "backbone_lr_mult": args.backbone_lr_mult, "weight_decay": weight_decay,
        "warmup_steps": warmup_steps, "total_steps": total_steps,
        "amp": args.amp, "ema": args.ema, "lpc": args.lpc, "lpc_weight": args.lpc_weight,
        "lpc_time_grid": args.lpc_time_grid, "seed": args.seed, "grad_clip": args.grad_clip,
        "static_graph": args.static_graph, "eval_both": args.eval_both,
    }
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    run_config = {"contract": contract, "fingerprint": fingerprint}

    def prepare_run():
        if args.resume:
            checkpoint = torch_load(args.resume, "cpu")
            check_checkpoint(checkpoint, run_config)
            best = torch_load(output_dir / "best.pth", "cpu")
            check_checkpoint(best, run_config)
            if best["run_id"] != checkpoint["run_id"]:
                raise ValueError("best.pth and resume checkpoint belong to different experiments")
            if best["global_step"] > checkpoint["global_step"]:
                raise ValueError("best.pth is newer than the resume checkpoint; resume last.pth")
            if best["best_score"] < checkpoint["best_score"]:
                raise ValueError("Missing best model from this checkpoint's training history")
            return {"run_id": checkpoint["run_id"], "initial_checkpoint": checkpoint["config"]["initial_checkpoint"]}
        if not Path(args.init_checkpoint).is_file():
            raise FileNotFoundError(args.init_checkpoint)
        if not args.check_only and not args.smoke_test:
            if (output_dir / "run.json").exists() or list(output_dir.glob("*.pth")):
                raise FileExistsError("Output directory contains an experiment. Use a new directory or --resume.")
        return {"run_id": uuid.uuid4().hex, "initial_checkpoint": {
            "path": str(Path(args.init_checkpoint).resolve()), "sha256": file_hash(args.init_checkpoint)}}
    run_config.update(rank_zero_call(prepare_run))

    import vqa.models as models
    model = getattr(models, config["model"]["type"])(**config["model"]["args"]).to(device)
    if not args.resume:
        load_initial_weights(model, args.init_checkpoint, device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    static_graph=args.static_graph,
                    find_unused_parameters=not args.static_graph)
    # DDP has now broadcast parameters/buffers from rank zero.
    ema_model = copy.deepcopy(ddp_model.module).eval() if args.ema else None
    if ema_model is not None:
        ema_model.requires_grad_(False)
    optimizer = build_optimizer(model, args, config)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler = make_grad_scaler(args.amp)
    start_epoch, start_batch, global_step, best_score = 0, 0, 0, -float("inf")
    if args.resume:
        checkpoint = torch_load(args.resume, "cpu")
        check_checkpoint(checkpoint, run_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if ema_model is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_batch = resume_cursor(checkpoint, steps_per_epoch, epochs)
        global_step, best_score = checkpoint["global_step"], checkpoint["best_score"]
        if start_batch:
            args._resume_rng = checkpoint["rng_states"][rank]
        del checkpoint
    if rank == 0:
        print(f"VDPVE counts={data_info['counts']}; run_id={run_config['run_id']}")
        print(f"GPUs={world_size}, per_gpu_batch={args.batch_size}, global_batch={global_batch}, "
              f"epochs={epochs}, steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
              f"warmup_steps={warmup_steps}, head_lr={args.lr:.2e}, "
              f"backbone_lr={args.lr * args.backbone_lr_mult:.2e}, AMP={args.amp}, LPC={args.lpc}")
        if total_steps < args.checkpoint_every:
            print(f"Total updates < {args.checkpoint_every}: periodic step saving will not trigger; "
                  "last.pth is saved each epoch and best.pth on validation improvement.")
        if args.resume:
            print(f"Resume: next_epoch={start_epoch + 1}, next_batch={start_batch}, step={global_step}")
    if args.check_only:
        if rank == 0:
            print("Paths, weights and DDP initialization passed. No video forward/backward was tested. "
                  "Use --smoke-test to test two real updates.")
        return

    fragment_config = config["data"]["train"]["args"]["sample_types"]["fragments"]
    if args.smoke_test:
        global_step, metrics = train_one_epoch(
            make_train_loader(split_datasets["train"], args, rank, world_size, 0),
            ddp_model, ema_model, optimizer, scheduler, scaler, device, rank,
            world_size, 0, 0, 0, args, fragment_config, total_steps, 0.0, run_config,
        )
        if metrics["smoke_status"] != "passed":
            raise RuntimeError(
                "Smoke test failed: no two consecutive successful DDP updates before the "
                f"attempt limit or end of loader. metrics={metrics}. Inspect skipped-update "
                "diagnostics above; persistent non-finite gradients need investigation. "
                "A separate --no-amp smoke test can help isolate FP16 issues if memory permits."
            )
        if rank == 0:
            print(f"Smoke test passed: two consecutive successful DDP updates, metrics={metrics}; "
                  "no weights saved. This short test does not establish convergence.", flush=True)
        return

    rank_zero_call(lambda: write_json(output_dir / "run.json", run_config))
    val_loader = make_eval_loader(split_datasets["val"], args, args.seed + 100000) if rank == 0 else None
    test_loader = make_eval_loader(split_datasets["test"], args, args.seed + 200000) if rank == 0 else None
    if not args.resume:
        baseline = evaluate_variants(model, ema_model, val_loader, device, "baseline_val", args)
        variant, selected, best_score = choose_best_variant(baseline, model, ema_model)
        if not math.isfinite(best_score):
            raise RuntimeError("Initial predictions are constant or invalid; fix baseline before training")
        rng_states = gather_rng(world_size)
        def save_baseline():
            payload = make_payload(model, ema_model, selected, variant, optimizer, scheduler,
                                   scaler, -1, 0, 0, best_score, run_config, world_size,
                                   baseline, rng_states)
            save_checkpoint(output_dir / "baseline.pth", payload)
            save_checkpoint(output_dir / "best.pth", payload)
            save_checkpoint(output_dir / "last.pth", payload)
            write_json(output_dir / "baseline_metrics.json", baseline)
        rank_zero_call(save_baseline)

    for epoch in range(start_epoch, epochs):
        global_step, train_metrics = train_one_epoch(
            make_train_loader(split_datasets["train"], args, rank, world_size, epoch),
            ddp_model, ema_model, optimizer, scheduler, scaler, device, rank, world_size,
            epoch, start_batch if epoch == start_epoch else 0, global_step, args,
            fragment_config, total_steps, best_score, run_config,
        )
        validation = None
        selected, variant, score = (ema_model if ema_model is not None else model), ("ema" if ema_model is not None else "raw"), -float("inf")
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == epochs:
            validation = evaluate_variants(model, ema_model, val_loader, device,
                                           f"epoch_{epoch + 1:03d}_val", args)
            variant, selected, score = choose_best_variant(validation, model, ema_model)
        improved = score > best_score
        best_score = max(best_score, score)
        rng_states = gather_rng(world_size)
        def save_epoch():
            payload = make_payload(model, ema_model, selected, variant, optimizer, scheduler,
                                   scaler, epoch, steps_per_epoch, global_step, best_score,
                                   run_config, world_size, validation, rng_states)
            payload.update(next_epoch=epoch + 1, next_batch=0)
            if improved:
                save_checkpoint(output_dir / "best.pth", payload)
                print(f"New best: epoch={epoch + 1}, step={global_step}, weights={variant}, score={score:.6f}")
            save_checkpoint(output_dir / "last.pth", payload)
            report = {"epoch": epoch + 1, "global_step": global_step, "train": train_metrics,
                      "validation": validation, "best_score": best_score,
                      "head_lr": optimizer.param_groups[-1]["lr"],
                      "backbone_lr": optimizer.param_groups[0]["lr"]}
            write_json(output_dir / "metrics" / f"epoch_{epoch + 1:03d}.json", report)
            print(f"Epoch {epoch + 1}/{epochs}: step={global_step}, train={train_metrics}", flush=True)
        rank_zero_call(save_epoch)

    def final_evaluation():
        best = torch_load(output_dir / "best.pth", "cpu")
        check_checkpoint(best, run_config)
        if best["run_id"] != run_config["run_id"]:
            raise ValueError("Best checkpoint has the wrong experiment ID")
        model.load_state_dict(best["state_dict"], strict=True)
        print(f"Selected best: epoch={best['epoch'] + 1} (0=baseline), step={best['global_step']}, "
              f"weights={best['selected_weights']}, recorded_validation={best['validation_metrics']}")
        with evaluation_seed(val_loader.dataset.seed):
            val_metrics = evaluate(val_loader, model, device, "Best validation recheck",
                                   output_dir / "predictions" / "best_val.json")
        with evaluation_seed(test_loader.dataset.seed):
            test_metrics = evaluate(test_loader, model, device, "Best test",
                                    output_dir / "predictions" / "best_test.json")
        # Compare the untouched initial model on exactly the same test videos.
        initial = run_config["initial_checkpoint"]
        if file_hash(initial["path"]) != initial["sha256"]:
            raise ValueError("Initial checkpoint changed during the experiment")
        load_initial_weights(model, initial["path"], device)
        with evaluation_seed(test_loader.dataset.seed):
            baseline_test = evaluate(test_loader, model, device, "Initial model test",
                                     output_dir / "predictions" / "baseline_test.json")
        report = {"run_id": run_config["run_id"], "selected_epoch": best["epoch"] + 1,
                  "selected_step": best["global_step"], "selected_weights": best["selected_weights"],
                  "recorded_validation": best["validation_metrics"],
                  "rechecked_validation": val_metrics, "best_test": test_metrics,
                  "initial_test": baseline_test}
        best.update(test_metrics=test_metrics, rechecked_validation=val_metrics)
        save_checkpoint(output_dir / "final.pth", best)
        write_json(output_dir / "summary.json", report)
        print(f"Best test: {format_metrics(test_metrics)}")
        print(f"Initial model test: {format_metrics(baseline_test)}")
    rank_zero_call(final_evaluation)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
