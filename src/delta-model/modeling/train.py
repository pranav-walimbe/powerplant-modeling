"""Train seasonal and context-aware vision emissions-change classifiers."""

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from modeling.autoencoder import MaskedNO2Autoencoder
from modeling.convgru import (
    DEFAULT_CONTEXT_DROPOUT,
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_DIM,
    DEFAULT_MODEL_WIDTH,
    JointVisionClassifier,
)
from modeling.dataset import (
    MODEL_FEATURE_NAMES,
    NormalizationStats,
    NOxDataset,
    compute_stats,
    save_stats,
)
from modeling.eval_utils import (
    LOGIT_COLUMNS,
    PREDICTED_CLASS_COL,
    PROBABILITY_COLUMNS,
    TRUE_CLASS_COL,
    classification_metrics,
    save_results,
)
from modeling.mlp import SeasonalMLP
from modeling.plot_utils import (
    plot_accuracy_by_characteristic,
    plot_joint_prediction_samples,
    plot_ops_evaluation,
    plot_split_class_f1,
    plot_training_curves,
)
from modeling.transport import TransportConfig, TransportTransform
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import (
    MODEL_CLASS_NAMES,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_IMAGE_KEYS,
    MODEL_INPUT_CHANNELS,
    MODEL_MASK_KEYS,
    MODEL_TARGET_COL,
    NUM_CORES,
    OPS_DATASET_CSV,
    PRETRAINED_MASKED_MODEL_WEIGHTS,
    RUNS_DIR,
    SEQUENCE_TIMESTEPS,
)

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 100
DEFAULT_SEASONAL_EPOCHS = 75
DEFAULT_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_SEASONAL_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_NORM = 5.0
DEFAULT_SCHEDULER_PATIENCE = 10
DEFAULT_SCHEDULER_FACTOR = 0.50
DEFAULT_EARLY_STOP_PATIENCE = 15
DEFAULT_RASTER_LOSS_WEIGHT = 1.0
DEFAULT_TARGET_SMOOTHING_SIGMA = 1.5
JOINT_PLOT_SAMPLE_COUNT = 20
DELTA_TARGET_SIZE = 12
DELTA_TARGET_UPWIND_FRACTION = 1.0 / 3.0
DELTA_TARGET_MIN_WIND_SPEED_MPS = 0.5
ROBUST_STD_NORMALIZER = 1.349
MIN_TARGET_SCALE = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse model-training command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seasonal-epochs", type=int, default=DEFAULT_SEASONAL_EPOCHS)
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, NUM_CORES))
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seasonal-learning-rate", type=float, default=DEFAULT_SEASONAL_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler-factor", type=float, default=DEFAULT_SCHEDULER_FACTOR)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--raster-loss-weight", type=float, default=DEFAULT_RASTER_LOSS_WEIGHT)
    parser.add_argument("--target-smoothing-sigma", type=float, default=DEFAULT_TARGET_SMOOTHING_SIGMA)
    parser.add_argument("--pretrained-masked-model-weights", default=PRETRAINED_MASKED_MODEL_WEIGHTS)
    parser.add_argument("--completed-raster-dir", required=True)
    parser.add_argument("--ops-manifest-path", default=OPS_DATASET_CSV)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    # Resolve the requested training device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    # Seed CPU and CUDA random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    # Move model inputs and class labels onto the training device
    image, seasonal, transport, target, index, *_ = batch
    non_blocking = device.type == "cuda"
    return (
        image.to(device, non_blocking=non_blocking),
        seasonal.to(device, non_blocking=non_blocking),
        transport.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
        index,
    )


def _move_joint_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    # Move joint classification and raster supervision onto the training device
    image, seasonal, transport, target, index, delta_target = batch
    non_blocking = device.type == "cuda"
    return (
        image.to(device, non_blocking=non_blocking),
        seasonal.to(device, non_blocking=non_blocking),
        transport.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
        index,
        delta_target.to(device, non_blocking=non_blocking),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    class_weights: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    """Train one epoch and return weighted mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_loss_weight = 0.0
    correct = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        image, seasonal, transport, target, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(image, seasonal, transport)
            per_record_loss = criterion(logits, target)
            loss_weight = class_weights[target].sum()
            loss = per_record_loss.sum() / loss_weight
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += per_record_loss.detach().sum().item()
        total_loss_weight += loss_weight.item()
        correct += int((logits.argmax(dim=1) == target).sum().item())
    return total_loss / total_loss_weight, correct / len(loader.dataset)


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return weighted loss, accuracy, and macro F1 for one validation epoch."""
    model.eval()
    total_loss = 0.0
    total_loss_weight = 0.0
    correct = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, seasonal, transport, target, _ = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(image, seasonal, transport)
                logits = output[0] if isinstance(output, tuple) else output
                per_record_loss = criterion(logits, target)
            predicted = logits.argmax(dim=1)
            total_loss += per_record_loss.sum().item()
            total_loss_weight += class_weights[target].sum().item()
            correct += int((predicted == target).sum().item())
            targets.append(target.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
    truth = np.concatenate(targets)
    predicted = np.concatenate(predictions)
    macro_f1 = float(
        f1_score(
            truth,
            predicted,
            labels=np.arange(len(MODEL_CLASS_NAMES)),
            average="macro",
            zero_division=0,
        )
    )
    return total_loss / total_loss_weight, correct / len(loader.dataset), macro_f1


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    class_weight_values: tuple[float, ...],
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, object],
    phase_name: str,
) -> tuple[list[float], list[float], float, float]:
    """Fit one classifier and restore its best-validation-macro-F1 state."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=args.weight_decay)
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_validation_macro_f1 = float("-inf")
    best_validation_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            class_weights,
            scaler,
            device,
            args.gradient_clip_norm,
        )
        validation_loss, validation_accuracy, validation_macro_f1 = val_epoch(
            model,
            val_loader,
            criterion,
            class_weights,
            device,
        )
        scheduler.step(validation_loss)
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        learning_rate_now = optimizer.param_groups[0]["lr"]
        print(
            f"{phase_name} epoch {epoch:03d} | train loss {train_loss:.5f} acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.5f} acc {validation_accuracy:.4f} macro F1 {validation_macro_f1:.4f} | "
            f"lr {learning_rate_now:.2e}"
        )
        selection_improved = validation_macro_f1 > best_validation_macro_f1 or (
            validation_macro_f1 == best_validation_macro_f1 and validation_loss < best_validation_loss
        )
        if selection_improved:
            best_validation_macro_f1 = validation_macro_f1
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_macro_f1,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping {phase_name} at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return train_losses, val_losses, best_validation_macro_f1, best_validation_loss


def raster_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return one unweighted raster L1 value per sample.

    Args:
        prediction: Predicted normalized rasters shaped batch by height by width.
        target: Target normalized rasters matching ``prediction``.

    Returns:
        Per-sample mean absolute errors.
    """
    spatial_dimensions = tuple(range(1, prediction.ndim))
    return (prediction - target).abs().mean(dim=spatial_dimensions)


def train_joint_epoch(
    model: JointVisionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    class_weights: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
    raster_loss_weight: float,
) -> tuple[float, float, float, float]:
    """Train one joint epoch and return total, component, and accuracy metrics."""
    model.train()
    total_classification_loss = 0.0
    total_classification_weight = 0.0
    total_raster_loss = 0.0
    correct = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        image, seasonal, transport, target, _, delta_target = _move_joint_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits, raster_prediction = model(image, seasonal, transport)
            per_record_classification = criterion(logits, target)
            classification_weight = class_weights[target].sum()
            classification_loss = per_record_classification.sum() / classification_weight
            per_record_raster = raster_l1(raster_prediction, delta_target)
            raster_loss = per_record_raster.mean()
            loss = classification_loss + raster_loss_weight * raster_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_classification_loss += per_record_classification.detach().sum().item()
        total_classification_weight += classification_weight.item()
        total_raster_loss += per_record_raster.detach().sum().item()
        correct += int((logits.argmax(dim=1) == target).sum().item())
    classification_epoch_loss = total_classification_loss / total_classification_weight
    raster_epoch_loss = total_raster_loss / len(loader.dataset)
    total_epoch_loss = classification_epoch_loss + raster_loss_weight * raster_epoch_loss
    return total_epoch_loss, classification_epoch_loss, raster_epoch_loss, correct / len(loader.dataset)


def val_joint_epoch(
    model: JointVisionClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    class_weights: torch.Tensor,
    device: torch.device,
    raster_loss_weight: float,
) -> tuple[float, float, float, float, float]:
    """Evaluate both joint tasks and return loss, accuracy, and macro F1 metrics."""
    model.eval()
    total_classification_loss = 0.0
    total_classification_weight = 0.0
    total_raster_loss = 0.0
    correct = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, seasonal, transport, target, _, delta_target = _move_joint_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, raster_prediction = model(image, seasonal, transport)
                per_record_classification = criterion(logits, target)
                per_record_raster = raster_l1(raster_prediction, delta_target)
            predicted = logits.argmax(dim=1)
            total_classification_loss += per_record_classification.sum().item()
            total_classification_weight += class_weights[target].sum().item()
            total_raster_loss += per_record_raster.sum().item()
            correct += int((predicted == target).sum().item())
            targets.append(target.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
    truth = np.concatenate(targets)
    predicted = np.concatenate(predictions)
    macro_f1 = float(
        f1_score(
            truth,
            predicted,
            labels=np.arange(len(MODEL_CLASS_NAMES)),
            average="macro",
            zero_division=0,
        )
    )
    classification_epoch_loss = total_classification_loss / total_classification_weight
    raster_epoch_loss = total_raster_loss / len(loader.dataset)
    total_epoch_loss = classification_epoch_loss + raster_loss_weight * raster_epoch_loss
    return (
        total_epoch_loss,
        classification_epoch_loss,
        raster_epoch_loss,
        correct / len(loader.dataset),
        macro_f1,
    )


def fit_joint_model(
    model: JointVisionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    class_weight_values: tuple[float, ...],
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, object],
) -> tuple[list[float], list[float], float, float, float, float]:
    """Fit the joint model and restore its best validation macro-F1 state."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=args.weight_decay)
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_validation_macro_f1 = float("-inf")
    best_classification_loss = float("inf")
    best_total_loss = float("inf")
    best_raster_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_total, train_classification, train_raster, train_accuracy = train_joint_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            class_weights,
            scaler,
            device,
            args.gradient_clip_norm,
            args.raster_loss_weight,
        )
        validation_total, validation_classification, validation_raster, validation_accuracy, validation_macro_f1 = (
            val_joint_epoch(
                model,
                val_loader,
                criterion,
                class_weights,
                device,
                args.raster_loss_weight,
            )
        )
        scheduler.step(validation_classification)
        train_losses.append(train_total)
        val_losses.append(validation_total)
        learning_rate_now = optimizer.param_groups[0]["lr"]
        print(
            f"Joint vision model epoch {epoch:03d} | "
            f"train loss {train_total:.5f} (CE {train_classification:.5f}, raster {train_raster:.5f}) "
            f"acc {train_accuracy:.4f} | "
            f"val loss {validation_total:.5f} "
            f"(CE {validation_classification:.5f}, raster {validation_raster:.5f}) "
            f"acc {validation_accuracy:.4f} macro F1 {validation_macro_f1:.4f} | lr {learning_rate_now:.2e}"
        )
        selection_improved = validation_macro_f1 > best_validation_macro_f1 or (
            validation_macro_f1 == best_validation_macro_f1
            and validation_classification < best_classification_loss
        )
        if selection_improved:
            best_validation_macro_f1 = validation_macro_f1
            best_classification_loss = validation_classification
            best_total_loss = validation_total
            best_raster_loss = validation_raster
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_total,
                    "validation_classification_loss": validation_classification,
                    "validation_raster_l1": validation_raster,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_macro_f1,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping joint vision model at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return (
        train_losses,
        val_losses,
        best_validation_macro_f1,
        best_total_loss,
        best_classification_loss,
        best_raster_loss,
    )


def _checkpoint_sha256(path: Path) -> str:
    # Identify the exact masked checkpoint used for reconstruction
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_masked_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, object], MaskedNO2Autoencoder]:
    # Load the reconstruction model used to fill missing NO2 pixels
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["image_keys"]) != tuple(MODEL_IMAGE_KEYS):
        raise ValueError("Masked checkpoint image keys do not match delta raster channels")
    model = MaskedNO2Autoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def _fill_missing_rasters(
    model: MaskedNO2Autoencoder,
    datasets: dict[str, NOxDataset],
    *,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Path]:
    # Materialize completed rasters and their original NO2 quality channel
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for split, dataset in datasets.items():
            destination_path = output_dir / f"{split}_completed_rasters.npy"
            destination = None
            loader = _loader(dataset, shuffle=False, args=args, device=device)
            for batch_number, batch in enumerate(loader, start=1):
                image, _, _, _, indices = _move_batch(batch, device)
                batch_size, timesteps, channels, height, width = image.shape
                frames = image.reshape(batch_size * timesteps, channels, height, width)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    completed_no2 = model.fill_missing(frames)
                completed_frames = frames[:, :MODEL_INPUT_CHANNELS].clone()
                completed_frames[:, :1] = completed_no2
                completed = completed_frames.reshape(
                    batch_size,
                    timesteps,
                    MODEL_INPUT_CHANNELS,
                    height,
                    width,
                )
                completed = completed.float().clamp_(-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS).cpu().numpy()
                if destination is None:
                    destination = np.lib.format.open_memmap(
                        destination_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(dataset), timesteps, MODEL_INPUT_CHANNELS, height, width),
                    )
                destination[indices.numpy()] = completed
                if batch_number % 100 == 0 or batch_number == len(loader):
                    print(f"Masked fill {split}: {batch_number:,}/{len(loader):,} batches")
            if destination is None:
                raise ValueError(f"Cannot fill empty {split} split")
            destination.flush()
            del destination, loader
            paths[split] = destination_path
    return paths


def _grid_coordinates(height: int, width: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Return east and north pixel offsets from the raster center
    rows = torch.arange(height, dtype=reference.dtype, device=reference.device) - (height - 1) / 2
    columns = torch.arange(width, dtype=reference.dtype, device=reference.device) - (width - 1) / 2
    east, south = torch.meshgrid(columns, rows, indexing="xy")
    return east, -south


def _upwind_background(no2: torch.Tensor, wind_u: torch.Tensor, wind_v: torch.Tensor) -> torch.Tensor:
    # Estimate one background per sample and timestep
    batch_size, timesteps, height, width = no2.shape
    reference_u = torch.nanmedian(wind_u.reshape(batch_size, timesteps, -1), dim=-1).values
    reference_v = torch.nanmedian(wind_v.reshape(batch_size, timesteps, -1), dim=-1).values
    finite_wind = torch.isfinite(reference_u) & torch.isfinite(reference_v)
    safe_u = torch.where(finite_wind, reference_u, torch.zeros_like(reference_u))
    safe_v = torch.where(finite_wind, reference_v, torch.zeros_like(reference_v))
    speed = torch.hypot(safe_u, safe_v)

    east, north = _grid_coordinates(height, width, no2)
    safe_speed = speed.clamp_min(DELTA_TARGET_MIN_WIND_SPEED_MPS)
    along_wind = (
        east[None, None] * safe_u[:, :, None, None]
        + north[None, None] * safe_v[:, :, None, None]
    ) / safe_speed[:, :, None, None]
    cutoff = torch.quantile(along_wind.flatten(2), DELTA_TARGET_UPWIND_FRACTION, dim=-1)
    upwind_values = no2.masked_fill(along_wind > cutoff[:, :, None, None], torch.nan)
    upwind_background = torch.nanmedian(upwind_values.flatten(2), dim=-1).values
    full_background = torch.nanmedian(no2.flatten(2), dim=-1).values
    use_upwind = (
        finite_wind
        & (speed >= DELTA_TARGET_MIN_WIND_SPEED_MPS)
        & torch.isfinite(upwind_background)
    )
    return torch.where(use_upwind, upwind_background, full_background)


def _window_means(values: torch.Tensor, elapsed_hours: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculate early and late trapezoidal means
    interval_means = (values[:, :-1] + values[:, 1:]) / 2.0
    spatial_dimensions = (1,) * (values.ndim - 2)
    weights = elapsed_hours.reshape(elapsed_hours.shape[0], elapsed_hours.shape[1], *spatial_dimensions)
    early = (interval_means[:, :2] * weights[:, :2]).sum(dim=1) / weights[:, :2].sum(dim=1)
    late = (interval_means[:, 2:] * weights[:, 2:]).sum(dim=1) / weights[:, 2:].sum(dim=1)
    return early, late


def _gaussian_smooth(values: torch.Tensor, sigma: float) -> torch.Tensor:
    # Smooth the pooled target in target-grid cells
    if sigma == 0.0:
        return values
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, dtype=values.dtype, device=values.device)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)[None, None]
    padded = F.pad(values[:, None], (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel_2d).squeeze(1)


def _delta_raster_target(
    physical_image: torch.Tensor,
    elapsed_hours: torch.Tensor,
    smoothing_sigma: float,
) -> torch.Tensor:
    # Build a pooled and smoothed physical NO2-change target
    no2 = physical_image[:, :, MODEL_IMAGE_KEYS.index("no2")]
    background = _upwind_background(
        no2,
        physical_image[:, :, MODEL_IMAGE_KEYS.index("wind_u_80m_mps")],
        physical_image[:, :, MODEL_IMAGE_KEYS.index("wind_v_80m_mps")],
    )
    early, late = _window_means(no2 - background[:, :, None, None], elapsed_hours)
    target = late - early
    pool_factor = physical_image.shape[-1] // DELTA_TARGET_SIZE
    pooled = F.avg_pool2d(target[:, None], pool_factor).squeeze(1)
    return _gaussian_smooth(pooled, smoothing_sigma)


def _delta_target_metadata(smoothing_sigma: float) -> dict[str, float | int | str]:
    # Describe the auxiliary target construction
    return {
        "target_size": DELTA_TARGET_SIZE,
        "smoothing_sigma_cells": smoothing_sigma,
        "upwind_fraction": DELTA_TARGET_UPWIND_FRACTION,
        "min_wind_speed_mps": DELTA_TARGET_MIN_WIND_SPEED_MPS,
        "background_method": "per_timestep_wind_relative_upwind_third_median",
        "calm_wind_fallback": "full_raster_median",
        "temporal_method": "late_minus_early_trapezoidal_mean",
        "no2_source": "masked_model_completed_physical_units",
    }


def _materialize_delta_targets(
    completed_paths: dict[str, Path],
    datasets: dict[str, NOxDataset],
    source_image_stats: dict[str, object],
    *,
    output_dir: Path,
    batch_size: int,
    smoothing_sigma: float,
) -> dict[str, Path]:
    # Materialize physical auxiliary targets
    output_dir.mkdir(parents=True, exist_ok=False)
    source_center = np.asarray(source_image_stats["image_center"], dtype=np.float32)
    source_scale = np.asarray(source_image_stats["image_scale"], dtype=np.float32)
    target_paths: dict[str, Path] = {}
    with torch.inference_mode():
        for split, dataset in datasets.items():
            completed = np.load(completed_paths[split], mmap_mode="r")
            target_path = output_dir / f"{split}_delta_targets.npy"
            target_destination = np.lib.format.open_memmap(
                target_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), DELTA_TARGET_SIZE, DELTA_TARGET_SIZE),
            )
            batch_count = math.ceil(len(dataset) / batch_size)
            for batch_number, start in enumerate(range(0, len(dataset), batch_size), start=1):
                stop = min(start + batch_size, len(dataset))
                completed_batch = np.asarray(completed[start:stop])
                numeric = completed_batch[:, :, : len(MODEL_IMAGE_KEYS)]
                physical = numeric * source_scale[None, None, :, None, None]
                physical = physical + source_center[None, None, :, None, None]
                elapsed_hours = dataset.elapsed_hours[start:stop]
                target = _delta_raster_target(
                    torch.from_numpy(physical),
                    torch.from_numpy(elapsed_hours),
                    smoothing_sigma,
                )
                target_destination[start:stop] = target.numpy()
                if batch_number % 100 == 0 or batch_number == batch_count:
                    print(f"Delta-target construction {split}: {batch_number:,}/{batch_count:,} batches")
            target_destination.flush()
            del completed, target_destination
            target_paths[split] = target_path
    return target_paths


def _fit_delta_target_stats(path: Path) -> dict[str, float]:
    # Fit robust auxiliary-target normalization on training values only
    target = np.load(path, mmap_mode="r")
    lower, center, upper = np.percentile(target, (25.0, 50.0, 75.0))
    scale = max(float(upper - lower) / ROBUST_STD_NORMALIZER, MIN_TARGET_SCALE)
    return {"center": float(center), "scale": scale}


def _fit_transport_residual_stats(
    loader: DataLoader,
    stats: NormalizationStats,
    config: TransportConfig,
) -> dict[str, float]:
    # Fit quality-weighted physical residual moments on the training split
    transport = TransportTransform(stats.image_center, stats.image_scale, config=config)
    total_weight = 0.0
    weighted_sum = 0.0
    weighted_square_sum = 0.0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            image, _, _, _, indices = batch
            source_indices = indices.numpy()
            elapsed_hours = torch.from_numpy(loader.dataset.elapsed_hours[source_indices])
            no2_normalization = torch.from_numpy(loader.dataset.no2_normalization[source_indices])
            residual, quality = transport.physical_residual(image, elapsed_hours, no2_normalization)
            weights = quality.double()
            residual = residual.double()
            total_weight += weights.sum().item()
            weighted_sum += (weights * residual).sum().item()
            weighted_square_sum += (weights * residual.square()).sum().item()
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(f"Transport-statistics scan: {batch_number:,}/{len(loader):,} batches")
    center = weighted_sum / total_weight
    variance = max(weighted_square_sum / total_weight - center * center, 1e-12)
    return {
        "center": center,
        "scale": variance**0.5,
        "effective_pixel_weight": total_weight,
    }


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    context_permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate class outputs and optional raster predictions for one split."""
    model.eval()
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    raster_batches: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, seasonal, transport, _, index = _move_batch(batch, device)
            if context_permutation is not None:
                source_indices = context_permutation[index.numpy()]
                seasonal = torch.from_numpy(loader.dataset.features[source_indices]).to(
                    device,
                    non_blocking=device.type == "cuda",
                )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(image, seasonal, transport)
                if isinstance(output, tuple):
                    logits, raster_prediction = output
                    raster_batches.append(raster_prediction.float().cpu().numpy())
                else:
                    logits = output
            logits = logits.float()
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(logits.softmax(dim=1).cpu().numpy())
            indices.append(index.numpy())
    raster_predictions = np.concatenate(raster_batches) if raster_batches else np.empty(0, dtype=np.float32)
    return (
        np.concatenate(logits_batches),
        np.concatenate(probability_batches),
        np.concatenate(indices),
        raster_predictions,
    )


def _loader(dataset: NOxDataset, *, shuffle: bool, args: argparse.Namespace, device: torch.device) -> DataLoader:
    # Configure one deterministic data loader
    options: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers:
        options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(dataset, **options)


def _prediction_frame(
    dataset: NOxDataset,
    logits: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    # Attach class predictions in source-record order
    frame = dataset.frame.iloc[indices].copy().reset_index(drop=True)
    frame[TRUE_CLASS_COL] = dataset.labels[indices]
    frame[PREDICTED_CLASS_COL] = probabilities.argmax(axis=1)
    for column, values in zip(LOGIT_COLUMNS, logits.T, strict=True):
        frame[column] = values
    for column, values in zip(PROBABILITY_COLUMNS, probabilities.T, strict=True):
        frame[column] = values
    return frame


def _joint_prediction_sample_arrays(
    dataset: NOxDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    raster_predictions: np.ndarray,
    transport_residual_stats: dict[str, float],
    seed: int,
) -> tuple[np.ndarray, ...]:
    # Collect deterministic test examples in physical units
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(indices), size=JOINT_PLOT_SAMPLE_COUNT, replace=False))
    sample_indices = indices[positions]
    records = [dataset[int(index)] for index in sample_indices]
    images = np.stack([record[0].numpy() for record in records])
    transport = np.stack([record[2].numpy() for record in records])
    normalized_targets = np.stack([record[5].numpy() for record in records])

    no2_channel = MODEL_IMAGE_KEYS.index("no2")
    no2_rasters = images[:, :, no2_channel]
    no2_rasters = no2_rasters * dataset.image_scale[no2_channel] + dataset.image_center[no2_channel]
    transport_residuals = transport[:, :, 0]
    transport_residuals = (
        transport_residuals * transport_residual_stats["scale"]
        + transport_residual_stats["center"]
    )
    target_rasters = normalized_targets * dataset.delta_target_scale + dataset.delta_target_center
    predicted_rasters = (
        raster_predictions[positions] * dataset.delta_target_scale + dataset.delta_target_center
    )
    true_classes = dataset.labels[sample_indices]
    predicted_classes = probabilities[positions].argmax(axis=1)
    return (
        no2_rasters,
        transport_residuals,
        target_rasters,
        predicted_rasters,
        true_classes,
        predicted_classes,
    )


def _class_counts(dataset: NOxDataset) -> dict[str, int]:
    # Count training examples in the configured class order
    return {name: int(np.sum(dataset.labels == index)) for index, name in enumerate(MODEL_CLASS_NAMES)}


def _class_weights(dataset: NOxDataset) -> dict[str, float]:
    # Give every training class equal total loss weight
    counts = np.bincount(dataset.labels, minlength=len(MODEL_CLASS_NAMES))
    if np.any(counts == 0):
        missing = [name for name, count in zip(MODEL_CLASS_NAMES, counts, strict=True) if count == 0]
        raise ValueError(f"Training data has no records for classes: {', '.join(missing)}")
    weights = len(dataset) / (len(MODEL_CLASS_NAMES) * counts)
    return {name: float(weights[index]) for index, name in enumerate(MODEL_CLASS_NAMES)}


def main() -> None:
    """Train and evaluate seasonal and context-aware vision classifiers."""
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)
    completed_root = Path(args.completed_raster_dir).resolve()
    if Path("/tmp") not in completed_root.parents:
        raise ValueError("Completed raster directory must be under job-local /tmp")
    masked_checkpoint_path = Path(args.pretrained_masked_model_weights)
    if not args.pretrained_masked_model_weights:
        raise ValueError("Set PRETRAINED_MASKED_MODEL_WEIGHTS or pass --pretrained-masked-model-weights")

    run_name = datetime.now(timezone.utc).strftime("delta_joint_raster_%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    print(f"Training seasonal and vision classifiers on {device}; outputs: {run_dir}")

    masked_model_checkpoint, imputation_model = _load_masked_checkpoint(masked_checkpoint_path, device)
    masked_image_stats = masked_model_checkpoint["normalization_stats"]
    stats = compute_stats("train")
    del masked_model_checkpoint
    save_stats(stats, run_dir / "normalization_stats.json")

    source_datasets = {
        split: NOxDataset(split, stats, source_image_stats=masked_image_stats) for split in ("train", "val", "test")
    }
    masked_checkpoint_sha256 = _checkpoint_sha256(masked_checkpoint_path)
    completed_dir = completed_root / run_name
    completed_paths = _fill_missing_rasters(
        imputation_model,
        source_datasets,
        output_dir=completed_dir,
        args=args,
        device=device,
    )
    delta_target_metadata = _delta_target_metadata(args.target_smoothing_sigma)
    delta_target_paths = _materialize_delta_targets(
        completed_paths,
        source_datasets,
        masked_image_stats,
        output_dir=completed_dir / "delta-targets",
        batch_size=args.batch_size,
        smoothing_sigma=args.target_smoothing_sigma,
    )
    delta_target_stats = _fit_delta_target_stats(delta_target_paths["train"])
    with (run_dir / "delta_target_stats.json").open("w") as destination:
        json.dump({**delta_target_stats, **delta_target_metadata}, destination, indent=2)
    del imputation_model, source_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    datasets = {
        split: NOxDataset(
            split,
            stats,
            completed_raster_path=completed_paths[split],
            source_image_stats=masked_image_stats,
        )
        for split in ("train", "val", "test")
    }
    transport_config = TransportConfig()
    transport_stats_loader = _loader(
        datasets["train"],
        shuffle=False,
        args=args,
        device=torch.device("cpu"),
    )
    transport_residual_stats = _fit_transport_residual_stats(transport_stats_loader, stats, transport_config)
    del transport_stats_loader
    transport_transform = TransportTransform(
        stats.image_center,
        stats.image_scale,
        residual_center=transport_residual_stats["center"],
        residual_scale=transport_residual_stats["scale"],
        config=transport_config,
    )
    transport_datasets = {
        split: NOxDataset(
            split,
            stats,
            completed_raster_path=completed_paths[split],
            source_image_stats=masked_image_stats,
            transport_transform=transport_transform,
            per_timestep_context=True,
            delta_target_path=delta_target_paths[split],
            delta_target_center=delta_target_stats["center"],
            delta_target_scale=delta_target_stats["scale"],
        )
        for split in datasets
    }
    seasonal_datasets = {split: NOxDataset(split, stats, load_images=False) for split in datasets}
    transport_eval_loaders = {
        split: _loader(dataset, shuffle=False, args=args, device=device)
        for split, dataset in transport_datasets.items()
    }
    seasonal_train_loader = _loader(seasonal_datasets["train"], shuffle=True, args=args, device=device)
    seasonal_eval_loaders = {
        split: _loader(dataset, shuffle=False, args=args, device=device) for split, dataset in seasonal_datasets.items()
    }
    class_weights = _class_weights(datasets["train"])
    class_weight_values = tuple(class_weights[name] for name in MODEL_CLASS_NAMES)
    common_checkpoint_metadata = {
        "normalization_stats": stats.to_dict(),
        "seasonal_feature_names": MODEL_FEATURE_NAMES,
        "target_name": MODEL_TARGET_COL,
        "class_names": MODEL_CLASS_NAMES,
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "pretrained_masked_model_sha256": masked_checkpoint_sha256,
        "loss": "training_prevalence_weighted_hard_cross_entropy",
        "class_weights": class_weights,
        "selection_metric": "validation_macro_f1",
        "selection_tiebreaker": "training_prevalence_weighted_validation_cross_entropy",
    }
    histories: dict[str, tuple[list[float], list[float]]] = {}
    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    training_summaries: dict[str, dict[str, object]] = {}
    ops_seasonal_dataset = NOxDataset(
        "ops",
        stats,
        manifest_path=args.ops_manifest_path,
        load_images=False,
    )
    ops_seasonal_loader = _loader(ops_seasonal_dataset, shuffle=False, args=args, device=device)
    ops_source_splits = sorted(ops_seasonal_dataset.frame["source_split"].astype(str).unique())

    seasonal_model = SeasonalMLP(len(MODEL_FEATURE_NAMES)).to(device)
    print(f"Training {seasonal_model.num_params():,}-parameter seasonal classifier on {device}")
    (
        seasonal_train_losses,
        seasonal_val_losses,
        seasonal_best_macro_f1,
        seasonal_checkpoint_loss,
    ) = fit_model(
        seasonal_model,
        seasonal_train_loader,
        seasonal_eval_loaders["val"],
        device=device,
        epochs=args.seasonal_epochs,
        learning_rate=args.seasonal_learning_rate,
        class_weight_values=class_weight_values,
        args=args,
        checkpoint_path=checkpoint_dir / "best_seasonal_classifier.pt",
        checkpoint_metadata=common_checkpoint_metadata,
        phase_name="Seasonal classifier",
    )
    seasonal_frames = {}
    for split, loader in seasonal_eval_loaders.items():
        logits, probabilities, indices, _ = run_inference(seasonal_model, loader, device)
        seasonal_frames[split] = _prediction_frame(seasonal_datasets[split], logits, probabilities, indices)
    logits, probabilities, indices, _ = run_inference(seasonal_model, ops_seasonal_loader, device)
    seasonal_frames["ops"] = _prediction_frame(ops_seasonal_dataset, logits, probabilities, indices)
    training_summaries["seasonal"] = {
        "maximum_epochs": args.seasonal_epochs,
        "learning_rate": args.seasonal_learning_rate,
        "parameters": seasonal_model.num_params(),
        "best_validation_macro_f1": seasonal_best_macro_f1,
        "checkpoint_validation_loss": seasonal_checkpoint_loss,
    }
    histories["seasonal"] = (seasonal_train_losses, seasonal_val_losses)
    model_frames["seasonal"] = seasonal_frames
    del seasonal_model, seasonal_train_loader, seasonal_eval_loaders, seasonal_datasets
    del ops_seasonal_dataset, ops_seasonal_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _seed_everything(args.seed)
    vision_model = JointVisionClassifier(
        len(MODEL_FEATURE_NAMES),
        model_width=args.model_width,
        head_dim=args.head_dim,
        dropout=args.dropout,
    ).to(device)
    vision_train_loader = _loader(transport_datasets["train"], shuffle=True, args=args, device=device)
    print(f"Training {vision_model.num_params():,}-parameter vision classifier")
    (
        vision_train_losses,
        vision_validation_losses,
        vision_best_macro_f1,
        vision_checkpoint_total_loss,
        vision_checkpoint_classification_loss,
        vision_checkpoint_raster_loss,
    ) = fit_joint_model(
        vision_model,
        vision_train_loader,
        transport_eval_loaders["val"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weight_values=class_weight_values,
        args=args,
        checkpoint_path=checkpoint_dir / "best_vision_classifier.pt",
        checkpoint_metadata={
            **common_checkpoint_metadata,
            "visual_fusion": "aligned_observation_and_transport_bottleneck_fusion",
            "context_fusion": "scale_and_shift_film_per_timestep",
            "context_dropout": DEFAULT_CONTEXT_DROPOUT,
            "model_width": args.model_width,
            "encoder_initialization": "random",
            "transport_residual_stats": transport_residual_stats,
            "transport_config": transport_config.to_metadata(),
            "loss": "weighted_hard_cross_entropy_plus_normalized_raster_l1",
            "classification_loss": "training_prevalence_weighted_hard_cross_entropy",
            "raster_loss": "per_sample_mean_absolute_error_on_train_normalized_target",
            "raster_loss_weight": args.raster_loss_weight,
            "selection_tiebreaker": "training_prevalence_weighted_validation_cross_entropy",
            "delta_target": {**delta_target_stats, **delta_target_metadata},
        },
    )
    vision_frames = {}
    shuffled_context_frames: dict[str, pd.DataFrame] = {}
    joint_sample_arrays: tuple[np.ndarray, ...] = ()
    for split, loader in transport_eval_loaders.items():
        logits, probabilities, indices, raster_predictions = run_inference(vision_model, loader, device)
        vision_frames[split] = _prediction_frame(transport_datasets[split], logits, probabilities, indices)
        if split == "test":
            joint_sample_arrays = _joint_prediction_sample_arrays(
                transport_datasets[split],
                indices,
                probabilities,
                raster_predictions,
                transport_residual_stats,
                args.seed,
            )
        permutation = np.random.default_rng(args.seed).permutation(len(transport_datasets[split]))
        shuffled_logits, shuffled_probabilities, shuffled_indices, _ = run_inference(
            vision_model,
            loader,
            device,
            context_permutation=permutation,
        )
        shuffled_context_frames[split] = _prediction_frame(
            transport_datasets[split],
            shuffled_logits,
            shuffled_probabilities,
            shuffled_indices,
        )
    ops_vision_dataset = NOxDataset(
        "ops",
        stats,
        manifest_path=args.ops_manifest_path,
        completed_raster_paths=completed_paths,
        source_image_stats=masked_image_stats,
        transport_transform=transport_transform,
        per_timestep_context=True,
    )
    ops_vision_loader = _loader(ops_vision_dataset, shuffle=False, args=args, device=device)
    logits, probabilities, indices, _ = run_inference(vision_model, ops_vision_loader, device)
    vision_frames["ops"] = _prediction_frame(ops_vision_dataset, logits, probabilities, indices)
    histories["vision"] = (vision_train_losses, vision_validation_losses)
    model_frames["vision"] = vision_frames
    training_summaries["vision"] = {
        "maximum_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "parameters": vision_model.num_params(),
        "model_width": args.model_width,
        "best_validation_macro_f1": vision_best_macro_f1,
        "checkpoint_validation_loss": vision_checkpoint_total_loss,
        "checkpoint_validation_classification_loss": vision_checkpoint_classification_loss,
        "checkpoint_validation_raster_l1": vision_checkpoint_raster_loss,
        "encoder_initialization": "random",
        "loss": "weighted_hard_cross_entropy_plus_normalized_raster_l1",
        "classification_loss": "training_prevalence_weighted_hard_cross_entropy",
        "raster_loss": "per_sample_mean_absolute_error_on_train_normalized_target",
        "raster_loss_weight": args.raster_loss_weight,
        "delta_target": {**delta_target_stats, **delta_target_metadata},
        "visual_fusion": "aligned_observation_and_transport_bottleneck_fusion",
        "context_fusion": "scale_and_shift_film_per_timestep",
        "context_dropout": DEFAULT_CONTEXT_DROPOUT,
    }
    del vision_model, vision_train_loader, ops_vision_dataset, ops_vision_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_config = {
        "device": str(device),
        "models": ["seasonal", "vision"],
        "loss": "weighted_hard_cross_entropy_plus_normalized_raster_l1",
        "classification_loss": "training_prevalence_weighted_hard_cross_entropy",
        "raster_loss": "per_sample_mean_absolute_error_on_train_normalized_target",
        "raster_loss_weight": args.raster_loss_weight,
        "class_weights": class_weights,
        "selection_metric": "validation_macro_f1",
        "selection_tiebreaker": "training_prevalence_weighted_validation_cross_entropy",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "training": training_summaries,
        "loss_history": {
            name: {"train": train_losses, "validation": validation_losses}
            for name, (train_losses, validation_losses) in histories.items()
        },
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
        "head_dim": args.head_dim,
        "model_width": args.model_width,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "seasonal_learning_rate": args.seasonal_learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "early_stop_patience": args.early_stop_patience,
        "pretrained_masked_model_weights": str(masked_checkpoint_path),
        "pretrained_masked_model_sha256": masked_checkpoint_sha256,
        "pretrained_masked_model_role": "missing_pixel_completion_only",
        "encoder_initialization": "random",
        "completed_raster_paths": {split: str(path) for split, path in completed_paths.items()},
        "ops_manifest_path": str(args.ops_manifest_path),
        "ops_source_splits": ops_source_splits,
        "delta_target_paths": {split: str(path) for split, path in delta_target_paths.items()},
        "completed_raster_keys": list(MODEL_IMAGE_KEYS + MODEL_MASK_KEYS),
        "image_keys": list(stats.image_keys),
        "quality_keys": ["no2_mask", "transport_residual_quality"],
        "image_center": list(stats.image_center),
        "image_scale": list(stats.image_scale),
        "raster_normalization": "global_training_split_statistics",
        "no2_normalization": "training_split_median_robust_scale",
        "weather_normalization": "training_split_mean_standard_deviation",
        "image_clip_range": [-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS],
        "target_name": MODEL_TARGET_COL,
        "class_names": list(MODEL_CLASS_NAMES),
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "class_counts": {split: _class_counts(dataset) for split, dataset in datasets.items()},
        "seasonal_features": list(MODEL_FEATURE_NAMES),
        "seasonal_feature_timestamp": "sequence_midpoint_local_solar_time",
        "vision_context_timestamps": "each_observation_local_solar_time_rounded_to_nearest_hour",
        "prediction_family": "joint_three_class_and_12x12_delta_raster",
        "delta_target": {
            **delta_target_stats,
            **delta_target_metadata,
            "normalization_fit_split": "train",
            "loss": "unweighted_L1",
        },
        "transport": {
            "execution": "cpu_dataloader_transform",
            "method": "backward_semi_lagrangian_bilinear",
            "wind": "endpoint_blended_80m_wind",
            "residual": "current_minus_advected_prior",
            "residual_quality": "current_no2_quality_times_warped_source_no2_quality",
            "residual_normalization": transport_residual_stats,
            **transport_config.to_metadata(),
        },
        "context_fusion": {
            "location": "branch_spatial_bottlenecks_at_each_timestep",
            "method": "scale_and_shift_film",
            "parameterization": "gamma_equals_one_plus_delta_gamma_beta_equals_delta_beta",
            "initial_timestep_context": True,
            "identity_initialized": True,
            "additive_shift": True,
            "context_dropout": DEFAULT_CONTEXT_DROPOUT,
        },
    }
    with (run_dir / "run_config.json").open("w") as destination:
        json.dump(run_config, destination, indent=2)

    shuffled_context_results = {
        split: classification_metrics(frame) for split, frame in shuffled_context_frames.items()
    }
    with (run_dir / "shuffled_context_results.json").open("w") as destination:
        json.dump(shuffled_context_results, destination, indent=2)
    for split, frame in shuffled_context_frames.items():
        frame.to_csv(run_dir / f"vision_shuffled_{split}_predictions.csv", index=False)

    plot_split_class_f1(model_frames, run_dir)
    plot_training_curves(histories, run_dir)
    plot_accuracy_by_characteristic(model_frames, run_dir)
    plot_joint_prediction_samples(*joint_sample_arrays, run_dir=run_dir)
    plot_ops_evaluation(model_frames, run_dir)
    save_results(model_frames, run_dir, primary_model_name="vision")


if __name__ == "__main__":
    main()
