"""Train seasonal and transport emissions-change classifiers."""

import argparse
import hashlib
import json
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
    MAX_CONTEXT_SCALE_DELTA,
    TransportContextClassifier,
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
from modeling.plot_utils import plot_accuracy_by_characteristic, plot_split_class_accuracy, plot_training_curves
from modeling.transport import AdvectionTransform
from torch import nn
from torch.utils.data import DataLoader

from config import (
    DATASET_AOI_NORMALIZATION_STATS,
    MODEL_CLASS_NAMES,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_IMAGE_KEYS,
    MODEL_INPUT_CHANNELS,
    MODEL_MASK_KEYS,
    MODEL_TARGET_COL,
    NUM_CORES,
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
    parser.add_argument("--pretrained-masked-model-weights", default=PRETRAINED_MASKED_MODEL_WEIGHTS)
    parser.add_argument("--completed-raster-dir", required=True)
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
    image, seasonal, transport, target, index = batch
    non_blocking = device.type == "cuda"
    return (
        image.to(device, non_blocking=non_blocking),
        seasonal.to(device, non_blocking=non_blocking),
        transport.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
        index,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    """Train one epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        image, seasonal, transport, target, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(image, seasonal, transport)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * target.numel()
        correct += int((logits.argmax(dim=1) == target).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch and return mean loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, seasonal, transport, target, _ = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image, seasonal, transport)
                loss = criterion(logits, target)
            total_loss += loss.item() * target.numel()
            correct += int((logits.argmax(dim=1) == target).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, object],
    phase_name: str,
) -> tuple[list[float], list[float], float]:
    """Fit one classifier and restore its lowest-validation-loss state."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.gradient_clip_norm,
        )
        validation_loss, validation_accuracy = val_epoch(model, val_loader, criterion, device)
        scheduler.step(validation_loss)
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        learning_rate_now = optimizer.param_groups[0]["lr"]
        print(
            f"{phase_name} epoch {epoch:03d} | train loss {train_loss:.5f} acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.5f} acc {validation_accuracy:.4f} | lr {learning_rate_now:.2e}"
        )
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
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
    return train_losses, val_losses, best_val_loss


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


def _fit_transport_residual_stats(
    loader: DataLoader,
    stats: NormalizationStats,
) -> dict[str, float]:
    # Fit quality-weighted physical residual moments on the training split
    transport = AdvectionTransform(stats.image_center, stats.image_scale)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate logits and probabilities for one dataset split."""
    model.eval()
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
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
                logits = model(image, seasonal, transport)
            logits = logits.float()
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(logits.softmax(dim=1).cpu().numpy())
            indices.append(index.numpy())
    return np.concatenate(logits_batches), np.concatenate(probability_batches), np.concatenate(indices)


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


def _class_counts(dataset: NOxDataset) -> dict[str, int]:
    # Count training examples in the configured class order
    return {name: int(np.sum(dataset.labels == index)) for index, name in enumerate(MODEL_CLASS_NAMES)}


def main() -> None:
    """Train and evaluate seasonal and transport classifiers."""
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)
    completed_root = Path(args.completed_raster_dir).resolve()
    if Path("/tmp") not in completed_root.parents:
        raise ValueError("Completed raster directory must be under job-local /tmp")
    masked_checkpoint_path = Path(args.pretrained_masked_model_weights)
    if not args.pretrained_masked_model_weights:
        raise ValueError("Set PRETRAINED_MASKED_MODEL_WEIGHTS or pass --pretrained-masked-model-weights")

    run_name = datetime.now(timezone.utc).strftime("delta_category_classification_%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    print(f"Training seasonal and transport classifiers on {device}; outputs: {run_dir}")

    masked_model_checkpoint, imputation_model = _load_masked_checkpoint(masked_checkpoint_path, device)
    stats = compute_stats("train", fixed_image_stats=masked_model_checkpoint["normalization_stats"])
    del masked_model_checkpoint
    save_stats(stats, run_dir / "normalization_stats.json")

    source_datasets = {split: NOxDataset(split, stats, aoi_normalize_no2=False) for split in ("train", "val", "test")}
    masked_checkpoint_sha256 = _checkpoint_sha256(masked_checkpoint_path)
    completed_dir = completed_root / run_name
    completed_paths = _fill_missing_rasters(
        imputation_model,
        source_datasets,
        output_dir=completed_dir,
        args=args,
        device=device,
    )
    del imputation_model, source_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    datasets = {
        split: NOxDataset(split, stats, completed_raster_path=completed_paths[split])
        for split in ("train", "val", "test")
    }
    transport_stats_loader = _loader(
        datasets["train"],
        shuffle=False,
        args=args,
        device=torch.device("cpu"),
    )
    transport_residual_stats = _fit_transport_residual_stats(transport_stats_loader, stats)
    del transport_stats_loader
    transport_transform = AdvectionTransform(
        stats.image_center,
        stats.image_scale,
        residual_center=transport_residual_stats["center"],
        residual_scale=transport_residual_stats["scale"],
    )
    transport_datasets = {
        split: NOxDataset(
            split,
            stats,
            completed_raster_path=completed_paths[split],
            transport_transform=transport_transform,
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
    common_checkpoint_metadata = {
        "normalization_stats": stats.to_dict(),
        "seasonal_feature_names": MODEL_FEATURE_NAMES,
        "target_name": MODEL_TARGET_COL,
        "class_names": MODEL_CLASS_NAMES,
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "pretrained_masked_model_sha256": masked_checkpoint_sha256,
    }
    histories: dict[str, tuple[list[float], list[float]]] = {}
    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    training_summaries: dict[str, dict[str, object]] = {}

    seasonal_model = SeasonalMLP(len(MODEL_FEATURE_NAMES)).to(device)
    print(f"Training {seasonal_model.num_params():,}-parameter seasonal classifier on {device}")
    seasonal_train_losses, seasonal_val_losses, seasonal_best_loss = fit_model(
        seasonal_model,
        seasonal_train_loader,
        seasonal_eval_loaders["val"],
        device=device,
        epochs=args.seasonal_epochs,
        learning_rate=args.seasonal_learning_rate,
        args=args,
        checkpoint_path=checkpoint_dir / "best_seasonal_classifier.pt",
        checkpoint_metadata=common_checkpoint_metadata,
        phase_name="Seasonal classifier",
    )
    seasonal_frames = {}
    for split, loader in seasonal_eval_loaders.items():
        logits, probabilities, indices = run_inference(seasonal_model, loader, device)
        seasonal_frames[split] = _prediction_frame(seasonal_datasets[split], logits, probabilities, indices)
    training_summaries["seasonal"] = {
        "maximum_epochs": args.seasonal_epochs,
        "learning_rate": args.seasonal_learning_rate,
        "parameters": seasonal_model.num_params(),
        "best_validation_loss": seasonal_best_loss,
    }
    histories["seasonal"] = (seasonal_train_losses, seasonal_val_losses)
    model_frames["seasonal"] = seasonal_frames
    del seasonal_model, seasonal_train_loader, seasonal_eval_loaders, seasonal_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_factories = (
        (
            "transport_image_only",
            "Transport image-only classifier",
            lambda: TransportContextClassifier(
                len(MODEL_FEATURE_NAMES),
                condition_on_context=False,
                model_width=args.model_width,
                head_dim=args.head_dim,
                dropout=args.dropout,
            ),
            {
                "fusion": "aligned_observation_and_transport_bottleneck_fusion",
                "context_conditioning": "none",
            },
        ),
        (
            "transport_context",
            "Transport context-conditioned classifier",
            lambda: TransportContextClassifier(
                len(MODEL_FEATURE_NAMES),
                condition_on_context=True,
                model_width=args.model_width,
                head_dim=args.head_dim,
                dropout=args.dropout,
            ),
            {
                "fusion": "aligned_observation_and_transport_bottleneck_fusion",
                "context_conditioning": "bounded_scale_only_bottleneck_residual",
                "context_direct_to_classifier": False,
            },
        ),
    )
    shuffled_context_frames: dict[str, pd.DataFrame] = {}
    for model_name, phase_name, factory, model_metadata in model_factories:
        _seed_everything(args.seed)
        model = factory().to(device)
        train_loader = _loader(transport_datasets["train"], shuffle=True, args=args, device=device)
        print(f"Training {model.num_params():,}-parameter {phase_name.lower()}")
        train_losses, validation_losses, best_loss = fit_model(
            model,
            train_loader,
            transport_eval_loaders["val"],
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            args=args,
            checkpoint_path=checkpoint_dir / f"best_{model_name}_classifier.pt",
            checkpoint_metadata={
                **common_checkpoint_metadata,
                **model_metadata,
                "model_width": args.model_width,
                "encoder_initialization": "random",
                "transport_residual_stats": transport_residual_stats,
                "loss": "unweighted_hard_cross_entropy",
            },
            phase_name=phase_name,
        )
        split_frames = {}
        for split, loader in transport_eval_loaders.items():
            logits, probabilities, indices = run_inference(model, loader, device)
            split_frames[split] = _prediction_frame(transport_datasets[split], logits, probabilities, indices)
            if model_name == "transport_context":
                permutation = np.random.default_rng(args.seed).permutation(len(transport_datasets[split]))
                shuffled_logits, shuffled_probabilities, shuffled_indices = run_inference(
                    model,
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
        histories[model_name] = (train_losses, validation_losses)
        model_frames[model_name] = split_frames
        training_summaries[model_name] = {
            "maximum_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "parameters": model.num_params(),
            "model_width": args.model_width,
            "best_validation_loss": best_loss,
            "encoder_initialization": "random",
            "loss": "unweighted_hard_cross_entropy",
            **model_metadata,
        }
        del model, train_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_config = {
        "device": str(device),
        "models": ["seasonal", "transport_image_only", "transport_context"],
        "loss": "unweighted_hard_cross_entropy",
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
        "completed_raster_keys": list(MODEL_IMAGE_KEYS + MODEL_MASK_KEYS),
        "image_keys": list(stats.image_keys),
        "quality_keys": ["no2_mask", "transport_residual_quality"],
        "image_center": list(stats.image_center),
        "image_scale": list(stats.image_scale),
        "no2_normalization": "aoi_specific_median_robust_scale",
        "aoi_normalization_stats": DATASET_AOI_NORMALIZATION_STATS,
        "weather_normalization": "global_pretrained_model_stats",
        "image_clip_range": [-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS],
        "target_name": MODEL_TARGET_COL,
        "class_names": list(MODEL_CLASS_NAMES),
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "class_counts": {split: _class_counts(dataset) for split, dataset in datasets.items()},
        "seasonal_features": list(MODEL_FEATURE_NAMES),
        "seasonal_feature_timestamp": "sequence_midpoint_local_solar_time",
        "prediction_family": "three_class_categorical_distribution",
        "transport": {
            "execution": "cpu_dataloader_transform",
            "method": "backward_semi_lagrangian_bilinear",
            "wind": "interval_mean_80m_wind",
            "residual": "current_minus_advected_prior",
            "decay_enabled": False,
            "elevated_wind_enabled": False,
            "residual_quality": "current_no2_quality_times_warped_source_no2_quality",
            "residual_normalization": transport_residual_stats,
        },
        "context_conditioning": {
            "location": "spatial_bottleneck",
            "method": "bounded_scale_only_residual_modulation",
            "identity_initialized": True,
            "direct_classifier_path": False,
            "context_dropout": DEFAULT_CONTEXT_DROPOUT,
            "maximum_scale_delta": MAX_CONTEXT_SCALE_DELTA,
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
        frame.to_csv(run_dir / f"transport_context_shuffled_{split}_predictions.csv", index=False)

    plot_split_class_accuracy(model_frames, run_dir)
    plot_training_curves(histories, run_dir)
    plot_accuracy_by_characteristic(model_frames, run_dir)
    save_results(model_frames, run_dir, primary_model_name="transport_context")


if __name__ == "__main__":
    main()
