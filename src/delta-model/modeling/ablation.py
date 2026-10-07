"""Run fixed-hyperparameter input ablations for the vision classifier."""

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch
from modeling.convgru import DEFAULT_CONTEXT_DROPOUT, VisionClassifier
from modeling.dataset import MODEL_FEATURE_NAMES, NormalizationStats, NOxDataset, compute_stats, save_stats
from modeling.eval_utils import classification_metrics
from modeling.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORKERS,
    PIXEL_FILL_METADATA,
    _class_weights,
    _device,
    _fill_missing_rasters,
    _fit_transport_residual_stats,
    _loader,
    _prediction_frame,
    _seed_everything,
    fit_model,
    run_inference,
)
from modeling.transport import TransportConfig, TransportTransform
from torch import nn
from torch.utils.data import DataLoader

from config import (
    MODEL_CLASS_NAMES,
    MODEL_IMAGE_KEYS,
    MODEL_TARGET_COL,
    NUM_CORES,
    RUNS_DIR,
    SEQUENCE_TIMESTEPS,
)

MODEL_WIDTH = 64
HEAD_DIM = 48
DROPOUT = 0.20
WIND_CHANNELS = (
    MODEL_IMAGE_KEYS.index("wind_u_80m_mps"),
    MODEL_IMAGE_KEYS.index("wind_v_80m_mps"),
)
TEMPERATURE_CHANNELS = (MODEL_IMAGE_KEYS.index("temperature_2m_k"),)


@dataclass(frozen=True)
class _AblationSpec:
    name: str
    description: str
    image_channels: tuple[int, ...] = ()
    zero_transport: bool = False
    zero_context: bool = False
    zero_timestep: int | None = None


@dataclass(frozen=True)
class _ExperimentResult:
    name: str
    description: str
    best_validation_macro_f1: float
    checkpoint_validation_loss: float
    validation_metrics: dict[str, object]
    test_metrics: dict[str, object]


ABLATIONS = (
    _AblationSpec("baseline", "All production inputs"),
    _AblationSpec(
        "no_wind_rasters",
        "Zero wind-u and wind-v observation rasters; retain wind-derived transport residuals",
        image_channels=WIND_CHANNELS,
    ),
    _AblationSpec(
        "no_temperature_rasters",
        "Zero temperature observation rasters",
        image_channels=TEMPERATURE_CHANNELS,
    ),
    _AblationSpec(
        "no_transport_residuals",
        "Zero transport residual and residual-quality rasters",
        zero_transport=True,
    ),
    _AblationSpec(
        "no_temporal_tabular_features",
        "Zero midpoint day-of-year and local-solar-hour features",
        zero_context=True,
    ),
    *tuple(
        _AblationSpec(
            f"no_t{timestep}_no2_weather",
            f"Zero all observation channels at t{timestep}; retain the transport branch",
            zero_timestep=timestep,
        )
        for timestep in range(SEQUENCE_TIMESTEPS)
    ),
)


class _AblatedVisionClassifier(VisionClassifier):
    def __init__(self, spec: _AblationSpec) -> None:
        super().__init__(
            len(MODEL_FEATURE_NAMES),
            model_width=MODEL_WIDTH,
            head_dim=HEAD_DIM,
            dropout=DROPOUT,
        )
        self.spec = spec

    def _context_embedding(self, context: torch.Tensor) -> torch.Tensor:
        # Remove the complete context branch for the temporal-feature ablation
        if self.spec.zero_context:
            return torch.zeros_like(self.context_encoder(context))
        return super()._context_embedding(context)

    def forward(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
        transport: torch.Tensor,
    ) -> torch.Tensor:
        """Classify a batch after applying the configured input ablation."""
        if self.spec.image_channels or self.spec.zero_timestep is not None:
            image = image.clone()
        if self.spec.image_channels:
            image[:, :, self.spec.image_channels] = 0.0
        if self.spec.zero_timestep is not None:
            image[:, self.spec.zero_timestep] = 0.0
        if self.spec.zero_transport:
            transport = torch.zeros_like(transport)
        return super().forward(image, context, transport)


def parse_args() -> argparse.Namespace:
    """Parse ablation job locations and runtime options.

    Returns:
        Parsed arguments with production training hyperparameters attached.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-raster-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.set_defaults(
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        workers=min(DEFAULT_WORKERS, NUM_CORES),
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        seed=DEFAULT_SEED,
        model_width=MODEL_WIDTH,
        head_dim=HEAD_DIM,
        dropout=DROPOUT,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM,
        scheduler_patience=DEFAULT_SCHEDULER_PATIENCE,
        scheduler_factor=DEFAULT_SCHEDULER_FACTOR,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
    )
    return parser.parse_args()


def _prepare_datasets(
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> tuple[
    dict[str, NOxDataset],
    NormalizationStats,
    TransportConfig,
    dict[str, float],
    dict[str, Path],
]:
    # Build shared interpolation and transport inputs once
    stats = compute_stats("train")
    save_stats(stats, run_dir / "normalization_stats.json")
    source_datasets = {split: NOxDataset(split, stats) for split in ("train", "val", "test")}
    completed_paths = _fill_missing_rasters(
        source_datasets,
        output_dir=Path(args.completed_raster_dir) / run_dir.name,
        args=args,
    )
    del source_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    completed_datasets = {
        split: NOxDataset(split, stats, completed_raster_path=completed_paths[split])
        for split in ("train", "val", "test")
    }
    transport_config = TransportConfig()
    stats_loader = _loader(
        completed_datasets["train"],
        shuffle=False,
        args=args,
        device=torch.device("cpu"),
    )
    transport_stats = _fit_transport_residual_stats(stats_loader, stats, transport_config)
    del stats_loader
    transform = TransportTransform(
        stats.image_center,
        stats.image_scale,
        residual_center=transport_stats["center"],
        residual_scale=transport_stats["scale"],
        config=transport_config,
    )
    datasets = {
        split: NOxDataset(
            split,
            stats,
            completed_raster_path=completed_paths[split],
            transport_transform=transform,
        )
        for split in completed_datasets
    }
    return datasets, stats, transport_config, transport_stats, completed_paths


def _evaluate(
    model: nn.Module,
    dataset: NOxDataset,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    # Calculate classification metrics from one fixed checkpoint
    logits, probabilities, indices = run_inference(model, loader, device)
    frame = _prediction_frame(dataset, logits, probabilities, indices)
    return classification_metrics(frame)


def _format_summary(
    results: list[_ExperimentResult],
    run_dir: Path,
    args: argparse.Namespace,
) -> str:
    # Render a compact report plus complete test confusion matrices
    baseline_macro_f1 = float(results[0].test_metrics["macro_f1"])
    lines = [
        "NO2 vision-model ablation results",
        f"Run: {run_dir}",
        "Selection: best validation macro F1; test evaluated after checkpoint selection",
        (
            f"Training: seed={args.seed}, epochs={args.epochs}, batch={args.batch_size}, width={MODEL_WIDTH}, "
            f"head={HEAD_DIM}, dropout={DROPOUT:g}, learning_rate={args.learning_rate:g}, "
            f"weight_decay={args.weight_decay:g}"
        ),
        "Zeroed normalized raster values correspond to the training mean.",
        "",
        (
            f"{'experiment':<34} {'val_f1':>8} {'test_f1':>8} {'delta_f1':>9} {'bal_acc':>8} "
            f"{'rec_dec':>8} {'rec_std':>8} {'rec_inc':>8}"
        ),
        "-" * 107,
    ]
    for result in results:
        test = result.test_metrics
        recall = cast(dict[str, float], test["class_recall"])
        test_macro_f1 = float(test["macro_f1"])
        lines.append(
            f"{result.name:<34} {result.best_validation_macro_f1:8.4f} {test_macro_f1:8.4f} "
            f"{test_macro_f1 - baseline_macro_f1:+9.4f} {float(test['balanced_accuracy']):8.4f} "
            f"{recall['decrease']:8.4f} {recall['steady']:8.4f} {recall['increase']:8.4f}"
        )

    lines.extend(("", "Experiment definitions"))
    lines.extend(f"- {result.name}: {result.description}" for result in results)
    lines.extend(("", "Test predicted-class frequencies and confusion matrices"))
    for result in results:
        frequencies = cast(dict[str, float], result.test_metrics["predicted_class_frequencies"])
        confusion = cast(list[list[int]], result.test_metrics["confusion_matrix"])
        lines.extend(
            (
                "",
                result.name,
                (
                    "predicted frequencies: "
                    + ", ".join(f"{name}={frequencies[name]:.4f}" for name in MODEL_CLASS_NAMES)
                ),
                "confusion matrix rows=true and columns=predicted [decrease, steady, increase]:",
                *("  " + " ".join(f"{value:6d}" for value in row) for row in confusion),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Train the baseline and every fixed-input ablation."""
    args = parse_args()
    device = _device(args.device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / f"input_ablation_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    print(f"Running ablation study on {device}; outputs: {run_dir}")

    datasets, stats, transport_config, transport_stats, completed_paths = _prepare_datasets(args, run_dir, device)
    class_weights = _class_weights(datasets["train"])
    class_weight_values = tuple(class_weights[name] for name in MODEL_CLASS_NAMES)
    eval_loaders = {
        split: _loader(datasets[split], shuffle=False, args=args, device=device) for split in ("val", "test")
    }
    common_metadata = {
        "normalization_stats": stats.to_dict(),
        "seasonal_feature_names": MODEL_FEATURE_NAMES,
        "target_name": MODEL_TARGET_COL,
        "class_names": MODEL_CLASS_NAMES,
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "pixel_fill": PIXEL_FILL_METADATA,
        "raster_normalization": "training_split_zscore",
        "loss": "training_prevalence_weighted_hard_cross_entropy",
        "class_weights": class_weights,
        "selection_metric": "validation_macro_f1",
        "model_width": MODEL_WIDTH,
        "head_dim": HEAD_DIM,
        "dropout": DROPOUT,
        "context_dropout": DEFAULT_CONTEXT_DROPOUT,
        "transport_residual_stats": transport_stats,
        "transport_config": transport_config.to_metadata(),
        "completed_raster_paths": {split: str(path) for split, path in completed_paths.items()},
    }
    results: list[_ExperimentResult] = []
    summary_path = run_dir / "ablation_summary.txt"

    for experiment_number, spec in enumerate(ABLATIONS, start=1):
        print(f"Ablation {experiment_number}/{len(ABLATIONS)}: {spec.name}")
        _seed_everything(args.seed)
        model = _AblatedVisionClassifier(spec).to(device)
        train_loader = _loader(datasets["train"], shuffle=True, args=args, device=device)
        _, _, best_validation_macro_f1, checkpoint_validation_loss = fit_model(
            model,
            train_loader,
            eval_loaders["val"],
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            class_weight_values=class_weight_values,
            args=args,
            checkpoint_path=checkpoint_dir / f"{spec.name}.pt",
            checkpoint_metadata={
                **common_metadata,
                "experiment": spec.name,
                "ablation": spec.description,
                "test_split_evaluated": False,
            },
            phase_name=spec.name,
        )
        validation_metrics = _evaluate(model, datasets["val"], eval_loaders["val"], device)
        test_metrics = _evaluate(model, datasets["test"], eval_loaders["test"], device)
        results.append(
            _ExperimentResult(
                name=spec.name,
                description=spec.description,
                best_validation_macro_f1=best_validation_macro_f1,
                checkpoint_validation_loss=checkpoint_validation_loss,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
            )
        )
        summary_path.write_text(_format_summary(results, run_dir, args))
        del model, train_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
