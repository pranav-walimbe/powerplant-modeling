"""Optimize joint vision-model hyperparameters on the validation split."""

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
from modeling.convgru import JointVisionClassifier
from modeling.dataset import (
    MODEL_FEATURE_NAMES,
    NormalizationStats,
    NOxDataset,
    compute_stats,
    contribution_target_metadata,
    materialize_contribution_targets,
    save_stats,
)
from modeling.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_SINKHORN_ITERATIONS,
    DEFAULT_WORKERS,
    _checkpoint_sha256,
    _class_counts,
    _class_weights,
    _device,
    _fill_missing_rasters,
    _fit_transport_residual_stats,
    _load_masked_checkpoint,
    _loader,
    _seed_everything,
    fit_joint_model,
)
from modeling.transport import TransportConfig, TransportTransform

from config import (
    MODEL_CLASS_NAMES,
    MODEL_TARGET_COL,
    NUM_CORES,
    PRETRAINED_MASKED_MODEL_WEIGHTS,
    RUNS_DIR,
    SEQUENCE_TIMESTEPS,
)

DEFAULT_TRIALS = 24
FIXED_MODEL_WIDTH = 64
FIXED_HEAD_DIM = 48
LEARNING_RATE_RANGE = (1e-5, 1e-4)
WEIGHT_DECAY_CHOICES = (1e-4, 1e-3)
DROPOUT_CHOICES = (0.2, 0.4)
RASTER_LOSS_WEIGHT_CHOICES = (0.5, 1.0)
CONTRIBUTION_SPLAT_SIGMA_CHOICES = (1.0, 1.5)
SINKHORN_BLUR_CHOICES = (1.0, 1.5)
SINKHORN_REACH_CHOICES = (3.0, 6.0)
TPE_STARTUP_TRIALS = 6
PRUNING_WARMUP_EPOCHS = 3


@dataclass(frozen=True)
class _PreparedSearch:
    datasets_by_sigma: dict[float, dict[str, NOxDataset]]
    class_weight_values: tuple[float, ...]
    normalization_stats: NormalizationStats
    transport_config: TransportConfig
    transport_residual_stats: dict[str, float]
    masked_checkpoint_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse hyperparameter-search options.

    Returns:
        Parsed command-line arguments with fixed training controls attached.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", default="contribution-joint-hpo")
    parser.add_argument("--study-dir")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, NUM_CORES))
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pretrained-masked-model-weights", default=PRETRAINED_MASKED_MODEL_WEIGHTS)
    parser.add_argument("--completed-raster-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.set_defaults(
        model_width=FIXED_MODEL_WIDTH,
        head_dim=FIXED_HEAD_DIM,
        gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM,
        scheduler_patience=DEFAULT_SCHEDULER_PATIENCE,
        scheduler_factor=DEFAULT_SCHEDULER_FACTOR,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
        sinkhorn_iterations=DEFAULT_SINKHORN_ITERATIONS,
    )
    return parser.parse_args()


def _resolve_study_dir(requested: str | None) -> Path:
    # Create a persistent study directory or resume the requested one
    if requested:
        study_dir = Path(requested).resolve()
        study_dir.mkdir(parents=True, exist_ok=True)
        return study_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    study_dir = Path(RUNS_DIR) / f"contribution_hpo_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=False)
    return study_dir


def _prepare_search(args: argparse.Namespace, study_dir: Path, device: torch.device) -> _PreparedSearch:
    # Materialize shared train and validation inputs once for all trials
    completed_root = Path(args.completed_raster_dir).resolve()
    if Path("/tmp") not in completed_root.parents:
        raise ValueError("Completed raster directory must be under job-local /tmp")
    if not args.pretrained_masked_model_weights:
        raise ValueError("Set PRETRAINED_MASKED_MODEL_WEIGHTS or pass --pretrained-masked-model-weights")
    masked_checkpoint_path = Path(args.pretrained_masked_model_weights)
    masked_checkpoint, imputation_model = _load_masked_checkpoint(masked_checkpoint_path, device)
    masked_image_stats = masked_checkpoint["normalization_stats"]
    masked_checkpoint_sha256 = _checkpoint_sha256(masked_checkpoint_path)
    stats = compute_stats("train")
    save_stats(stats, study_dir / "normalization_stats.json")

    source_datasets = {
        split: NOxDataset(split, stats, source_image_stats=masked_image_stats) for split in ("train", "val")
    }
    completed_dir = completed_root / study_dir.name
    completed_paths = _fill_missing_rasters(
        imputation_model,
        source_datasets,
        output_dir=completed_dir,
        args=args,
        device=device,
    )
    del imputation_model, masked_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    target_paths_by_sigma = {
        sigma: materialize_contribution_targets(
            source_datasets,
            completed_dir / f"contribution-targets-sigma-{sigma:g}",
            sigma=sigma,
        )
        for sigma in CONTRIBUTION_SPLAT_SIGMA_CHOICES
    }

    base_datasets = {
        split: NOxDataset(
            split,
            stats,
            completed_raster_path=completed_paths[split],
            source_image_stats=masked_image_stats,
        )
        for split in ("train", "val")
    }
    transport_config = TransportConfig()
    transport_stats_loader = _loader(
        base_datasets["train"],
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
    datasets_by_sigma = {
        sigma: {
            split: NOxDataset(
                split,
                stats,
                completed_raster_path=completed_paths[split],
                source_image_stats=masked_image_stats,
                transport_transform=transport_transform,
                per_timestep_context=True,
                contribution_target_path=target_paths[split],
            )
            for split in ("train", "val")
        }
        for sigma, target_paths in target_paths_by_sigma.items()
    }
    class_weights = _class_weights(base_datasets["train"])
    search_config = {
        "objective": "best_validation_macro_f1",
        "test_split_loaded": False,
        "sampler": "seeded_tpe",
        "pruner": "median",
        "tpe_startup_trials": TPE_STARTUP_TRIALS,
        "pruning_warmup_epochs": PRUNING_WARMUP_EPOCHS,
        "seed": args.seed,
        "trials_requested": args.trials,
        "search_space": {
            "learning_rate": {"low": LEARNING_RATE_RANGE[0], "high": LEARNING_RATE_RANGE[1], "log": True},
            "weight_decay": list(WEIGHT_DECAY_CHOICES),
            "dropout": list(DROPOUT_CHOICES),
            "raster_loss_weight": list(RASTER_LOSS_WEIGHT_CHOICES),
            "contribution_splat_sigma": list(CONTRIBUTION_SPLAT_SIGMA_CHOICES),
            "sinkhorn_blur_cells": list(SINKHORN_BLUR_CHOICES),
            "sinkhorn_reach_cells": list(SINKHORN_REACH_CHOICES),
        },
        "contribution_target_contracts": {
            str(sigma): contribution_target_metadata(sigma) for sigma in CONTRIBUTION_SPLAT_SIGMA_CHOICES
        },
        "fixed": {
            "model_width": FIXED_MODEL_WIDTH,
            "head_dim": FIXED_HEAD_DIM,
            "sinkhorn_iterations": DEFAULT_SINKHORN_ITERATIONS,
            "early_stop_patience": DEFAULT_EARLY_STOP_PATIENCE,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
        },
        "class_counts": {split: _class_counts(dataset) for split, dataset in base_datasets.items()},
        "class_weights": class_weights,
        "normalization_fit_split": "train",
        "transport": {
            "residual_normalization": transport_residual_stats,
            **transport_config.to_metadata(),
        },
        "pretrained_masked_model_weights": str(masked_checkpoint_path),
        "pretrained_masked_model_sha256": masked_checkpoint_sha256,
    }
    with (study_dir / "search_config.json").open("w") as destination:
        json.dump(search_config, destination, indent=2)
    return _PreparedSearch(
        datasets_by_sigma=datasets_by_sigma,
        class_weight_values=tuple(class_weights[name] for name in MODEL_CLASS_NAMES),
        normalization_stats=stats,
        transport_config=transport_config,
        transport_residual_stats=transport_residual_stats,
        masked_checkpoint_sha256=masked_checkpoint_sha256,
    )


def _suggest_parameters(trial: optuna.Trial) -> dict[str, float]:
    # Sample the focused search space
    return {
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE_RANGE, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", WEIGHT_DECAY_CHOICES),
        "dropout": trial.suggest_categorical("dropout", DROPOUT_CHOICES),
        "raster_loss_weight": trial.suggest_categorical("raster_loss_weight", RASTER_LOSS_WEIGHT_CHOICES),
        "contribution_splat_sigma": trial.suggest_categorical(
            "contribution_splat_sigma", CONTRIBUTION_SPLAT_SIGMA_CHOICES
        ),
        "sinkhorn_blur_cells": trial.suggest_categorical("sinkhorn_blur_cells", SINKHORN_BLUR_CHOICES),
        "sinkhorn_reach_cells": trial.suggest_categorical("sinkhorn_reach_cells", SINKHORN_REACH_CHOICES),
    }


def _objective(
    trial: optuna.Trial,
    *,
    args: argparse.Namespace,
    study_dir: Path,
    device: torch.device,
    prepared: _PreparedSearch,
) -> float:
    # Train one fixed-seed vision model and select only on validation macro F1
    parameters = _suggest_parameters(trial)
    trial_args = argparse.Namespace(**{**vars(args), **parameters})
    sigma = parameters["contribution_splat_sigma"]
    datasets = prepared.datasets_by_sigma[sigma]
    _seed_everything(args.seed)
    model = JointVisionClassifier(
        len(MODEL_FEATURE_NAMES),
        model_width=FIXED_MODEL_WIDTH,
        head_dim=FIXED_HEAD_DIM,
        dropout=parameters["dropout"],
    ).to(device)
    train_loader = _loader(datasets["train"], shuffle=True, args=trial_args, device=device)
    val_loader = _loader(datasets["val"], shuffle=False, args=trial_args, device=device)
    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = trial_dir / "best_vision_classifier.pt"
    best_reported = float("-inf")

    def report_validation(epoch: int, validation_macro_f1: float) -> None:
        nonlocal best_reported
        best_reported = max(best_reported, validation_macro_f1)
        trial.report(best_reported, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {epoch} with best validation macro F1 {best_reported:.4f}")

    try:
        (
            train_losses,
            validation_losses,
            best_validation_macro_f1,
            checkpoint_total_loss,
            checkpoint_classification_loss,
            checkpoint_raster_loss,
        ) = fit_joint_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=parameters["learning_rate"],
            class_weight_values=prepared.class_weight_values,
            args=trial_args,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata={
                "trial_number": trial.number,
                "hyperparameters": parameters,
                "normalization_stats": prepared.normalization_stats.to_dict(),
                "target_name": MODEL_TARGET_COL,
                "class_names": MODEL_CLASS_NAMES,
                "sequence_timesteps": SEQUENCE_TIMESTEPS,
                "pretrained_masked_model_sha256": prepared.masked_checkpoint_sha256,
                "transport_residual_stats": prepared.transport_residual_stats,
                "transport_config": prepared.transport_config.to_metadata(),
                "contribution_target": contribution_target_metadata(sigma),
                "raster_loss": {
                    "name": "debiased_unbalanced_sinkhorn_divergence",
                    "blur_cells": parameters["sinkhorn_blur_cells"],
                    "reach_cells": parameters["sinkhorn_reach_cells"],
                    "iterations": DEFAULT_SINKHORN_ITERATIONS,
                },
                "raster_loss_weight": parameters["raster_loss_weight"],
                "selection_metric": "validation_macro_f1",
                "test_split_evaluated": False,
            },
            validation_callback=report_validation,
        )
        trial_summary = {
            "trial_number": trial.number,
            "state": "complete",
            "hyperparameters": parameters,
            "best_validation_macro_f1": best_validation_macro_f1,
            "checkpoint_validation_loss": checkpoint_total_loss,
            "checkpoint_validation_classification_loss": checkpoint_classification_loss,
            "checkpoint_validation_raster_sinkhorn": checkpoint_raster_loss,
            "loss_history": {"train": train_losses, "validation": validation_losses},
            "test_split_evaluated": False,
        }
        with (trial_dir / "trial_summary.json").open("w") as destination:
            json.dump(trial_summary, destination, indent=2)
        trial.set_user_attr("checkpoint_path", str(checkpoint_path))
        return best_validation_macro_f1
    except optuna.TrialPruned:
        with (trial_dir / "trial_summary.json").open("w") as destination:
            json.dump(
                {
                    "trial_number": trial.number,
                    "state": "pruned",
                    "hyperparameters": parameters,
                    "best_validation_macro_f1": best_reported,
                    "test_split_evaluated": False,
                },
                destination,
                indent=2,
            )
        raise
    finally:
        del model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _write_study_outputs(study: optuna.Study, study_dir: Path) -> None:
    # Persist a human-readable view alongside the resumable SQLite study
    completed_trials = [trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE]
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "trial_counts": {
            state.name.lower(): sum(trial.state is state for trial in study.trials)
            for state in optuna.trial.TrialState
        },
        "best_trial": (
            {
                "number": study.best_trial.number,
                "validation_macro_f1": study.best_value,
                "hyperparameters": study.best_params,
                "checkpoint_path": study.best_trial.user_attrs.get("checkpoint_path"),
            }
            if completed_trials
            else None
        ),
    }
    with (study_dir / "study_summary.json").open("w") as destination:
        json.dump(summary, destination, indent=2)
    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)


def main() -> None:
    """Run or resume the validation-only hyperparameter study."""
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)
    study_dir = _resolve_study_dir(args.study_dir)
    print(f"Preparing validation-only hyperparameter study on {device}; outputs: {study_dir}")
    prepared = _prepare_search(args, study_dir, device)
    storage = f"sqlite:///{study_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=TPE_STARTUP_TRIALS,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=TPE_STARTUP_TRIALS,
            n_warmup_steps=PRUNING_WARMUP_EPOCHS,
        ),
    )

    def persist_after_trial(completed_study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        _write_study_outputs(completed_study, study_dir)

    study.optimize(
        lambda trial: _objective(
            trial,
            args=args,
            study_dir=study_dir,
            device=device,
            prepared=prepared,
        ),
        n_trials=args.trials,
        callbacks=[persist_after_trial],
        gc_after_trial=True,
    )
    _write_study_outputs(study, study_dir)
    print(f"Best validation macro F1: {study.best_value:.4f}; parameters: {study.best_params}")


if __name__ == "__main__":
    main()
