"""Optuna hyper-parameter tuning and multi-seed retraining of detection arms.

Two arms (LeJEPA-init and COCO-init) share one search space, one trial budget
and one sampler seed. Trials are 30-epoch proxies scored on the *validation*
split only (mAP50-95); the *test* split is read only by
:func:`retrain_best`. Studies live in an SQLite file so runs can resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .core_evaluation import (
    DetectionEvalConfig,
    coco_state_dict,
    detection_init_weights,
    finetune_detector,
    prepare_detection_split,
    ultralytics_device,
    validate_detector,
)
from .core_pretraining import select_device

# arm name -> detection variant (neck and head are random in both arms)
TUNING_ARMS: Dict[str, str] = {"lejepa_init": "lejepa", "coco_init": "coco_backbone"}
VAL_METRIC_KEY = "metrics/mAP50-95(B)"


@dataclass
class TuningConfig:
    """Settings shared by both arms.

    Parameters
    ----------
    n_trials : int, default=20
        Finished trials (complete, pruned or failed) per arm; identical for both.
    proxy_epochs : int, default=30
        Epochs per tuning trial.
    final_epochs : int, default=120
        Epochs of each final retraining.
    seeds : tuple of int, default=(0, 1, 2)
        Retraining seeds.
    sampler_seed : int, default=42
        Seed of the TPE sampler (same for both arms).
    n_startup_trials, n_warmup_steps : int
        Median-pruner settings.
    """

    n_trials: int = 20
    proxy_epochs: int = 30
    final_epochs: int = 120
    seeds: Tuple[int, ...] = (0, 1, 2)
    sampler_seed: int = 42
    n_startup_trials: int = 5
    n_warmup_steps: int = 5


def suggest_params(trial: optuna.Trial) -> Dict[str, object]:
    """Draw one configuration from the search space (same for every arm)."""
    return {
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
        "lrf": trial.suggest_float("lrf", 0.01, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW"]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
        "freeze_backbone_epochs": trial.suggest_categorical("freeze_backbone_epochs", [0, 5, 10, 20]),
        "backbone_lr_mult": trial.suggest_float("backbone_lr_mult", 0.1, 1.0),
        "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
        "close_mosaic": trial.suggest_categorical("close_mosaic", [0, 10, 20]),
        "mixup": trial.suggest_float("mixup", 0.0, 0.2),
    }


def apply_params(cfg: DetectionEvalConfig, params: Dict[str, object], epochs: int,
                 seed: Optional[int] = None) -> DetectionEvalConfig:
    """Copy a detection config with tuned parameters applied."""
    overrides = {**cfg.train_overrides, "warmup_epochs": params["warmup_epochs"],
                 "mosaic": params["mosaic"], "mixup": params["mixup"]}
    return replace(
        cfg, epochs=epochs, seed=cfg.seed if seed is None else seed, lr0=params["lr0"], lrf=params["lrf"],
        weight_decay=params["weight_decay"], optimizer=params["optimizer"],
        freeze_backbone_epochs=min(int(params["freeze_backbone_epochs"]), epochs),
        backbone_lr_mult=params["backbone_lr_mult"], close_mosaic=int(params["close_mosaic"]),
        train_overrides=overrides,
    )


def study_storage(cfg: DetectionEvalConfig) -> str:
    """SQLite URL holding both studies."""
    folder = cfg.output_dir / "optuna"
    folder.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(folder / 'studies.db').resolve()}"


def load_study(cfg: DetectionEvalConfig, arm: str, tcfg: TuningConfig) -> optuna.Study:
    """Create or resume the persisted study of an arm (TPE + median pruner)."""
    return optuna.create_study(
        study_name=f"{arm}", storage=study_storage(cfg), direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=tcfg.sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=tcfg.n_startup_trials,
                                           n_warmup_steps=tcfg.n_warmup_steps),
    )


def _init_weights(cfg: DetectionEvalConfig, arm: str) -> str:
    return detection_init_weights(TUNING_ARMS[arm], cfg.weights, cfg.init_dir, f"tune_{arm}")


def tune_arm(cfg: DetectionEvalConfig, data_yaml: Path, arm: str, tcfg: TuningConfig,
             device: str, coco_state=None) -> optuna.Study:
    """Run (or resume) the proxy search of one arm; objective is val mAP50-95."""
    study = load_study(cfg, arm, tcfg)
    done = [t for t in study.trials if t.state.is_finished()]
    remaining = tcfg.n_trials - len(done)
    if remaining <= 0:
        return study
    init = _init_weights(cfg, arm)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = apply_params(cfg, suggest_params(trial), tcfg.proxy_epochs)

        def report(epoch: int, metrics: Dict[str, float]) -> None:
            value = metrics.get(VAL_METRIC_KEY)
            if value is None or not np.isfinite(value):
                return
            trial.report(float(value), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        name = f"tune_{arm}_trial{trial.number:03d}"
        run_dir, _ = finetune_detector(init, data_yaml, trial_cfg, name, device, coco_state,
                                       TUNING_ARMS[arm], epoch_callback=report)
        return validate_detector(run_dir / "weights" / "best.pt", data_yaml, trial_cfg, name, device,
                                 split="val")["mAP50-95"]

    study.optimize(objective, n_trials=remaining)
    return study


def run_tuning(cfg: DetectionEvalConfig, tcfg: TuningConfig = TuningConfig()) -> Dict[str, optuna.Study]:
    """Tune both arms with the same search space and ``n_trials``.

    ``cfg.weights.lejepa_checkpoint`` must point at the pretrained encoder and
    ``cfg.test_ratio`` must be positive so a test split is held out.
    """
    if cfg.test_ratio <= 0:
        raise ValueError("Set DetectionEvalConfig.test_ratio > 0 so tuning never touches the test split.")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = prepare_detection_split(cfg.source_dir, cfg.split_dir, cfg.class_names, cfg.val_ratio,
                                        cfg.max_train_images, cfg.seed, cfg.split_by, cfg.test_ratio)
    coco_state = coco_state_dict(cfg.weights)
    return {arm: tune_arm(cfg, data_yaml, arm, tcfg, device, coco_state) for arm in TUNING_ARMS}


def retrain_best(cfg: DetectionEvalConfig, tcfg: TuningConfig = TuningConfig(),
                 studies: Optional[Dict[str, optuna.Study]] = None) -> pd.DataFrame:
    """Retrain each arm's best config for ``final_epochs`` with every seed.

    Returns
    -------
    pandas.DataFrame
        One row per arm and seed with test mAP50 and mAP50-95; also saved to
        ``<output_dir>/optuna/final_runs.csv`` and ``final_summary.csv``
        (mean and std per arm). Finished runs are reused on resume.
    """
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = cfg.split_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found; run run_tuning first.")
    studies = studies or {arm: load_study(cfg, arm, tcfg) for arm in TUNING_ARMS}
    coco_state = coco_state_dict(cfg.weights)
    folder = cfg.output_dir / "optuna"
    rows = []
    for arm, study in studies.items():
        init = _init_weights(cfg, arm)
        for seed in tcfg.seeds:
            cache = folder / f"final_{arm}_seed{seed}.json"
            if cache.exists():
                rows.append(json.loads(cache.read_text()))
                continue
            run_cfg = apply_params(cfg, study.best_params, tcfg.final_epochs, seed)
            name = f"final_{arm}_seed{seed}"
            run_dir, _ = finetune_detector(init, data_yaml, run_cfg, name, device, coco_state,
                                           TUNING_ARMS[arm])
            test = validate_detector(run_dir / "weights" / "best.pt", data_yaml, run_cfg, name, device,
                                     split="test")
            row = {"arm": arm, "seed": seed, "mAP50": test["mAP50"], "mAP50-95": test["mAP50-95"]}
            cache.write_text(json.dumps(row))
            rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(folder / "final_runs.csv", index=False)
    summary_table(runs).to_csv(folder / "final_summary.csv")
    return runs


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std (sample std over seeds) of test mAP per arm."""
    grouped = runs.groupby("arm")[["mAP50", "mAP50-95"]]
    table = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    for metric in ("mAP50", "mAP50-95"):
        table[metric] = [f"{m:.4f} ± {s:.4f}" for m, s in
                         zip(table[f"{metric}_mean"], table[f"{metric}_std"].fillna(0.0))]
    return table


def print_best_params(studies: Dict[str, optuna.Study]) -> None:
    """Print best value and parameters of every arm."""
    for arm, study in studies.items():
        print(f"[{arm}] best val mAP50-95 = {study.best_value:.4f} (trial {study.best_trial.number})")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")


def plot_optimization_history(studies: Dict[str, optuna.Study], out_path: Optional[Path] = None):
    """Plot trial values and running best per arm."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in trials]
        line, = ax.plot([t.number for t in trials], values, "o", alpha=0.5, label=f"{arm} trials")
        ax.plot([t.number for t in trials], np.maximum.accumulate(values) if values else [],
                color=line.get_color(), label=f"{arm} best so far")
    ax.set(xlabel="Trial", ylabel="Val mAP50-95", title="Optuna optimisation history")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
