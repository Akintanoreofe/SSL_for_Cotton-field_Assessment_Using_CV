"""Optuna hyper-parameter tuning of the LeJEPA arm and final training of both arms.

Only the LeJEPA-init arm is tuned (pruned Optuna study, 100 trials of 10 epochs,
scored on the *validation* split by mAP50-95). The best configuration is saved
and logged, then reused unchanged to train the LeJEPA-init and the COCO-init
backbones for 30 epochs with the same pipeline. Studies live in an SQLite file
so runs can resume.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
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
    display_variant_name,
    finetune_detector,
    prepare_detection_split,
    ultralytics_device,
    validate_detector,
)
from .core_pretraining import select_device

# arm name -> detection variant (neck and head are random in both arms)
TUNING_ARMS: Dict[str, str] = {"lejepa_init": "lejepa", "coco_init": "coco_backbone"}
TUNED_ARMS: Tuple[str, ...] = ("lejepa_init",)  # only the LeJEPA backbone is searched
VAL_METRIC_KEY = "metrics/mAP50-95(B)"

# Recommended configuration for small labelled sets (~200 images) with a pretrained backbone
# and random neck/head: AdamW with a moderate rate, a short backbone freeze so the random
# head settles first, a damped backbone rate, and full mosaic with light mixup.
RECOMMENDED_PARAMS: Dict[str, object] = {
    "lr0": 2e-3, "lrf": 0.05, "weight_decay": 5e-4, "optimizer": "AdamW", "warmup_epochs": 5,
    "freeze_backbone_epochs": 5, "backbone_lr_mult": 0.5, "mosaic": 1.0, "close_mosaic": 15, "mixup": 0.1,
}


@dataclass
class TuningConfig:
    """Settings shared by both arms.

    Parameters
    ----------
    n_trials : int, default=100
        Finished trials (complete, pruned or failed) of the LeJEPA study.
    prune_percentile : float, default=33
        Percentile threshold used by the pruning algorithm.
    proxy_epochs : int, default=10
        Epochs per tuning trial.
    final_epochs : int, default=30
        Epochs of the final training of both arms.
    seeds : tuple of int, default=(0,)
        Final-training seeds.
    sampler_seed : int, default=42
        Seed of the TPE sampler (same for both arms).
    n_startup_trials, n_warmup_steps : int
        Percentile-pruner settings.
    skip_tuning : bool, default=False
        Bypass the Optuna search: :func:`run_tuning` only prepares the split and the
        retraining functions use ``fixed_params`` for both arms.
    fixed_params : dict, default=RECOMMENDED_PARAMS
        Parameters used when ``skip_tuning`` is True (same keys as :func:`suggest_params`).
    """

    n_trials: int = 100
    proxy_epochs: int = 10
    final_epochs: int = 30
    seeds: Tuple[int, ...] = (0,)
    sampler_seed: int = 42
    prune_percentile: float = 33
    n_startup_trials: int = 4
    n_warmup_steps: int = 5
    skip_tuning: bool = False
    fixed_params: Dict[str, object] = field(default_factory=lambda: dict(RECOMMENDED_PARAMS))


# Bump when the search space changes: Optuna cannot resume a study whose categorical choices differ.
SEARCH_SPACE_VERSION = "v2"


def suggest_params(trial: optuna.Trial) -> Dict[str, object]:
    """Draw one configuration from the search space (same for every arm)."""
    return {
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
        "lrf": trial.suggest_float("lrf", 0.01, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW"]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 3),
        "freeze_backbone_epochs": trial.suggest_categorical("freeze_backbone_epochs", [0, 2, 5, 8]),
        "backbone_lr_mult": trial.suggest_float("backbone_lr_mult", 0.1, 1.0),
        "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
        "close_mosaic": trial.suggest_categorical("close_mosaic", [0, 3, 5]),
        "mixup": trial.suggest_float("mixup", 0.0, 0.2),
    }


def apply_params(cfg: DetectionEvalConfig, params: Dict[str, object], epochs: int,
                 seed: Optional[int] = None) -> DetectionEvalConfig:
    """Copy a detection config with tuned parameters applied."""
    overrides = {**cfg.train_overrides, "mosaic": params["mosaic"], "mixup": params["mixup"]}
    return replace(
        cfg, epochs=epochs, seed=cfg.seed if seed is None else seed, lr0=params["lr0"], lrf=params["lrf"],
        warmup_epochs=int(params["warmup_epochs"]),
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
    """Create or resume the persisted study of an arm (TPE + percentile pruner)."""
    return optuna.create_study(
        study_name=f"{arm}_{SEARCH_SPACE_VERSION}", storage=study_storage(cfg), direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=tcfg.sampler_seed),
        pruner=optuna.pruners.PercentilePruner(
            percentile=tcfg.prune_percentile,
            n_startup_trials=tcfg.n_startup_trials,
            n_warmup_steps=tcfg.n_warmup_steps,
        ),
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
        try:
            run_dir, _ = finetune_detector(init, data_yaml, trial_cfg, name, device, coco_state,
                                           TUNING_ARMS[arm], epoch_callback=report)
        except RuntimeError as error:
            # Diverged (NaN) predictions or MPS masked-assignment bugs break the TAL assigner.
            if "shape mismatch" not in str(error):
                raise
            print(f"Trial {trial.number} pruned: assigner shape mismatch ({error}). "
                  "If this repeats, set DetectionEvalConfig.device='cpu' or lower lr0.")
            raise optuna.TrialPruned() from error
        return validate_detector(run_dir / "weights" / "best.pt", data_yaml, trial_cfg, name, device,
                                 split="val")["mAP50-95"]

    study.optimize(objective, n_trials=remaining)
    return study


def evaluation_split(cfg: DetectionEvalConfig) -> str:
    """Designated evaluation split: ``"test"`` when held out, else ``"val"``."""
    return "test" if cfg.test_ratio > 0 else "val"


def best_params_path(cfg: DetectionEvalConfig) -> Path:
    """JSON file holding the best LeJEPA hyper-parameters."""
    return cfg.output_dir / "optuna" / "best_params_lejepa.json"


def save_best_params(cfg: DetectionEvalConfig, study: optuna.Study) -> Path:
    """Write the best LeJEPA configuration to JSON and log it."""
    path = best_params_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = dict(Counter(t.state.name for t in study.trials))
    record = {"best_value_val_mAP50-95": study.best_value, "best_trial": study.best_trial.number,
              "params": study.best_params, "trials": states}
    path.write_text(json.dumps(record, indent=2))
    print(f"Best LeJEPA configuration (val mAP50-95 = {study.best_value:.4f}, "
          f"trial {study.best_trial.number}; trials: {states}) saved to {path}")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    return path


def run_tuning(cfg: DetectionEvalConfig, tcfg: TuningConfig = TuningConfig()) -> Dict[str, optuna.Study]:
    """Tune the LeJEPA arm only and save its best configuration.

    ``cfg.weights.lejepa_checkpoint`` must point at the pretrained encoder and
    ``cfg.test_ratio`` must be positive so a test split is held out.
    """
    if cfg.test_ratio <= 0:
        raise ValueError("Set DetectionEvalConfig.test_ratio > 0 so tuning never touches the test split.")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = prepare_detection_split(cfg.source_dir, cfg.split_dir, cfg.class_names, cfg.val_ratio,
                                        cfg.max_train_images, cfg.seed, cfg.split_by, cfg.test_ratio)
    if tcfg.skip_tuning:
        print("skip_tuning=True: using fixed hyper-parameters, no Optuna search.")
        return {}
    coco_state = coco_state_dict(cfg.weights)
    studies = {arm: tune_arm(cfg, data_yaml, arm, tcfg, device, coco_state) for arm in TUNED_ARMS}
    for study in studies.values():
        save_best_params(cfg, study)
    return studies


def best_params(cfg: DetectionEvalConfig, tcfg: TuningConfig,
                studies: Optional[Dict[str, optuna.Study]] = None) -> Dict[str, object]:
    """Parameters shared by both final arms: the fixed set when tuning is skipped, else the LeJEPA best."""
    if tcfg.skip_tuning:
        return tcfg.fixed_params
    if studies and "lejepa_init" in studies:
        return studies["lejepa_init"].best_params
    path = best_params_path(cfg)
    if path.exists():
        return json.loads(path.read_text())["params"]
    return load_study(cfg, "lejepa_init", tcfg).best_params


def retrain_best(cfg: DetectionEvalConfig, tcfg: TuningConfig = TuningConfig(),
                 studies: Optional[Dict[str, optuna.Study]] = None) -> pd.DataFrame:
    """Train both arms for ``final_epochs`` with the best LeJEPA hyper-parameters.

    The LeJEPA-init and COCO-init backbones use the identical pipeline and
    parameters. Each model is scored on the evaluation split (the test split
    when ``cfg.test_ratio > 0``, else validation), the primary benchmark, and
    on the training split as a diagnostic.

    Returns
    -------
    pandas.DataFrame
        One row per arm, seed and split (``"eval"`` or ``"train"``) with
        precision, recall, F1, mAP50, mAP50-95 and mAP75; also saved to
        ``<output_dir>/optuna/final_runs.csv`` and ``final_summary.csv``
        (evaluation split, mean ± std per arm). Finished runs are reused.
    """
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = cfg.split_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found; run run_tuning first.")
    coco_state = coco_state_dict(cfg.weights)
    folder = cfg.output_dir / "optuna"
    folder.mkdir(parents=True, exist_ok=True)
    params = best_params(cfg, tcfg, studies)
    print(f"Final training ({tcfg.final_epochs} epochs) of both arms with: {params}")
    eval_split = evaluation_split(cfg)
    rows = []
    for arm in TUNING_ARMS:
        init = _init_weights(cfg, arm)
        for seed in tcfg.seeds:
            cache = folder / f"final_{arm}_seed{seed}.json"
            cached = json.loads(cache.read_text()) if cache.exists() else None
            if isinstance(cached, list):  # older caches hold a single test-only dict and are re-run
                rows.extend(cached)
                continue
            run_cfg = apply_params(cfg, params, tcfg.final_epochs, seed)
            name = f"final_{arm}_seed{seed}"
            run_dir, _ = finetune_detector(init, data_yaml, run_cfg, name, device, coco_state,
                                           TUNING_ARMS[arm])
            best = run_dir / "weights" / "best.pt"
            arm_rows = [{"arm": arm, "seed": seed, "split": label,
                         **validate_detector(best, data_yaml, run_cfg, name, device, split=split)}
                        for label, split in (("eval", eval_split), ("train", "train"))]
            cache.write_text(json.dumps(arm_rows))
            rows.extend(arm_rows)
    runs = pd.DataFrame(rows)
    runs.to_csv(folder / "final_runs.csv", index=False)
    summary_table(runs[runs["split"] == "eval"]).to_csv(folder / "final_summary.csv")
    return runs


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std (sample std over seeds) of the detection metrics per arm."""
    metrics = [m for m in ("precision", "recall", "f1", "mAP50", "mAP50-95") if m in runs]
    grouped = runs.groupby("arm")[metrics]
    table = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    for metric in metrics:
        table[metric] = [f"{m:.4f} ± {s:.4f}" for m, s in
                         zip(table[f"{metric}_mean"], table[f"{metric}_std"].fillna(0.0))]
    return table


def print_best_params(studies: Dict[str, optuna.Study], tcfg: Optional[TuningConfig] = None) -> None:
    """Print best value and parameters of every arm (or the fixed set when tuning is skipped)."""
    if not studies:
        print("No studies (tuning skipped); fixed parameters:")
        for key, value in (tcfg or TuningConfig()).fixed_params.items():
            print(f"    {key}: {value}")
        return
    for arm, study in studies.items():
        print(f"[{arm}] best val mAP50-95 = {study.best_value:.4f} (trial {study.best_trial.number})")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")


def plot_optimization_history(studies: Dict[str, optuna.Study], out_path: Optional[Path] = None):
    """Plot trial values and running best per arm."""
    if not studies:
        print("No studies to plot (tuning skipped).")
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in trials]
        label = display_variant_name(TUNING_ARMS[arm])
        line, = ax.plot([t.number for t in trials], values, "o", alpha=0.5, label=f"{label} trials")
        ax.plot([t.number for t in trials], np.maximum.accumulate(values) if values else [],
                color=line.get_color(), label=f"{label} best so far")
    ax.set(xlabel="Trial", ylabel="Val mAP50-95", title="Optuna optimisation history")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
