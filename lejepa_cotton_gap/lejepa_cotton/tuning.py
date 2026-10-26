"""Optuna hyper-parameter tuning and multi-seed retraining of detection arms.

Two arms (LeJEPA-init and COCO-init) share one search space, one trial budget
and one sampler seed. Trials are 30-epoch proxies scored on the *validation*
split only (mAP50-95); the *test* split is read only by
:func:`retrain_best`. Studies live in an SQLite file so runs can resume.
"""

from __future__ import annotations

import json
import random
import shutil
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
VAL_METRIC_KEY = "metrics/mAP50-95(B)"

# Recommended configuration for small labelled sets (~200 images) with a pretrained backbone
# and random neck/head: AdamW with a moderate rate, a short backbone freeze so the random
# head settles first, a damped backbone rate, and full mosaic with light mixup.
RECOMMENDED_PARAMS: Dict[str, object] = {
    "lr0": 1e-3, "lrf": 0.05, "weight_decay": 5e-4, "optimizer": "AdamW", "warmup_epochs": 3,
    "freeze_backbone_epochs": 5, "backbone_lr_mult": 0.5, "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.1,
}


@dataclass
class TuningConfig:
    """Settings shared by both arms.

    Parameters
    ----------
    n_trials : int, default=40
        Finished trials (complete, pruned or failed) per arm; identical for both.
    prune_percentile : float, default=33
        Percentile threshold used by the pruning algorithm.
    proxy_epochs : int, default=30
        Epochs per tuning trial.
    final_epochs : int, default=120
        Epochs of each final retraining.
    seeds : tuple of int, default=(0, 1, 2)
        Retraining seeds.
    regime_fractions : tuple of float, default=(0.1, 0.25, 0.5, 1.0)
        Fractions of the training split used by :func:`retrain_by_regime`
        (smallest to largest); validation and test sets stay fixed.
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

    n_trials: int = 40
    proxy_epochs: int = 30
    final_epochs: int = 120
    seeds: Tuple[int, ...] = (0, 1, 2)
    regime_fractions: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    sampler_seed: int = 42
    prune_percentile: float = 33
    n_startup_trials: int = 4
    n_warmup_steps: int = 5
    skip_tuning: bool = False
    fixed_params: Dict[str, object] = field(default_factory=lambda: dict(RECOMMENDED_PARAMS))


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
    """Create or resume the persisted study of an arm (TPE + percentile pruner)."""
    return optuna.create_study(
        study_name=f"{arm}", storage=study_storage(cfg), direction="maximize", load_if_exists=True,
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
    if tcfg.skip_tuning:
        print("skip_tuning=True: using fixed hyper-parameters, no Optuna search.")
        return {}
    coco_state = coco_state_dict(cfg.weights)
    return {arm: tune_arm(cfg, data_yaml, arm, tcfg, device, coco_state) for arm in TUNING_ARMS}


def _arm_params(arm: str, tcfg: TuningConfig, cfg: DetectionEvalConfig,
                studies: Optional[Dict[str, optuna.Study]]) -> Dict[str, object]:
    """Parameters for an arm: the fixed set when tuning is skipped, else the study's best."""
    if tcfg.skip_tuning:
        return tcfg.fixed_params
    return (studies[arm] if studies else load_study(cfg, arm, tcfg)).best_params


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
    coco_state = coco_state_dict(cfg.weights)
    folder = cfg.output_dir / "optuna"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in TUNING_ARMS:
        init = _init_weights(cfg, arm)
        for seed in tcfg.seeds:
            cache = folder / f"final_{arm}_seed{seed}.json"
            if cache.exists():
                rows.append(json.loads(cache.read_text()))
                continue
            run_cfg = apply_params(cfg, _arm_params(arm, tcfg, cfg, studies), tcfg.final_epochs, seed)
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


def regime_dataset(split_dir: Path, fraction: float, out_dir: Path, seed: int) -> Tuple[Path, int]:
    """Build a dataset whose train split is a nested random subset of ``split_dir``'s train split.

    Val and test images are shared with the full split (symlinked folders), so every
    regime is scored on identical images. Subsets are nested: smaller regimes are
    prefixes of one seeded shuffle.
    """
    train_images = sorted((split_dir / "images" / "train").iterdir())
    random.Random(seed).shuffle(train_images)
    n = max(1, round(len(train_images) * fraction))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("images", "labels"):
        (out_dir / sub / "train").mkdir(parents=True)
        for extra in ("val", "test"):
            if (split_dir / sub / extra).exists():
                (out_dir / sub / extra).symlink_to((split_dir / sub / extra).resolve())
    for image in train_images[:n]:
        shutil.copy2(image, out_dir / "images" / "train" / image.name)
        label = split_dir / "labels" / "train" / f"{image.stem}.txt"
        if label.exists():
            shutil.copy2(label, out_dir / "labels" / "train" / label.name)
    yaml_text = (split_dir / "dataset.yaml").read_text().replace(
        str(split_dir.resolve()), str(out_dir.resolve()), 1)
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_text)
    return yaml_path, n


def retrain_by_regime(cfg: DetectionEvalConfig, tcfg: TuningConfig = TuningConfig(),
                      studies: Optional[Dict[str, optuna.Study]] = None) -> pd.DataFrame:
    """Retrain the tuned LeJEPA-init and COCO-init arms at several training-set sizes.

    Uses each arm's best tuned parameters, every seed in ``tcfg.seeds`` and the
    fractions in ``tcfg.regime_fractions`` (smallest to largest). Metrics are on the
    fixed test split. Results are saved to ``<output_dir>/optuna/regime_runs.csv``,
    ``regime_summary.csv`` and ``regime_comparison.csv``; finished runs are reused.

    Returns
    -------
    pandas.DataFrame
        One row per regime, arm and seed.
    """
    device = ultralytics_device(select_device(cfg.device))
    base_yaml = cfg.split_dir / "dataset.yaml"
    if not base_yaml.exists():
        raise FileNotFoundError(f"{base_yaml} not found; run run_tuning first.")
    coco_state = coco_state_dict(cfg.weights)
    folder = cfg.output_dir / "optuna"
    rows = []
    for fraction in sorted(tcfg.regime_fractions):
        tag = f"frac{int(round(fraction * 100)):03d}"
        data_yaml, n_train = regime_dataset(cfg.split_dir, fraction, cfg.output_dir / "regimes" / tag,
                                            cfg.seed)
        for arm in TUNING_ARMS:
            init = _init_weights(cfg, arm)
            for seed in tcfg.seeds:
                cache = folder / f"regime_{tag}_{arm}_seed{seed}.json"
                if cache.exists():
                    rows.append(json.loads(cache.read_text()))
                    continue
                run_cfg = apply_params(cfg, _arm_params(arm, tcfg, cfg, studies), tcfg.final_epochs, seed)
                name = f"regime_{tag}_{arm}_seed{seed}"
                run_dir, _ = finetune_detector(init, data_yaml, run_cfg, name, device, coco_state,
                                               TUNING_ARMS[arm])
                test = validate_detector(run_dir / "weights" / "best.pt", data_yaml, run_cfg, name, device,
                                         split="test")
                row = {"regime": tag, "fraction": fraction, "n_train": n_train, "arm": arm, "seed": seed,
                       "mAP50": test["mAP50"], "mAP50-95": test["mAP50-95"]}
                cache.write_text(json.dumps(row))
                rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(folder / "regime_runs.csv", index=False)
    regime_summary(runs).to_csv(folder / "regime_summary.csv")
    regime_comparison(runs).to_csv(folder / "regime_comparison.csv")
    return runs


def regime_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of test mAP per regime and arm."""
    grouped = runs.groupby(["n_train", "arm"])[["mAP50", "mAP50-95"]]
    mean, std = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    return pd.DataFrame({m: [f"{a:.4f} ± {b:.4f}" for a, b in zip(mean[m], std[m])] for m in mean.columns},
                        index=mean.index)


def regime_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side LeJEPA-init vs COCO-init test mAP per regime, with the LeJEPA gain."""
    mean = runs.groupby(["n_train", "arm"])[["mAP50", "mAP50-95"]].mean().unstack("arm")
    table = pd.DataFrame(index=mean.index)
    for metric in ("mAP50", "mAP50-95"):
        table[f"{metric} LeJEPA"] = mean[(metric, "lejepa_init")]
        table[f"{metric} COCO"] = mean[(metric, "coco_init")]
        table[f"{metric} gain"] = table[f"{metric} LeJEPA"] - table[f"{metric} COCO"]
        table[f"{metric} gain %"] = 100 * table[f"{metric} gain"] / table[f"{metric} COCO"]
    return table.sort_index().round(4)


def plot_regime_bars(runs: pd.DataFrame, out_path: Optional[Path] = None):
    """Grouped bars of test mAP (mean ± std over seeds) by training-set size, LeJEPA vs COCO."""
    metrics = ("mAP50", "mAP50-95")
    sizes = sorted(runs["n_train"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 1, 4.5), sharey=True)
    width = 0.8 / len(TUNING_ARMS)
    for ax, metric in zip(axes, metrics):
        for i, (arm, variant) in enumerate(TUNING_ARMS.items()):
            sub = runs[runs["arm"] == arm].groupby("n_train")[metric]
            mean = sub.mean().reindex(sizes)
            std = sub.std(ddof=1).reindex(sizes).fillna(0.0)
            x = np.arange(len(sizes)) + (i - (len(TUNING_ARMS) - 1) / 2) * width
            ax.bar(x, mean, width, yerr=std, capsize=3, label=display_variant_name(variant))
        ax.set_xticks(np.arange(len(sizes)))
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set(xlabel="Training images", ylabel=f"Test {metric}", title=metric)
        ax.grid(True, axis="y", alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle("LeJEPA vs COCO initialisation by data regime")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std (sample std over seeds) of test mAP per arm."""
    grouped = runs.groupby("arm")[["mAP50", "mAP50-95"]]
    table = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    for metric in ("mAP50", "mAP50-95"):
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
