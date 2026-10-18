"""End-to-end runners that wire pretraining, evaluation and visualization.

These functions are the only place where the three core modules meet. They
decide *which* artefacts to write, always inside the ``output_dir`` supplied
through the configuration objects. Notebooks and scripts should only build
configurations and call these runners.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .core_evaluation import (
    DetectionEvalConfig,
    ProbeEvalConfig,
    build_probe_loaders,
    classification_summary,
    detection_init_weights,
    coco_state_dict,
    extract_embeddings,
    finetune_detector,
    fit_probe_variant,
    predict,
    save_box_predictions,
    prepare_detection_split,
    ultralytics_device,
    validate_detector,
)
from .core_pretraining import (
    PretrainConfig,
    YOLOv8GAPEncoder,
    build_pretraining_loader,
    config_for_lambda,
    lambda_artifact_paths,
    lambda_tag,
    pretrain,
    read_pretrain_config,
    save_encoder,
    select_device,
)
from .visualization import (
    save_confusion_matrix,
    save_detection_curves,
    save_detection_overlays,
    save_lambda_loss_curves,
    save_metric_bars,
    save_metric_vs_lambda,
    save_pca_2d,
    save_pca_3d,
    save_pretraining_loss_curves,
    save_probe_loss_curves,
)

DETECTION_METRICS = ("precision", "recall", "mAP50", "mAP50-95")
PROBE_METRICS = ("accuracy", "macro_f1")
COCO_MODEL_NAME = "coco"


def default_pca_epochs(epochs: int) -> Set[int]:
    """Zero-based epochs at which pretraining embeddings are visualised.

    Parameters
    ----------
    epochs : int
        Total number of pretraining epochs.

    Returns
    -------
    set of int
        First, middle and last epoch.
    """
    return {0, epochs // 2, epochs - 1}


def log_pretraining_pca(epoch: int, encoder: YOLOv8GAPEncoder, loader: DataLoader, *,
                        epochs_to_log: Set[int], out_dir: Path, max_batches: int) -> Optional[Path]:
    """Epoch callback saving a 3-D PCA of GAP embeddings coloured by camera.

    Well-mixed camera colours indicate that the embedding is invariant to the
    viewpoint, which is what the cross-camera prediction loss encourages.

    Parameters
    ----------
    epoch : int
        Zero-based epoch just finished.
    encoder : YOLOv8GAPEncoder
        Encoder being trained.
    loader : torch.utils.data.DataLoader
        Pretraining loader yielding ``(views, camera_ids)``.
    epochs_to_log : set of int
        Zero-based epochs to visualise.
    out_dir : pathlib.Path
        Destination folder for the HTML files.
    max_batches : int
        Batches used to build the embedding cloud.

    Returns
    -------
    pathlib.Path or None
        Written HTML file, or ``None`` when ``epoch`` is not logged.
    """
    if epoch not in epochs_to_log:
        return None
    features, cameras = extract_embeddings(encoder, loader, max_batches)
    return save_pca_3d(features, [f"cam{c}" for c in cameras],
                       f"GAP LeJEPA embeddings - epoch {epoch + 1}",
                       Path(out_dir) / f"pca_epoch_{epoch + 1:03d}.html", legend_title="camera")


def run_pretraining(cfg: PretrainConfig, pca_epochs: Optional[Iterable[int]] = None,
                    pca_max_batches: int = 60,
                    checkpoint_path: Optional[Path] = None) -> Tuple[Path, pd.DataFrame]:
    """Pretrain, then save the checkpoint, loss history and plots.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    pca_epochs : iterable of int or None, default=None
        Zero-based epochs to visualise; defaults to first, middle and last.
    pca_max_batches : int, default=60
        Batches used per PCA snapshot.
    checkpoint_path : pathlib.Path or None, default=None
        Destination of the checkpoint; defaults to ``cfg.checkpoint_path``.

    Returns
    -------
    checkpoint : pathlib.Path
        Saved encoder checkpoint.
    history : pandas.DataFrame
        Per-epoch loss components.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.output_dir / "plots"
    epochs_to_log = set(pca_epochs) if pca_epochs is not None else default_pca_epochs(cfg.epochs)
    callback = partial(log_pretraining_pca, epochs_to_log=epochs_to_log, out_dir=plots_dir / "pca_3d",
                       max_batches=pca_max_batches)
    encoder, history = pretrain(cfg, build_pretraining_loader(cfg), on_epoch_end=callback)

    checkpoint = save_encoder(encoder, checkpoint_path or cfg.checkpoint_path, cfg)
    history.to_csv(cfg.output_dir / "loss_history.csv", index=False)
    save_pretraining_loss_curves(history, plots_dir / "loss_curves.png")
    print(f"Checkpoint: {checkpoint}")
    return checkpoint, history


def pretrain_one_lambda(cfg: PretrainConfig, lam: float) -> Path:
    """Pretrain one lambda and save its checkpoint, config and loss history.

    Parameters
    ----------
    cfg : PretrainConfig
        Base configuration.
    lam : float
        SIGReg weight of this run.

    Returns
    -------
    pathlib.Path
        ``backbone_lambda_<lam>.pt``; the config JSON and history CSV are
        written next to it.
    """
    lam_cfg = config_for_lambda(cfg, lam)
    paths = lambda_artifact_paths(cfg, lam)
    # Per-lambda output folder keeps the PCA and loss plots of different runs apart.
    run_cfg = replace(lam_cfg, output_dir=cfg.output_dir / f"lambda_{lambda_tag(lam)}")
    _, history = run_pretraining(run_cfg, checkpoint_path=paths["checkpoint"])
    history.to_csv(paths["history"], index=False)
    paths["config"].write_text(json.dumps(lam_cfg.to_dict(), indent=2))
    return paths["checkpoint"]


def run_lambda_sweep(cfg: PretrainConfig) -> Dict[float, Path]:
    """Pretrain one backbone per lambda in ``cfg.lambdas``.

    A lambda whose checkpoint already exists is skipped unless
    ``cfg.overwrite`` is true, so an interrupted sweep can be resumed.

    Parameters
    ----------
    cfg : PretrainConfig
        Base configuration (``lambdas``, ``epochs``, ``pred_loss_mode``, ...).

    Returns
    -------
    dict of float to pathlib.Path
        Lambda to checkpoint path, in ``cfg.lambdas`` order.
    """
    checkpoints: Dict[float, Path] = {}
    for index, lam in enumerate(cfg.lambdas, start=1):
        checkpoint = lambda_artifact_paths(cfg, lam)["checkpoint"]
        print(f"=== lambda {lam:g} ({index}/{len(cfg.lambdas)}) ===")
        if checkpoint.exists() and not cfg.overwrite:
            print(f"Skipping: {checkpoint} already exists (set overwrite=True to retrain).")
        else:
            pretrain_one_lambda(cfg, lam)
        checkpoints[lam] = checkpoint
    return checkpoints


def load_lambda_histories(cfg: PretrainConfig) -> Dict[float, pd.DataFrame]:
    """Read the saved loss history of every lambda that has been trained.

    Parameters
    ----------
    cfg : PretrainConfig
        Configuration providing ``lambdas`` and the output folder.

    Returns
    -------
    dict of float to pandas.DataFrame
        Lambda to history; lambdas without a history file are omitted.
    """
    paths = {lam: lambda_artifact_paths(cfg, lam)["history"] for lam in cfg.lambdas}
    return {lam: pd.read_csv(path) for lam, path in paths.items() if path.exists()}


def plot_lambda_losses(cfg: PretrainConfig, out_path: Optional[Path] = None) -> Path:
    """Plot the saved pretraining losses of all lambdas.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    out_path : pathlib.Path or None, default=None
        Destination ``.png``; defaults to ``<output_dir>/plots/lambda_loss_curves.png``.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    return save_lambda_loss_curves(load_lambda_histories(cfg),
                                   out_path or cfg.output_dir / "plots" / "lambda_loss_curves.png")


def check_checkpoint_image_size(checkpoint: Path, image_size: int) -> None:
    """Ensure a backbone was pretrained at the detection image size.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Encoder checkpoint.
    image_size : int
        Detection image size.

    Raises
    ------
    ValueError
        If the checkpoint records a different pretraining size.
    """
    pretrained = read_pretrain_config(checkpoint).get("image_size")
    if pretrained is not None and pretrained != image_size:
        raise ValueError(f"{checkpoint} was pretrained at image_size={pretrained} but detection uses "
                         f"{image_size}; use the same size for both.")


def evaluate_model(name: str, init_weights: str, audit_variant: str, data_yaml: Path,
                   cfg: DetectionEvalConfig, device: str, coco_state: Dict[str, torch.Tensor],
                   max_overlays: int = 20, reuse_existing: bool = False) -> Tuple[dict, Path]:
    """Fine-tune one model, validate it and save overlays and box files.

    Parameters
    ----------
    name : str
        Model name; used for the run folder and ``eval/boxes_<name>.csv``.
    init_weights : str
        Starting weights from :func:`detection_init_weights`.
    audit_variant : str
        Variant whose initialisation is audited (e.g. ``"lejepa"``).
    data_yaml : pathlib.Path
        Ultralytics dataset file.
    cfg : DetectionEvalConfig
        Detection settings.
    device : str
        Ultralytics device string.
    coco_state : dict
        COCO state dictionary for the initialisation audit.
    max_overlays : int, default=20
        Validation images rendered with boxes.
    reuse_existing : bool, default=False
        Skip fine-tuning when a completed run of this name exists.

    Returns
    -------
    row : dict
        ``variant`` (the model name), metrics and COCO-share audit values.
    run_dir : pathlib.Path
        Run folder.
    """
    run_dir = cfg.runs_dir / name
    done_marker = run_dir / "finetune_done.txt"
    init_report: Dict[str, float] = {}
    if reuse_existing and done_marker.exists() and (run_dir / "weights" / "best.pt").exists():
        print(f"[{name}] reusing completed fine-tuning in {run_dir}")
    else:
        run_dir, init_report = finetune_detector(init_weights, data_yaml, cfg, name, device, coco_state,
                                                 audit_variant)
        done_marker.write_text("done\n")
    best = run_dir / "weights" / "best.pt"
    row = {"variant": name, **validate_detector(best, data_yaml, cfg, name, device), **init_report}
    val_images, val_labels = cfg.split_dir / "images" / "val", cfg.split_dir / "labels" / "val"
    save_detection_overlays(best, val_images, val_labels, cfg.output_dir / "overlays" / name,
                            cfg.image_size, cfg.conf, device, max_overlays)
    save_box_predictions(best, val_images, val_labels, cfg.eval_dir / f"boxes_{name}.csv", cfg.image_size,
                         cfg.conf, device)
    return row, run_dir


def run_lambda_evaluation(cfg: DetectionEvalConfig, checkpoints: Dict[float, Path],
                          overwrite: bool = False, max_overlays: int = 20) -> pd.DataFrame:
    """Fine-tune and evaluate a detector per lambda backbone and for COCO.

    The COCO baseline uses the COCO *backbone* only (``coco_backbone``);
    neck and head start random for every model, as for the lambda models.

    Parameters
    ----------
    cfg : DetectionEvalConfig
        Detection settings; ``cfg.weights.lejepa_checkpoint`` is ignored.
    checkpoints : dict of float to pathlib.Path
        Lambda to backbone checkpoint (from :func:`run_lambda_sweep`).
    overwrite : bool, default=False
        Re-run fine-tuning even if a completed run exists.
    max_overlays : int, default=20
        Validation images rendered with boxes per model.

    Returns
    -------
    pandas.DataFrame
        One row per model (``variant``, ``lambda``, metrics); also saved as
        ``summary_metrics.csv`` together with the two comparison plots.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = prepare_detection_split(cfg.source_dir, cfg.split_dir, cfg.class_names,
                                        cfg.val_ratio, cfg.max_train_images, cfg.seed, cfg.split_by)
    coco_state = coco_state_dict(cfg.weights)
    rows = []
    for lam, checkpoint in checkpoints.items():
        name = f"lambda_{lambda_tag(lam)}"
        check_checkpoint_image_size(checkpoint, cfg.image_size)
        sources = replace(cfg.weights, lejepa_checkpoint=checkpoint)
        init = detection_init_weights("lejepa", sources, cfg.init_dir, name)
        row, _ = evaluate_model(name, init, "lejepa", data_yaml, cfg, device, coco_state, max_overlays,
                                reuse_existing=not overwrite)
        rows.append({**row, "lambda": lam})
    init = detection_init_weights("coco_backbone", cfg.weights, cfg.init_dir, COCO_MODEL_NAME)
    row, _ = evaluate_model(COCO_MODEL_NAME, init, "coco_backbone", data_yaml, cfg, device, coco_state,
                            max_overlays, reuse_existing=not overwrite)
    rows.append({**row, "lambda": np.nan})

    summary = pd.DataFrame(rows)
    summary = summary[["variant", "lambda", *DETECTION_METRICS,
                       *[c for c in summary if c.startswith("coco_share_")]]]
    summary.to_csv(cfg.output_dir / "summary_metrics.csv", index=False)
    plots_dir = cfg.output_dir / "plots"
    save_metric_vs_lambda(summary, plots_dir / "map_vs_lambda.png", COCO_MODEL_NAME)
    save_metric_bars(summary, DETECTION_METRICS, "Cotton boll detection: each $\\lambda$ vs COCO",
                     plots_dir / "lambda_vs_coco_bars.png", figsize=(11, 5))
    return summary


def run_detection_evaluation(cfg: DetectionEvalConfig, max_overlays: int = 20) -> pd.DataFrame:
    """Fine-tune one detector per variant on the same split and compare them.

    Parameters
    ----------
    cfg : DetectionEvalConfig
        Detection evaluation configuration.
    max_overlays : int, default=20
        Validation images rendered with boxes per variant.

    Returns
    -------
    pandas.DataFrame
        One row per variant with precision, recall, mAP50 and mAP50-95, plus
        the share of backbone, neck and head tensors identical to COCO at the
        start of training (``coco_share_*``).
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = prepare_detection_split(cfg.source_dir, cfg.split_dir, cfg.class_names,
                                        cfg.val_ratio, cfg.max_train_images, cfg.seed, cfg.split_by)
    rows, run_dirs = [], {}
    coco_state = coco_state_dict(cfg.weights)
    for variant in cfg.variants:
        init_weights = detection_init_weights(variant, cfg.weights, cfg.init_dir)
        row, run_dirs[variant] = evaluate_model(variant, init_weights, variant, data_yaml, cfg, device,
                                                coco_state, max_overlays)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.output_dir / "detection_summary.csv", index=False)
    save_detection_curves(run_dirs, cfg.output_dir / "plots" / "detection_curves.png")
    save_metric_bars(summary, DETECTION_METRICS, "Cotton boll detection: LeJEPA vs COCO",
                     cfg.output_dir / "plots" / "detection_metrics.png")
    return summary


def run_probe_evaluation(cfg: ProbeEvalConfig) -> pd.DataFrame:
    """Train a frozen-backbone GAP linear probe per variant and compare them.

    Parameters
    ----------
    cfg : ProbeEvalConfig
        Linear-probe configuration.

    Returns
    -------
    pandas.DataFrame
        One row per variant with accuracy and macro F1.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(cfg.device)
    plots_dir = cfg.output_dir / "plots"
    train_loader, test_loader = build_probe_loaders(cfg)
    rows, histories = [], {}
    for variant in cfg.variants:
        probe, histories[variant] = fit_probe_variant(variant, cfg, train_loader, test_loader)
        predictions, labels, _ = predict(probe, test_loader, device)
        summary = classification_summary(labels, predictions, cfg.class_names)
        print(f"\n[{variant}] linear probe on test split\n{summary['report']}")
        rows.append({"variant": variant, "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]})

        histories[variant].to_csv(cfg.output_dir / f"probe_history_{variant}.csv", index=False)
        save_confusion_matrix(summary["confusion"], cfg.class_names, f"{variant}: confusion matrix",
                              plots_dir / f"confusion_{variant}.png")
        features, feature_labels = extract_embeddings(probe, test_loader)
        names = [cfg.class_names[i] for i in feature_labels]
        save_pca_2d(features, names, f"{variant}: GAP features (test)", plots_dir / f"pca2d_{variant}.png",
                    legend_title="plot status")
        save_pca_3d(features, names, f"{variant}: GAP features (test)", plots_dir / f"pca3d_{variant}.html",
                    legend_title="plot status")

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(cfg.output_dir / "probe_summary.csv", index=False)
    save_probe_loss_curves(histories, plots_dir / "probe_loss_curves.png")
    save_metric_bars(summary_frame, PROBE_METRICS, "Plot-status linear probe: LeJEPA vs COCO",
                     plots_dir / "probe_metrics.png")
    return summary_frame


def run_all(pretrain_cfg: PretrainConfig, detection_cfg: Optional[DetectionEvalConfig] = None,
            probe_cfg: Optional[ProbeEvalConfig] = None) -> Dict[str, object]:
    """Pretrain, then run every evaluation that has a configuration.

    The LeJEPA checkpoint path of each evaluation configuration is pointed at
    the checkpoint that was just trained.

    Parameters
    ----------
    pretrain_cfg : PretrainConfig
        Pretraining configuration.
    detection_cfg : DetectionEvalConfig or None, default=None
        Detection evaluation; skipped when ``None``.
    probe_cfg : ProbeEvalConfig or None, default=None
        Linear-probe evaluation; skipped when ``None``.

    Returns
    -------
    dict
        ``checkpoint``, ``history`` and, when run, ``detection`` and
        ``probe`` summaries.
    """
    checkpoint, history = run_pretraining(pretrain_cfg)
    results: Dict[str, object] = {"checkpoint": checkpoint, "history": history}
    for key, cfg, runner in (("detection", detection_cfg, run_detection_evaluation),
                             ("probe", probe_cfg, run_probe_evaluation)):
        if cfg is not None:
            cfg.weights.lejepa_checkpoint = checkpoint
            results[key] = runner(cfg)
    return results