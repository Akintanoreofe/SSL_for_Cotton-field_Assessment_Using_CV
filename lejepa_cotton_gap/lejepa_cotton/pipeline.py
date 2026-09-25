"""End-to-end runners that wire pretraining, evaluation and visualization.

These functions are the only place where the three core modules meet. They
decide *which* artefacts to write, always inside the ``output_dir`` supplied
through the configuration objects. Notebooks and scripts should only build
configurations and call these runners.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from .core_evaluation import (
    DetectionEvalConfig,
    ProbeEvalConfig,
    build_probe_loaders,
    classification_summary,
    detection_init_weights,
    extract_embeddings,
    finetune_detector,
    fit_probe_variant,
    predict,
    prepare_detection_split,
    ultralytics_device,
    validate_detector,
)
from .core_pretraining import (
    PretrainConfig,
    YOLOv8GAPEncoder,
    build_pretraining_loader,
    pretrain,
    save_encoder,
    select_device,
)
from .visualization import (
    save_confusion_matrix,
    save_detection_curves,
    save_detection_overlays,
    save_metric_bars,
    save_pca_2d,
    save_pca_3d,
    save_pretraining_loss_curves,
    save_probe_loss_curves,
)

DETECTION_METRICS = ("precision", "recall", "mAP50", "mAP50-95")
PROBE_METRICS = ("accuracy", "macro_f1")


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
                       f"Cross-camera GAP LeJEPA embeddings - epoch {epoch + 1}",
                       Path(out_dir) / f"pca_epoch_{epoch + 1:03d}.html", legend_title="camera")


def run_pretraining(cfg: PretrainConfig, pca_epochs: Optional[Iterable[int]] = None,
                    pca_max_batches: int = 60) -> Tuple[Path, pd.DataFrame]:
    """Pretrain, then save the checkpoint, loss history and plots.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    pca_epochs : iterable of int or None, default=None
        Zero-based epochs to visualise; defaults to first, middle and last.
    pca_max_batches : int, default=60
        Batches used per PCA snapshot.

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

    checkpoint = save_encoder(encoder, cfg.checkpoint_path, cfg)
    history.to_csv(cfg.output_dir / "loss_history.csv", index=False)
    save_pretraining_loss_curves(history, plots_dir / "loss_curves.png")
    print(f"Checkpoint: {checkpoint}")
    return checkpoint, history


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
        One row per variant with precision, recall, mAP50 and mAP50-95.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = ultralytics_device(select_device(cfg.device))
    data_yaml = prepare_detection_split(cfg.source_dir, cfg.split_dir, cfg.class_names,
                                        cfg.val_ratio, cfg.subset_ratio, cfg.seed, cfg.split_by)
    rows, run_dirs = [], {}
    for variant in cfg.variants:
        init_weights = detection_init_weights(variant, cfg.weights, cfg.init_dir)
        run_dirs[variant] = finetune_detector(init_weights, data_yaml, cfg, variant, device)
        best = run_dirs[variant] / "weights" / "best.pt"
        rows.append({"variant": variant, **validate_detector(best, data_yaml, cfg, variant, device)})
        save_detection_overlays(best, cfg.split_dir / "images" / "val", cfg.split_dir / "labels" / "val",
                                cfg.output_dir / "overlays" / variant, cfg.image_size, cfg.conf, device,
                                max_overlays)

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