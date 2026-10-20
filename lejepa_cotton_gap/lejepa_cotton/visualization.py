"""Plots for pretraining diagnostics and LeJEPA-vs-COCO comparisons.

Every function draws exactly one artefact and writes it to the path given by
the caller; nothing is shown interactively and no directory is hard coded.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import ImageDraw
from sklearn.decomposition import PCA
from ultralytics import YOLO

from .core_evaluation import load_training_log, read_yolo_labels, yolo_to_xyxy
from .core_pretraining import (
    IMAGENET_MEAN, IMAGENET_STD, PretrainConfig, build_pretraining_dataset, list_images, load_rgb,
)

DETECTION_CURVE_METRICS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 200) -> Path:
    """Save a Matplotlib figure, creating parent folders, and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    out_path : pathlib.Path
        Destination image file.
    dpi : int, default=200
        Output resolution.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def pca_project(features: np.ndarray, n_components: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Project features onto their leading principal components.

    Parameters
    ----------
    features : numpy.ndarray
        Shape ``(N, D)``.
    n_components : int
        Number of components.
    seed : int, default=0
        PCA random state.

    Returns
    -------
    coords : numpy.ndarray
        Shape ``(N, n_components)``.
    explained : numpy.ndarray
        Explained-variance percentage per component.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(features), pca.explained_variance_ratio_ * 100


def save_pca_2d(features: np.ndarray, labels: Optional[Sequence[str]], title: str, out_path: Path,
                legend_title: str = "label") -> Path:
    """Static 2-D PCA scatter coloured by label.

    Parameters
    ----------
    features : numpy.ndarray
        Shape ``(N, D)``.
    labels : sequence of str or None
        Group name of each row; ``None`` draws a single group.
    title : str
        Figure title.
    out_path : pathlib.Path
        Destination ``.png``.
    legend_title : str, default="label"
        Legend heading.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    coords, explained = pca_project(features, 2)
    labels = np.asarray(labels if labels is not None else ["all"] * len(coords)).astype(str)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in np.unique(labels):
        mask = labels == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=name)
    ax.set(title=title, xlabel=f"PC1 ({explained[0]:.1f}%)", ylabel=f"PC2 ({explained[1]:.1f}%)")
    ax.legend(title=legend_title)
    ax.grid(True, linestyle="--", alpha=0.4)
    return save_figure(fig, out_path)


def save_pca_3d(features: np.ndarray, labels: Optional[Sequence[str]], title: str, out_path: Path,
                legend_title: str = "label") -> Path:
    """Interactive 3-D PCA scatter saved as HTML.

    Parameters
    ----------
    features : numpy.ndarray
        Shape ``(N, D)``.
    labels : sequence of str or None
        Group name of each row; ``None`` draws a single colour.
    title : str
        Figure title.
    out_path : pathlib.Path
        Destination ``.html``.
    legend_title : str, default="label"
        Legend heading.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    coords, explained = pca_project(features, 3)
    frame = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    if labels is not None:
        frame[legend_title] = np.asarray(labels).astype(str)
    fig = px.scatter_3d(
        frame, x="PC1", y="PC2", z="PC3", color=legend_title if labels is not None else None, opacity=0.75,
        title=f"{title}<br><sup>Variance: " + ", ".join(f"PC{i + 1}={v:.1f}%" for i, v in enumerate(explained))
              + "</sup>",
    )
    fig.update_traces(marker={"size": 3})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def save_pretraining_loss_curves(history: pd.DataFrame, out_path: Path) -> Path:
    """Plot the total, prediction and SIGReg losses of pretraining.

    Parameters
    ----------
    history : pandas.DataFrame
        Output of :func:`core_pretraining.pretrain`.
    out_path : pathlib.Path
        Destination ``.png``.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    panels = {"Total LeJEPA loss": ["total"],
              "Cross-camera prediction": sorted(c for c in history if c.startswith("pred")),
              "SIGReg": sorted(c for c in history if c.startswith("sigreg"))}
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (title, columns) in zip(axes, panels.items()):
        for column in columns:
            ax.plot(history["epoch"], history[column], label=column)
        ax.set(title=title, xlabel="Epoch", ylabel="Loss")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    return save_figure(fig, out_path)


def save_lambda_loss_curves(histories: Dict[float, pd.DataFrame], out_path: Path) -> Path:
    """Plot total, prediction and SIGReg loss per epoch, one line per lambda.

    Parameters
    ----------
    histories : dict of float to pandas.DataFrame
        Lambda to loss history (columns ``epoch``, ``total``, ``pred``,
        ``sigreg``).
    out_path : pathlib.Path
        Destination ``.png``.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    panels = {"Total loss": "total", "Prediction loss": "pred", "SIGReg": "sigreg"}
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (title, column) in zip(axes, panels.items()):
        for lam, history in sorted(histories.items()):
            ax.plot(history["epoch"], history[column], label=f"$\\lambda$={lam:g}")
        ax.set(title=title, xlabel="Epoch", ylabel="Loss")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    return save_figure(fig, out_path)


def save_metric_vs_lambda(summary: pd.DataFrame, out_path: Path, baseline: str = "coco_backbone") -> Path:
    """Line plot of mAP50 and mAP50-95 against lambda with a COCO baseline.

    Parameters
    ----------
    summary : pandas.DataFrame
        One row per model with columns ``variant``, ``lambda`` (NaN for the
        baseline), ``mAP50`` and ``mAP50-95``.
    out_path : pathlib.Path
        Destination ``.png``.
    baseline : str, default="coco_backbone"
        ``variant`` value of the baseline row, drawn as dashed lines.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    sweep = summary[summary["variant"] != baseline].sort_values("lambda")
    base = summary[summary["variant"] == baseline]
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in ("mAP50", "mAP50-95"):
        line, = ax.plot(sweep["lambda"], sweep[metric], marker="o", label=f"LeJEPA Backbone {metric}")
        if len(base):
            ax.axhline(float(base[metric].iloc[0]), linestyle="--", color=line.get_color(),
                       label=f"COCO Backbone {metric}")
    ax.set(title="Detection accuracy vs $\\lambda$", xlabel="$\\lambda$ (SIGReg weight)", ylabel="Score")
    ax.set_xticks(sweep["lambda"])
    ax.grid(True, alpha=0.4)
    ax.legend()
    return save_figure(fig, out_path)


def save_box_comparison(boxes_csv: Path, image_dir: Path, out_path: Path, max_images: int = 4) -> Path:
    """Draw ground-truth (green) and predicted (red) boxes from a saved box file.

    Parameters
    ----------
    boxes_csv : pathlib.Path
        File written by :func:`core_evaluation.save_box_predictions`.
    image_dir : pathlib.Path
        Folder holding the images named by ``image_id``.
    out_path : pathlib.Path
        Destination ``.png`` (one panel per image).
    max_images : int, default=4
        Number of images drawn.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    boxes = pd.read_csv(boxes_csv)
    image_ids = list(dict.fromkeys(boxes["image_id"]))[:max_images]
    fig, axes = plt.subplots(1, len(image_ids), figsize=(5 * len(image_ids), 5), squeeze=False)
    for ax, image_id in zip(axes[0], image_ids):
        image = load_rgb(Path(image_dir) / image_id)
        draw = ImageDraw.Draw(image)
        for row in boxes[boxes["image_id"] == image_id].itertuples():
            if not np.isnan(row.gt_x1):
                draw.rectangle([row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2], outline=(0, 200, 0), width=3)
            if not np.isnan(row.pred_x1):
                draw.rectangle([row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2],
                               outline=(220, 0, 0), width=3)
        ax.imshow(image)
        ax.set_title(f"{image_id} (green=GT, red=pred)", fontsize=8)
        ax.axis("off")
    return save_figure(fig, out_path)


def save_probe_loss_curves(histories: Dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Overlay linear-probe train and test losses of several variants.

    Parameters
    ----------
    histories : dict of str to pandas.DataFrame
        Variant name to output of :func:`core_evaluation.train_linear_probe`.
    out_path : pathlib.Path
        Destination ``.png``.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, history in histories.items():
        line, = ax.plot(history["epoch"], history["train_loss"], label=f"{variant} train")
        ax.plot(history["epoch"], history["test_loss"], linestyle="--", color=line.get_color(),
                label=f"{variant} test")
    ax.set(title="Linear probe loss", xlabel="Epoch", ylabel="Cross-entropy")
    ax.grid(True, alpha=0.4)
    ax.legend()
    return save_figure(fig, out_path)


def save_detection_curves(run_dirs: Dict[str, Path], out_path: Path,
                          metrics: Sequence[str] = DETECTION_CURVE_METRICS) -> Path:
    """Overlay per-epoch validation metrics of several fine-tuning runs.

    Parameters
    ----------
    run_dirs : dict of str to pathlib.Path
        Variant name to Ultralytics run folder.
    out_path : pathlib.Path
        Destination ``.png``.
    metrics : sequence of str, default=DETECTION_CURVE_METRICS
        ``results.csv`` columns to plot, one panel each.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    logs = {variant: load_training_log(run_dir) for variant, run_dir in run_dirs.items()}
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for variant, log in logs.items():
            ax.plot(log["epoch"], log[metric], label=variant)
        ax.set(title=metric, xlabel="Epoch", ylabel=metric.split("/")[-1])
        ax.grid(True, alpha=0.4)
        ax.legend()
    return save_figure(fig, out_path)


def save_metric_bars(summary: pd.DataFrame, metrics: Sequence[str], title: str, out_path: Path,
                     index: str = "variant", figsize: Tuple[float, float] = (8, 5)) -> Path:
    """Grouped bar chart comparing final metrics across variants.

    Parameters
    ----------
    summary : pandas.DataFrame
        One row per variant.
    metrics : sequence of str
        Columns to plot.
    title : str
        Figure title.
    out_path : pathlib.Path
        Destination ``.png``.
    index : str, default="variant"
        Column naming each row.
    figsize : tuple of float, default=(8, 5)
        Figure size in inches.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    ax = summary.set_index(index)[list(metrics)].T.plot.bar(figsize=figsize, rot=0)
    ax.set(title=title, ylabel="Score", ylim=(0, 1))
    ax.grid(True, axis="y", alpha=0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    return save_figure(ax.figure, out_path)


def save_confusion_matrix(confusion: np.ndarray, class_names: Sequence[str], title: str,
                          out_path: Path) -> Path:
    """Annotated confusion-matrix heat map.

    Parameters
    ----------
    confusion : numpy.ndarray
        Square count matrix (rows true, columns predicted).
    class_names : sequence of str
        Names ordered by class index.
    title : str
        Figure title.
    out_path : pathlib.Path
        Destination ``.png``.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ticks = range(len(class_names))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    threshold = confusion.max() / 2 if confusion.size else 0
    for (row, col), value in np.ndenumerate(confusion):
        ax.text(col, row, int(value), ha="center", va="center",
                color="white" if value > threshold else "black")
    return save_figure(fig, out_path)


def save_detection_overlays(weights: Path, image_dir: Path, label_dir: Path, out_dir: Path,
                            image_size: int, conf: float, device: str, max_images: int = 20) -> List[Path]:
    """Draw ground-truth (green) and predicted (red) boxes on validation images.

    Parameters
    ----------
    weights : pathlib.Path
        Trained detector.
    image_dir : pathlib.Path
        Validation images.
    label_dir : pathlib.Path
        Validation YOLO labels.
    out_dir : pathlib.Path
        Destination folder.
    image_size : int
        Inference size.
    conf : float
        Confidence threshold for drawn predictions.
    device : str
        Ultralytics device string.
    max_images : int, default=20
        Number of images to render.

    Returns
    -------
    list of pathlib.Path
        Written overlay images.
    """
    model = YOLO(str(weights))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for image_path in list_images(image_dir)[:max_images]:
        image = load_rgb(image_path)
        draw = ImageDraw.Draw(image)
        truth = yolo_to_xyxy(read_yolo_labels(Path(label_dir) / f"{image_path.stem}.txt"), *image.size)
        for box in truth:
            draw.rectangle(box.tolist(), outline=(0, 200, 0), width=2)
        boxes = model.predict(str(image_path), imgsz=image_size, conf=conf, device=device, verbose=False)[0].boxes
        for box, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            draw.rectangle(box.tolist(), outline=(220, 0, 0), width=2)
            draw.text((box[0], box[3] + 2), f"{score:.2f}", fill=(220, 0, 0))
        written.append(out_dir / f"overlay_{image_path.name}")
        image.save(written[-1])
    return written


def save_pretraining_view_examples(cfg: PretrainConfig, out_path: Path, num_samples: int = 3) -> Path:
    """Draw the final views fed to the encoder, as used by the prediction loss.

    Each row is one training sample; each column is one view, labelled with its
    source camera (``cross_camera``) or view index (``intra_image``).

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration (selects the mode, cameras and view size).
    out_path : pathlib.Path
        Destination image file.
    num_samples : int, default=3
        Number of samples (rows) shown.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    dataset = build_pretraining_dataset(cfg, verbose=False)
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    num_views = samples[0][0].shape[0]
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    fig, axes = plt.subplots(len(samples), num_views, figsize=(3.2 * num_views, 3.4 * len(samples)), squeeze=False)
    for row, (views, camera_ids) in enumerate(samples):
        for col in range(num_views):
            image = np.clip(views[col].numpy() * std + mean, 0, 1).transpose(1, 2, 0)
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if cfg.pred_loss_mode == "intra_image":
                ax.set_title(f"Sample {row + 1} | View {col + 1}\n(camera {int(camera_ids[col])}, "
                             f"random flips)", fontsize=9)
            else:
                ax.set_title(f"Sample {row + 1} | View {col + 1}\ncamera {int(camera_ids[col])}", fontsize=9)
    fig.suptitle(f"Final views entering the encoder (mode: {cfg.pred_loss_mode}, "
                 f"{cfg.image_size}x{cfg.image_size}, de-normalised for display).\n"
                 "The prediction loss pulls the embeddings of the views in each row together.",
                 fontsize=11)
    return save_figure(fig, out_path)
