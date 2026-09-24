"""Downstream evaluation of LeJEPA-pretrained vs COCO-pretrained YOLOv8 backbones.

Two protocols are provided, both parameterised by a *variant* name:

``"lejepa"``
    Backbone from the cross-camera GAP LeJEPA checkpoint (empty-weight
    ``yolov8n.yaml`` before pretraining); neck and head random.
``"coco"``
    Official ``yolov8n.pt`` COCO weights (backbone, neck and head).
``"coco_backbone"``
    COCO backbone only; neck and head random. This is the like-for-like
    control for ``"lejepa"`` in detection.
``"scratch"``
    Empty ``yolov8n.yaml`` weights (lower bound).

1. **Detection fine-tuning** of an Ultralytics YOLOv8 detector on labelled
   cotton-boll images, reported as precision, recall, mAP50 and mAP50-95.
2. **Linear probing** of frozen, globally average-pooled backbone features on
   the plot-status classification task.

No directory is hard coded; every path comes from the configuration objects.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from ultralytics import YOLO

from .core_pretraining import (
    EMPTY_MODEL_CFG,
    YOLOv8MultiScaleBackbone,
    build_eval_transform,
    export_backbone_to_yolo,
    list_images,
    load_backbone,
    load_rgb,
    select_device,
    subsample,
)

VARIANTS = ("lejepa", "coco", "coco_backbone", "scratch")


# --------------------------------------------------------------------------- #
# Weight sources shared by both protocols
# --------------------------------------------------------------------------- #
@dataclass
class WeightSources:
    """Where each variant's initial weights come from.

    Parameters
    ----------
    lejepa_checkpoint : str or pathlib.Path
        Encoder checkpoint written by :func:`core_pretraining.save_encoder`.
    model_cfg : str, default="yolov8n.yaml"
        Empty-weight architecture definition.
    coco_weights : str, default="yolov8n.pt"
        Ultralytics COCO checkpoint (downloaded automatically if missing).
    """

    lejepa_checkpoint: Path
    model_cfg: str = EMPTY_MODEL_CFG
    coco_weights: str = "yolov8n.pt"

    def __post_init__(self) -> None:
        self.lejepa_checkpoint = Path(self.lejepa_checkpoint)


def validate_variant(variant: str) -> str:
    """Check that a variant name is supported.

    Parameters
    ----------
    variant : str
        Candidate name.

    Returns
    -------
    str
        The same name.

    Raises
    ------
    ValueError
        If the name is not in :data:`VARIANTS`.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANTS}.")
    return variant


def build_backbone_for_variant(variant: str, sources: WeightSources) -> YOLOv8MultiScaleBackbone:
    """Instantiate the multi-scale backbone of a variant.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.
    sources : WeightSources
        Weight locations.

    Returns
    -------
    YOLOv8MultiScaleBackbone
        Backbone initialised according to ``variant``.
    """
    validate_variant(variant)
    if variant == "lejepa":
        return load_backbone(sources.lejepa_checkpoint)
    if variant == "scratch":
        return YOLOv8MultiScaleBackbone(sources.model_cfg)
    return YOLOv8MultiScaleBackbone(sources.coco_weights)


def ultralytics_device(device: torch.device) -> str:
    """Translate a torch device into the string Ultralytics expects.

    Parameters
    ----------
    device : torch.device
        Torch device.

    Returns
    -------
    str
        ``"cpu"``, ``"mps"`` or a CUDA index such as ``"0"``.
    """
    if device.type == "cuda":
        return str(device.index or 0)
    return device.type


# --------------------------------------------------------------------------- #
# Detection: labels and dataset split
# --------------------------------------------------------------------------- #
def read_yolo_labels(path: Path) -> np.ndarray:
    """Read a YOLO text label file.

    Parameters
    ----------
    path : pathlib.Path
        ``.txt`` file with ``class xc yc w h`` rows (normalised).

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 5)``; empty when the file is missing or blank.
    """
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = [line.split()[:5] for line in path.read_text().splitlines() if len(line.split()) >= 5]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def write_yolo_labels(labels: np.ndarray, path: Path) -> Path:
    """Write labels in YOLO text format.

    Parameters
    ----------
    labels : numpy.ndarray
        Shape ``(n, 5)`` rows of ``class xc yc w h``.
    path : pathlib.Path
        Destination file.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    lines = [f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in labels]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))
    return Path(path)


def yolo_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalised YOLO boxes to pixel corner coordinates.

    Parameters
    ----------
    labels : numpy.ndarray
        Shape ``(n, 5)`` rows of ``class xc yc w h``.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 4)`` rows of ``x1 y1 x2 y2``.
    """
    xc, yc, w, h = labels[:, 1] * width, labels[:, 2] * height, labels[:, 3] * width, labels[:, 4] * height
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def find_labeled_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair every image with a non-empty YOLO label file.

    Images are read from ``source_dir/images`` and labels from
    ``source_dir/labels`` when those folders exist, otherwise both come from
    ``source_dir`` itself.

    Parameters
    ----------
    source_dir : pathlib.Path
        Root of the labelled detection dataset.

    Returns
    -------
    list of tuple of pathlib.Path
        ``(image_path, label_path)`` pairs.

    Raises
    ------
    RuntimeError
        If no labelled image is found.
    """
    source_dir = Path(source_dir)
    image_dir = source_dir / "images" if (source_dir / "images").exists() else source_dir
    label_dir = source_dir / "labels" if (source_dir / "labels").exists() else source_dir
    pairs = [(img, label_dir / f"{img.stem}.txt") for img in list_images(image_dir)]
    pairs = [(img, lbl) for img, lbl in pairs if len(read_yolo_labels(lbl))]
    if not pairs:
        raise RuntimeError(f"No labelled images found in {source_dir}")
    return pairs


def copy_pairs(pairs: Sequence[Tuple[Path, Path]], image_dir: Path, label_dir: Path,
               single_class: bool) -> None:
    """Copy image/label pairs into a split folder.

    Parameters
    ----------
    pairs : sequence of tuple of pathlib.Path
        ``(image_path, label_path)`` pairs.
    image_dir : pathlib.Path
        Destination image folder.
    label_dir : pathlib.Path
        Destination label folder.
    single_class : bool
        Rewrite every class id to ``0``.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        labels = read_yolo_labels(label_path)
        if single_class:
            labels[:, 0] = 0
        write_yolo_labels(labels, label_dir / f"{image_path.stem}.txt")


def prepare_detection_split(
    source_dir: Path,
    split_dir: Path,
    class_names: Sequence[str],
    val_ratio: float = 0.2,
    subset_ratio: float = 1.0,
    seed: int = 42,
) -> Path:
    """Create a fresh train/val split and its Ultralytics ``dataset.yaml``.

    Parameters
    ----------
    source_dir : pathlib.Path
        Labelled detection dataset.
    split_dir : pathlib.Path
        Output folder (deleted and recreated). Keep it outside ``source_dir``.
    class_names : sequence of str
        Class names; a single name forces every label to class ``0``.
    val_ratio : float, default=0.2
        Fraction of pairs used for validation.
    subset_ratio : float, default=1.0
        Fraction of all labelled pairs to keep.
    seed : int, default=42
        Seed for subsetting and splitting.

    Returns
    -------
    pathlib.Path
        Path of the written ``dataset.yaml``.
    """
    split_dir = Path(split_dir)
    if split_dir.exists():
        shutil.rmtree(split_dir)
    pairs = find_labeled_pairs(source_dir)
    pairs = subsample(pairs, max(1, int(len(pairs) * subset_ratio)), seed)
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_ratio, random_state=seed)
    single_class = len(class_names) == 1
    for name, subset in (("train", train_pairs), ("val", val_pairs)):
        copy_pairs(subset, split_dir / "images" / name, split_dir / "labels" / name, single_class)
    print(f"Detection split: {len(train_pairs)} train | {len(val_pairs)} val -> {split_dir}")

    yaml_path = split_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(split_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(class_names)),
    }))
    return yaml_path


# --------------------------------------------------------------------------- #
# Detection: fine-tuning and validation
# --------------------------------------------------------------------------- #
@dataclass
class DetectionEvalConfig:
    """Settings for comparing variants by detector fine-tuning.

    Parameters
    ----------
    source_dir : str or pathlib.Path
        Labelled detection dataset (``images/`` and ``labels/``).
    output_dir : str or pathlib.Path
        Folder for the split, initial weights, runs and plots.
    weights : WeightSources
        Initial-weight locations.
    class_names : tuple of str, default=("cotton_boll",)
        Detection classes.
    variants : tuple of str, default=("lejepa", "coco")
        Variants to fine-tune and compare.
    val_ratio : float, default=0.2
        Validation fraction.
    subset_ratio : float, default=1.0
        Fraction of labelled images used.
    image_size : int, default=256
        Training and validation image size.
    batch_size : int, default=16
        Batch size.
    epochs : int, default=40
        Fine-tuning epochs.
    optimizer : str, default="AdamW"
        Ultralytics optimiser name.
    lr0 : float, default=0.002
        Initial learning rate.
    lrf : float, default=0.01
        Final learning-rate fraction.
    weight_decay : float, default=1e-4
        Weight decay.
    freeze : int or None, default=None
        Number of leading layers to freeze (``10`` freezes the backbone).
    close_mosaic : int, default=10
        Epochs at the end without mosaic augmentation.
    patience : int, default=20
        Early-stopping patience.
    workers : int, default=0
        Data-loading workers.
    amp : bool, default=False
        Mixed precision (disabled for MPS/CPU stability).
    conf : float, default=0.25
        Confidence threshold for overlay visualisations only.
    seed : int, default=42
        Seed for splitting and training.
    device : str or None, default=None
        Torch device string; ``None`` selects automatically.
    train_overrides : dict, default={}
        Extra keyword arguments forwarded to ``YOLO.train``.
    """

    source_dir: Path
    output_dir: Path
    weights: WeightSources
    class_names: Tuple[str, ...] = ("cotton_boll",)
    variants: Tuple[str, ...] = ("lejepa", "coco")
    val_ratio: float = 0.2
    subset_ratio: float = 1.0
    image_size: int = 256
    batch_size: int = 16
    epochs: int = 40
    optimizer: str = "AdamW"
    lr0: float = 0.002
    lrf: float = 0.01
    weight_decay: float = 1e-4
    freeze: Optional[int] = None
    close_mosaic: int = 10
    patience: int = 20
    workers: int = 0
    amp: bool = False
    conf: float = 0.25
    seed: int = 42
    device: Optional[str] = None
    train_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_dir = Path(self.source_dir)
        self.output_dir = Path(self.output_dir)
        self.variants = tuple(validate_variant(v) for v in self.variants)

    @property
    def split_dir(self) -> Path:
        """pathlib.Path : Folder of the generated train/val split."""
        return self.output_dir / "split_dataset"

    @property
    def runs_dir(self) -> Path:
        """pathlib.Path : Ultralytics ``project`` folder."""
        return self.output_dir / "runs"

    @property
    def init_dir(self) -> Path:
        """pathlib.Path : Folder of exported initial checkpoints."""
        return self.output_dir / "init_weights"


def detection_init_weights(variant: str, sources: WeightSources, export_dir: Path) -> str:
    """Return the weights ``YOLO(...)`` should start from for a variant.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.
    sources : WeightSources
        Weight locations.
    export_dir : pathlib.Path
        Folder where backbone-only checkpoints are written.

    Returns
    -------
    str
        A ``.yaml`` definition or a ``.pt`` checkpoint path.
    """
    validate_variant(variant)
    if variant == "coco":
        return str(sources.coco_weights)
    if variant == "scratch":
        return str(sources.model_cfg)
    backbone = build_backbone_for_variant(variant, sources)
    return str(export_backbone_to_yolo(backbone, Path(export_dir) / f"{variant}_init.pt", sources.model_cfg))


def finetune_detector(init_weights: str, data_yaml: Path, cfg: DetectionEvalConfig,
                      run_name: str, device: str) -> Path:
    """Fine-tune a YOLOv8 detector.

    Parameters
    ----------
    init_weights : str
        Starting weights from :func:`detection_init_weights`.
    data_yaml : pathlib.Path
        Ultralytics dataset file.
    cfg : DetectionEvalConfig
        Training settings.
    run_name : str
        Run folder name inside ``cfg.runs_dir``.
    device : str
        Ultralytics device string.

    Returns
    -------
    pathlib.Path
        The run folder (contains ``weights/best.pt`` and ``results.csv``).
    """
    model = YOLO(init_weights)
    model.train(
        data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.image_size, batch=cfg.batch_size,
        device=device, optimizer=cfg.optimizer, lr0=cfg.lr0, lrf=cfg.lrf,
        weight_decay=cfg.weight_decay, freeze=cfg.freeze, close_mosaic=cfg.close_mosaic,
        patience=cfg.patience, workers=cfg.workers, amp=cfg.amp, seed=cfg.seed,
        project=str(cfg.runs_dir), name=run_name, exist_ok=True, plots=True, val=True,
        **cfg.train_overrides,
    )
    return Path(model.trainer.save_dir)


def validate_detector(weights: Path, data_yaml: Path, cfg: DetectionEvalConfig,
                      run_name: str, device: str) -> Dict[str, float]:
    """Compute detection metrics on the validation split.

    The default low confidence threshold of Ultralytics is kept so mAP is
    integrated over the full precision-recall curve.

    Parameters
    ----------
    weights : pathlib.Path
        Trained detector (``best.pt``).
    data_yaml : pathlib.Path
        Ultralytics dataset file.
    cfg : DetectionEvalConfig
        Evaluation settings.
    run_name : str
        Name of the validation output folder.
    device : str
        Ultralytics device string.

    Returns
    -------
    dict of str to float
        ``precision``, ``recall``, ``mAP50`` and ``mAP50-95``.
    """
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), imgsz=cfg.image_size, batch=cfg.batch_size, device=device,
        split="val", plots=False, project=str(cfg.runs_dir), name=f"{run_name}_val", exist_ok=True,
    )
    box = metrics.box
    return {"precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50-95": float(box.map)}


def load_training_log(run_dir: Path) -> pd.DataFrame:
    """Read an Ultralytics ``results.csv`` with clean column names.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run folder returned by :func:`finetune_detector`.

    Returns
    -------
    pandas.DataFrame
        Per-epoch training log.
    """
    log = pd.read_csv(Path(run_dir) / "results.csv")
    log.columns = [c.strip() for c in log.columns]
    return log


# --------------------------------------------------------------------------- #
# Linear probe: data
# --------------------------------------------------------------------------- #
class PlotStatusDataset(Dataset):
    """Images labelled through a ``{file_name: label}`` JSON file.

    Parameters
    ----------
    image_dir : pathlib.Path
        Folder searched recursively for images.
    annotation_path : pathlib.Path
        JSON mapping image file names to string labels.
    label_mapping : dict of str to int
        String label to class index.
    transform : callable
        Transform applied to each PIL image.
    """

    def __init__(self, image_dir: Path, annotation_path: Path, label_mapping: Dict[str, int],
                 transform) -> None:
        annotations = json.loads(Path(annotation_path).read_text())
        self.transform = transform
        self.samples = [
            (path, label_mapping[annotations[path.name]])
            for path in list_images(image_dir)
            if annotations.get(path.name) in label_mapping
        ]
        if not self.samples:
            raise RuntimeError(f"No annotated images found in {image_dir}")

    @property
    def targets(self) -> List[int]:
        """list of int : Class index of every sample."""
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        """Return the number of annotated images.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load one image and its class index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        image : torch.Tensor
            Transformed image of shape ``(3, H, W)``.
        label : int
            Class index.
        """
        path, label = self.samples[idx]
        return self.transform(load_rgb(path)), label


@dataclass
class ProbeEvalConfig:
    """Settings for comparing variants with a frozen-backbone linear probe.

    Parameters
    ----------
    image_dir : str or pathlib.Path
        Folder with plot-status images.
    annotation_path : str or pathlib.Path
        JSON file mapping image names to labels.
    output_dir : str or pathlib.Path
        Folder for histories, metrics and plots.
    weights : WeightSources
        Initial-weight locations.
    label_mapping : dict of str to int
        String label to class index.
    variants : tuple of str, default=("lejepa", "coco")
        Variants to probe.
    scales : tuple of str, default=("P3", "P4", "P5")
        Scales whose GAP vectors are concatenated as probe input.
    image_size : int, default=256
        Input image size.
    batch_size : int, default=16
        Batch size.
    epochs : int, default=50
        Probe training epochs.
    lr : float, default=0.017711616697652244
        Adam learning rate.
    weight_decay : float, default=1.009744159203988e-06
        Adam weight decay.
    dropout : float, default=0.3
        Dropout before the linear layer.
    test_ratio : float, default=0.5
        Stratified test fraction.
    num_workers : int, default=0
        DataLoader workers.
    seed : int, default=48
        Seed for splitting and training.
    device : str or None, default=None
        Torch device string; ``None`` selects automatically.
    """

    image_dir: Path
    annotation_path: Path
    output_dir: Path
    weights: WeightSources
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {"headland": 0, "between_plots": 1, "in_plot": 2})
    variants: Tuple[str, ...] = ("lejepa", "coco")
    scales: Tuple[str, ...] = ("P3", "P4", "P5")
    image_size: int = 256
    batch_size: int = 16
    epochs: int = 50
    lr: float = 0.017711616697652244
    weight_decay: float = 1.009744159203988e-06
    dropout: float = 0.3
    test_ratio: float = 0.5
    num_workers: int = 0
    seed: int = 48
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.image_dir = Path(self.image_dir)
        self.annotation_path = Path(self.annotation_path)
        self.output_dir = Path(self.output_dir)
        self.variants = tuple(validate_variant(v) for v in self.variants)

    @property
    def class_names(self) -> List[str]:
        """list of str : Class names ordered by index."""
        return sorted(self.label_mapping, key=self.label_mapping.get)


def build_probe_loaders(cfg: ProbeEvalConfig) -> Tuple[DataLoader, DataLoader]:
    """Load the plot-status dataset and split it with stratification.

    Parameters
    ----------
    cfg : ProbeEvalConfig
        Probe configuration.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Shuffled training loader.
    test_loader : torch.utils.data.DataLoader
        Ordered test loader.
    """
    dataset = PlotStatusDataset(cfg.image_dir, cfg.annotation_path, cfg.label_mapping,
                                build_eval_transform(cfg.image_size))
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=cfg.test_ratio,
                                           stratify=dataset.targets, random_state=cfg.seed)
    print(f"Probe split: {len(train_idx)} train | {len(test_idx)} test")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Linear probe: model and training
# --------------------------------------------------------------------------- #
class LinearProbe(nn.Module):
    """Frozen backbone, multi-scale GAP features, dropout and a linear layer.

    Parameters
    ----------
    backbone : YOLOv8MultiScaleBackbone
        Backbone to freeze.
    num_classes : int
        Number of output classes.
    dropout : float, default=0.3
        Dropout probability before the classifier.
    scales : sequence of str, default=("P3", "P4", "P5")
        Scales whose pooled vectors are concatenated.
    """

    def __init__(self, backbone: YOLOv8MultiScaleBackbone, num_classes: int, dropout: float = 0.3,
                 scales: Sequence[str] = ("P3", "P4", "P5")) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.scales = tuple(scales)
        channels = backbone.out_channels()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sum(channels[s] for s in self.scales), num_classes)

    def train(self, mode: bool = True) -> "LinearProbe":
        """Set training mode while keeping the frozen backbone in eval mode.

        Parameters
        ----------
        mode : bool, default=True
            Training flag for the probe head.

        Returns
        -------
        LinearProbe
            ``self``.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Concatenated GAP features of the selected scales.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, sum_of_channels)``.
        """
        pooled = self.backbone.forward_pooled(images)
        return torch.cat([pooled[s] for s in self.scales], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Class logits.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, num_classes)``.
        """
        return self.classifier(self.dropout(self.embed(images)))


def train_probe_epoch(probe: LinearProbe, loader: DataLoader, optimizer: torch.optim.Optimizer,
                      device: torch.device) -> float:
    """Train the probe head for one epoch.

    Parameters
    ----------
    probe : LinearProbe
        Model with a frozen backbone.
    loader : torch.utils.data.DataLoader
        Training data.
    optimizer : torch.optim.Optimizer
        Optimiser over the head parameters.
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Mean cross-entropy over the epoch.
    """
    probe.train()
    criterion = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(probe(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(labels)
        count += len(labels)
    return total / max(1, count)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    """Predict classes and compute the mean cross-entropy.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier returning logits.
    loader : torch.utils.data.DataLoader
        Evaluation data.
    device : torch.device
        Compute device.

    Returns
    -------
    predictions : numpy.ndarray
        Predicted class indices.
    labels : numpy.ndarray
        True class indices.
    loss : float
        Mean cross-entropy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    preds, labels, total = [], [], 0.0
    for images, targets in loader:
        logits = model(images.to(device))
        total += float(criterion(logits, targets.to(device)).cpu())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(targets.numpy())
    labels_arr = np.concatenate(labels)
    return np.concatenate(preds), labels_arr, total / max(1, len(labels_arr))


def train_linear_probe(probe: LinearProbe, train_loader: DataLoader, test_loader: DataLoader,
                       cfg: ProbeEvalConfig, device: torch.device) -> pd.DataFrame:
    """Train a probe and restore the weights with the lowest test loss.

    Parameters
    ----------
    probe : LinearProbe
        Probe on ``device``.
    train_loader : torch.utils.data.DataLoader
        Training data.
    test_loader : torch.utils.data.DataLoader
        Held-out data used for monitoring and model selection.
    cfg : ProbeEvalConfig
        Probe configuration.
    device : torch.device
        Compute device.

    Returns
    -------
    pandas.DataFrame
        Columns ``epoch``, ``train_loss`` and ``test_loss``.
    """
    optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss, best_state, records = float("inf"), None, []
    for epoch in range(cfg.epochs):
        train_loss = train_probe_epoch(probe, train_loader, optimizer, device)
        _, _, test_loss = predict(probe, test_loader, device)
        records.append({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})
        if test_loss < best_loss:
            best_loss, best_state = test_loss, deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    return pd.DataFrame(records)


def fit_probe_variant(variant: str, cfg: ProbeEvalConfig, train_loader: DataLoader,
                      test_loader: DataLoader) -> Tuple[LinearProbe, pd.DataFrame]:
    """Build and train the linear probe of one variant.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.
    cfg : ProbeEvalConfig
        Probe configuration.
    train_loader : torch.utils.data.DataLoader
        Training data.
    test_loader : torch.utils.data.DataLoader
        Test data.

    Returns
    -------
    probe : LinearProbe
        Trained probe (best test-loss weights).
    history : pandas.DataFrame
        Per-epoch losses.
    """
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    backbone = build_backbone_for_variant(variant, cfg.weights)
    probe = LinearProbe(backbone, len(cfg.label_mapping), cfg.dropout, cfg.scales).to(device)
    return probe, train_linear_probe(probe, train_loader, test_loader, cfg, device)


def classification_summary(labels: np.ndarray, predictions: np.ndarray,
                           class_names: Sequence[str]) -> Dict[str, object]:
    """Accuracy, macro F1, text report and confusion matrix.

    Parameters
    ----------
    labels : numpy.ndarray
        True class indices.
    predictions : numpy.ndarray
        Predicted class indices.
    class_names : sequence of str
        Names ordered by class index.

    Returns
    -------
    dict
        Keys ``accuracy``, ``macro_f1``, ``report`` and ``confusion``.
    """
    indices = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=indices, zero_division=0)),
        "report": classification_report(labels, predictions, labels=indices, target_names=list(class_names),
                                        digits=4, zero_division=0),
        "confusion": confusion_matrix(labels, predictions, labels=indices),
    }


# --------------------------------------------------------------------------- #
# Embeddings (shared by pretraining diagnostics and probing)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader,
                       max_batches: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Collect ``model.embed`` outputs and labels from a loader.

    Multi-view batches of shape ``(B, V, 3, H, W)`` are flattened so every
    view becomes one embedding, with its per-view label (e.g. camera id).

    Parameters
    ----------
    model : torch.nn.Module
        Module exposing ``embed(images) -> (N, D)``.
    loader : torch.utils.data.DataLoader
        Yields ``(images, labels)``.
    max_batches : int or None, default=None
        Stop after this many batches.

    Returns
    -------
    features : numpy.ndarray
        Shape ``(N, D)``.
    labels : numpy.ndarray
        Shape ``(N,)``.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    features, labels = [], []
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        if images.dim() == 5:
            images, targets = images.flatten(0, 1), targets.flatten()
        features.append(model.embed(images.to(device)).cpu().numpy())
        labels.append(np.asarray(targets))
    model.train(was_training)
    return np.concatenate(features), np.concatenate(labels)
