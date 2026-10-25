"""Downstream evaluation of LeJEPA-pretrained vs COCO-pretrained YOLOv8 backbones.

Two protocols are provided, both parameterised by a *variant* name:

``"lejepa_backbone"``
    Backbone from a LeJEPA checkpoint, with neck transfer controlled by its
    saved pretraining configuration; the Detect head remains random.
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

import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner

from .core_pretraining import (
    EMPTY_MODEL_CFG,
    YOLOv8MultiScaleBackbone,
    build_eval_transform,
    export_backbone_to_yolo,
    list_images,
    load_backbone,
    read_pretrain_config,
    load_rgb,
    select_device,
    validate_image_size,
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
        Set this to a local ``.pt`` path to use a previously downloaded
        checkpoint or to work offline.
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
    return YOLOv8MultiScaleBackbone(str(sources.coco_weights))


def run_assigner_on_cpu() -> None:
    """Run the Ultralytics task-aligned assigner on the CPU (idempotent).

    Masked indexing in the assigner returns inconsistent sizes on Apple MPS
    ("shape mismatch: value tensor ... cannot be broadcast"), so its inputs are
    moved to the CPU and the targets moved back; the network stays on the GPU.
    """
    if getattr(TaskAlignedAssigner.forward, "_cpu_patched", False):
        return
    original = TaskAlignedAssigner.forward

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        device = pd_scores.device
        outputs = original(self, *(t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels,
                                                      gt_bboxes, mask_gt)))
        return tuple(t.to(device) for t in outputs)

    forward._cpu_patched = True
    TaskAlignedAssigner.forward = forward


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


def find_label_file(image_path: Path, unique_label_index: Dict[str, Path]) -> Optional[Path]:
    """Locate the YOLO label file that belongs to an image.

    Three layouts are recognised, in this order:

    1. Ultralytics/Roboflow convention: the last ``images`` folder in the
       image path is replaced by ``labels`` (e.g. ``014/images/a.jpg`` ->
       ``014/labels/a.txt`` or ``train/images/a.jpg`` -> ``train/labels/a.txt``).
    2. A ``.txt`` file next to the image with the same stem.
    3. A ``.txt`` file with the same stem elsewhere in the dataset, used only
       when that stem is unique so labels are never borrowed from another
       sub-dataset.

    Parameters
    ----------
    image_path : pathlib.Path
        Image file.
    unique_label_index : dict of str to pathlib.Path
        Map from file stem to label path for ``.txt`` stems that occur once
        in the dataset.

    Returns
    -------
    pathlib.Path or None
        The label file, or ``None`` if no candidate exists.
    """
    parts = image_path.parts
    if "images" in parts:
        cut = len(parts) - 1 - parts[::-1].index("images")
        candidate = Path(*parts[:cut], "labels", *parts[cut + 1:]).with_suffix(".txt")
        if candidate.exists():
            return candidate
    sibling = image_path.with_suffix(".txt")
    if sibling.exists():
        return sibling
    return unique_label_index.get(image_path.stem)


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """MD5 digest of a file's bytes, used to detect duplicate images.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.
    chunk_size : int, default=1048576
        Bytes read per chunk.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_labeled_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair every unique image in a dataset with its non-empty YOLO label file.

    The dataset is searched recursively, so a folder of sub-datasets (e.g.
    ``014/images`` + ``014/labels``, ``ssl_active_1/images`` + ...), a single
    ``images/`` + ``labels/`` folder, or a flat folder all work. Stray ``.txt``
    files such as split lists (``014.txt``) are ignored because no image
    matches them. Byte-identical images that appear in several sub-datasets
    are kept once, so the same image can never land in both train and
    validation.

    Parameters
    ----------
    source_dir : pathlib.Path
        Root of the labelled detection dataset.

    Returns
    -------
    list of tuple of pathlib.Path
        ``(image_path, label_path)`` pairs, one per unique image.

    Raises
    ------
    RuntimeError
        If no image has a non-empty label file; the message reports what was
        found.
    """
    source_dir = Path(source_dir)
    images = list_images(source_dir)
    stems = Counter(path.stem for path in source_dir.rglob("*.txt"))
    unique_index = {p.stem: p for p in source_dir.rglob("*.txt") if stems[p.stem] == 1}
    matched = [(img, find_label_file(img, unique_index)) for img in images]
    matched = [(img, lbl) for img, lbl in matched if lbl is not None]
    labelled = [(img, lbl) for img, lbl in matched if len(read_yolo_labels(lbl))]
    if not labelled:
        raise RuntimeError(
            f"No labelled images found in {source_dir}: {len(images)} images, "
            f"{sum(stems.values())} .txt files, {len(matched)} images with a matching label file, "
            f"{len(matched) - len(labelled)} of them empty. Labels must be YOLO text files (class xc yc w h) "
            f"stored in a 'labels' folder next to the 'images' folder, or next to the image."
        )
    seen, pairs = set(), []
    for img, lbl in labelled:
        key = file_digest(img)
        if key not in seen:
            seen.add(key)
            pairs.append((img, lbl))
    print(f"Detection data: {len(images)} images | {len(labelled)} with boxes | "
          f"{len(labelled) - len(pairs)} duplicates removed | {len(pairs)} unique labelled images.")
    return pairs


def unique_stem(image_path: Path, source_dir: Path) -> str:
    """File stem that stays unique when images from sub-folders are merged.

    Parameters
    ----------
    image_path : pathlib.Path
        Image inside ``source_dir``.
    source_dir : pathlib.Path
        Dataset root.

    Returns
    -------
    str
        Relative folders (without ``images``) and the stem joined by ``__``,
        e.g. ``014__IMG_0001`` for ``014/images/IMG_0001.jpg``.
    """
    relative = image_path.relative_to(source_dir)
    folders = [part for part in relative.parts[:-1] if part != "images"]
    return "__".join([*folders, relative.stem])


def copy_pairs(pairs: Sequence[Tuple[Path, Path]], source_dir: Path, image_dir: Path, label_dir: Path,
               single_class: bool) -> None:
    """Copy image/label pairs into a split folder under collision-free names.

    Parameters
    ----------
    pairs : sequence of tuple of pathlib.Path
        ``(image_path, label_path)`` pairs.
    source_dir : pathlib.Path
        Dataset root, used to build unique names with :func:`unique_stem`.
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
        stem = unique_stem(image_path, source_dir)
        shutil.copy2(image_path, image_dir / f"{stem}{image_path.suffix.lower()}")
        labels = read_yolo_labels(label_path)
        if single_class:
            labels[:, 0] = 0
        write_yolo_labels(labels, label_dir / f"{stem}.txt")


def top_level_folder(image_path: Path, source_dir: Path) -> str:
    """Name of the first folder below ``source_dir`` that contains an image.

    Parameters
    ----------
    image_path : pathlib.Path
        Image inside ``source_dir``.
    source_dir : pathlib.Path
        Dataset root.

    Returns
    -------
    str
        Top-level sub-folder name, or ``"."`` for images directly in the root.
    """
    relative = image_path.relative_to(source_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "."


def split_pairs(pairs: List[Tuple[Path, Path]], source_dir: Path, val_ratio: float, seed: int,
                split_by: str = "image") -> Tuple[list, list]:
    """Split pairs into train and validation sets.

    Parameters
    ----------
    pairs : list of tuple of pathlib.Path
        ``(image_path, label_path)`` pairs.
    source_dir : pathlib.Path
        Dataset root.
    val_ratio : float
        Validation fraction (of images, or of folders when ``split_by="folder"``).
    seed : int
        Random seed.
    split_by : {"image", "folder"}, default="image"
        ``"image"`` splits images at random. ``"folder"`` keeps every
        top-level sub-folder (sequence) entirely in train or in validation,
        which prevents near-identical neighbouring frames from leaking across
        the split.

    Returns
    -------
    train_pairs, val_pairs : list
        The two subsets.
    """
    if split_by == "image":
        return train_test_split(pairs, test_size=val_ratio, random_state=seed)
    if split_by != "folder":
        raise ValueError("split_by must be 'image' or 'folder'.")
    groups = [top_level_folder(img, source_dir) for img, _ in pairs]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(pairs, groups=groups))
    print("Validation folders:", sorted({groups[i] for i in val_idx}))
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def prepare_detection_split(
    source_dir: Path,
    split_dir: Path,
    class_names: Sequence[str],
    val_ratio: float = 0.2,
    max_train_images: Optional[int] = 200,
    seed: int = 42,
    split_by: str = "image",
    test_ratio: float = 0.0,
) -> Path:
    """Create a fresh train/val(/test) split and its Ultralytics ``dataset.yaml``.

    Parameters
    ----------
    source_dir : pathlib.Path
        Labelled detection dataset (any layout accepted by
        :func:`find_labeled_pairs`).
    split_dir : pathlib.Path
        Output folder (deleted and recreated). Keep it outside ``source_dir``.
    class_names : sequence of str
        Class names; a single name forces every label to class ``0``.
    val_ratio : float, default=0.2
        Validation fraction.
    max_train_images : int or None, default=200
        Number of training images kept after splitting (low-label regime);
        ``None`` keeps them all. The validation split is never reduced, so
        every training-set size is scored on the same images.
    seed : int, default=42
        Seed for subsetting and splitting.
    split_by : {"image", "folder"}, default="image"
        Split strategy, see :func:`split_pairs`.
    test_ratio : float, default=0.0
        Fraction of the data held out as a test split (never used for tuning).
        ``val_ratio`` and ``test_ratio`` are both fractions of the full data.

    Returns
    -------
    pathlib.Path
        Path of the written ``dataset.yaml``.
    """
    source_dir, split_dir = Path(source_dir), Path(split_dir)
    if split_dir.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("output_dir must be outside the detection dataset folder.")
    if split_dir.exists():
        shutil.rmtree(split_dir)
    pairs = find_labeled_pairs(source_dir)
    test_pairs: list = []
    if test_ratio > 0:
        if val_ratio + test_ratio >= 1:
            raise ValueError("val_ratio and test_ratio must leave data for training.")
        pairs, test_pairs = split_pairs(pairs, source_dir, test_ratio, seed, split_by)
        val_ratio = val_ratio / (1.0 - test_ratio)
    train_pairs, val_pairs = split_pairs(pairs, source_dir, val_ratio, seed, split_by)
    available = len(train_pairs)
    train_pairs = subsample(train_pairs, max_train_images, seed)
    print(f"Training images: {len(train_pairs)} of {available} available")
    single_class = len(class_names) == 1
    splits = [("train", train_pairs), ("val", val_pairs)] + ([("test", test_pairs)] if test_pairs else [])
    for name, subset in splits:
        copy_pairs(subset, source_dir, split_dir / "images" / name, split_dir / "labels" / name, single_class)
    print(f"Detection split: {len(train_pairs)} train | {len(val_pairs)} val -> {split_dir}")

    yaml_path = split_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(split_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        **({"test": "images/test"} if test_pairs else {}),
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
    variants : tuple of str, default=("lejepa", "coco_backbone")
        Variants to fine-tune and compare. The default pair differs only in
        the backbone (neck and head start from random weights in both); add
        ``"coco"`` for the fully COCO-pretrained reference.
    val_ratio : float, default=0.2
        Validation fraction.
    test_ratio : float, default=0.0
        Held-out test fraction (used only by final reporting, never tuning).
    max_train_images : int or None, default=200
        Training images used for fine-tuning, drawn after the split so the
        validation set stays fixed; ``None`` uses all of them.
    split_by : {"image", "folder"}, default="image"
        ``"image"`` splits images at random; ``"folder"`` keeps each top-level
        sub-folder (sequence) wholly in train or validation.
    image_size : int, default=640
        Training and validation image size (multiple of 32). Use the same
        value as the pretraining ``image_size``.
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
    freeze_backbone_epochs : int, default=0
        Freeze the backbone for this many initial epochs, then unfreeze it.
    backbone_lr_mult : float, default=1.0
        Multiplier applied to the backbone learning rate.
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
    variants: Tuple[str, ...] = ("lejepa", "coco_backbone")
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    max_train_images: Optional[int] = 200
    split_by: str = "image"
    image_size: int = 640
    batch_size: int = 16
    epochs: int = 40
    optimizer: str = "AdamW"
    lr0: float = 0.002
    lrf: float = 0.01
    weight_decay: float = 1e-4
    freeze: Optional[int] = None
    freeze_backbone_epochs: int = 0
    backbone_lr_mult: float = 1.0
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
        validate_image_size(self.image_size)
        if not 0 <= self.freeze_backbone_epochs <= self.epochs or self.backbone_lr_mult <= 0:
            raise ValueError("freeze_backbone_epochs must be in [0, epochs] and backbone_lr_mult must be positive.")

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

    @property
    def eval_dir(self) -> Path:
        """pathlib.Path : Folder of per-model predicted-vs-truth box files."""
        return self.output_dir / "eval"


def detection_init_weights(variant: str, sources: WeightSources, export_dir: Path,
                           name: Optional[str] = None) -> str:
    """Return the weights ``YOLO(...)`` should start from for a variant.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.
    sources : WeightSources
        Weight locations.
    export_dir : pathlib.Path
        Folder where backbone-only checkpoints are written.
    name : str or None, default=None
        File-name stem of the exported checkpoint; defaults to ``variant``.
        Distinct names let several checkpoints of one variant coexist.

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
    pretrain_cfg = read_pretrain_config(sources.lejepa_checkpoint)
    transfer = pretrain_cfg.get("transfer", "backbone")
    init_path = export_backbone_to_yolo(
        backbone, Path(export_dir) / f"{name or variant}_init.pt", sources.model_cfg, transfer=transfer
    )
    transferred_layers = 10 if transfer == "backbone" else len(backbone.layers)
    max_diff = verify_backbone_transfer(init_path, backbone, transferred_layers)
    if max_diff != 0.0:
        raise RuntimeError(f"Backbone transfer for '{variant}' failed: max |difference| = {max_diff:.3e}")
    print(f"[{name or variant}] layers 0-{transferred_layers - 1} verified in {init_path}")
    return str(init_path)


def verify_backbone_transfer(init_weights: Path, backbone: YOLOv8MultiScaleBackbone,
                             num_layers: Optional[int] = None) -> float:
    """Measure how exactly a detector checkpoint reproduces a backbone.

    Every tensor of ``backbone`` (weights, biases and BatchNorm statistics)
    is compared with the matching tensor of the detector loaded from
    ``init_weights``.

    Parameters
    ----------
    init_weights : pathlib.Path
        Ultralytics checkpoint written by
        :func:`core_pretraining.export_backbone_to_yolo`.
    backbone : YOLOv8MultiScaleBackbone
        Backbone that should have been transferred.

    Returns
    -------
    float
        Largest absolute element-wise difference; ``0.0`` means an exact copy.

    Raises
    ------
    KeyError
        If the two state dictionaries do not have the same keys.
    """
    num_layers = num_layers or len(backbone.layers)
    source = nn.ModuleList(list(backbone.layers[:num_layers])).state_dict()
    detector = YOLO(str(init_weights)).model
    target = nn.ModuleList(list(detector.model[:num_layers])).state_dict()
    if source.keys() != target.keys():
        raise KeyError("Backbone and detector layer keys differ.")
    return max(float((source[k].float().cpu() - target[k].float().cpu()).abs().max()) for k in source)


DETECTOR_PARTS = ("backbone", "neck", "head")
EXPECTED_COCO_SHARE = {
    "lejepa": {"backbone": 0.0, "neck": 0.0, "head": 0.0},
    "scratch": {"backbone": 0.0, "neck": 0.0, "head": 0.0},
    "coco_backbone": {"backbone": 1.0, "neck": 0.0, "head": 0.0},
}


def coco_state_dict(sources: WeightSources) -> Dict[str, torch.Tensor]:
    """Load the COCO ``DetectionModel`` weights used as the audit reference.

    Parameters
    ----------
    sources : WeightSources
        Weight locations; ``sources.coco_weights`` is loaded.

    Returns
    -------
    dict of str to torch.Tensor
        CPU copy of the COCO detector state dictionary.
    """
    try:
        model = YOLO(str(sources.coco_weights))
    except ConnectionError as exc:
        raise ConnectionError(
            f"Could not load COCO weights {sources.coco_weights!r}; Ultralytics could not download them. "
            "Download yolov8n.pt manually and pass its local path as "
            "WeightSources(..., coco_weights='/path/to/yolov8n.pt') to run without network access."
        ) from exc
    return {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}


def detector_part(layer_index: int, num_layers: int, backbone_layers: int = 10) -> str:
    """Name the detector part that a layer index belongs to.

    Parameters
    ----------
    layer_index : int
        Index of the layer in ``DetectionModel.model``.
    num_layers : int
        Total number of layers; the last one is the Detect head.
    backbone_layers : int, default=10
        Number of leading backbone layers.

    Returns
    -------
    str
        ``"backbone"``, ``"neck"`` or ``"head"``.
    """
    if layer_index < backbone_layers:
        return "backbone"
    return "head" if layer_index == num_layers - 1 else "neck"


def coco_share_by_part(model: nn.Module, coco_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Fraction of trainable tensors in each detector part identical to COCO.

    Only trainable parameters with the same name and shape in both models are
    compared. Fixed buffers and the non-trainable DFL projection (identical in
    every YOLOv8 model by construction) are ignored, as are head tensors whose
    shape changed because the number of classes differs.

    Parameters
    ----------
    model : torch.nn.Module
        Ultralytics ``DetectionModel`` to inspect.
    coco_state : dict of str to torch.Tensor
        State dictionary of the COCO ``DetectionModel``.

    Returns
    -------
    dict of str to float
        Share of compared tensors equal to COCO, per part (``nan`` if none).
    """
    num_layers = len(model.model)
    equal = {part: 0 for part in DETECTOR_PARTS}
    total = {part: 0 for part in DETECTOR_PARTS}
    for name, param in model.named_parameters():
        reference = coco_state.get(name)
        if ".dfl." in name or reference is None or reference.shape != param.shape:
            continue
        part = detector_part(int(name.split(".")[1]), num_layers)
        total[part] += 1
        equal[part] += int(torch.equal(param.detach().float().cpu(), reference.float().cpu()))
    return {part: equal[part] / total[part] if total[part] else float("nan") for part in DETECTOR_PARTS}


def check_initialisation(variant: str, shares: Dict[str, float]) -> None:
    """Fail if a variant does not start from the weights it claims to.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.
    shares : dict of str to float
        Output of :func:`coco_share_by_part`.

    Raises
    ------
    RuntimeError
        If, for example, the neck or head of ``"lejepa"`` or
        ``"coco_backbone"`` contains COCO weights.
    """
    expected = EXPECTED_COCO_SHARE.get(variant)
    if expected is None:
        return
    wrong = {part: shares[part] for part, value in expected.items() if shares[part] != value}
    if wrong:
        raise RuntimeError(f"Variant '{variant}' starts from unexpected weights: share identical to "
                           f"COCO {wrong}, expected {expected}.")


def record_initialisation(trainer, *, variant: str, coco_state: Dict[str, torch.Tensor],
                          report: Dict[str, float]) -> None:
    """Ultralytics ``on_train_start`` callback that audits the model being trained.

    It inspects ``trainer.model`` - the network Ultralytics actually
    optimises - so any weight loading done inside Ultralytics is caught.

    Parameters
    ----------
    trainer : ultralytics.engine.trainer.BaseTrainer
        Trainer passed by Ultralytics.
    variant : str
        Variant being trained.
    coco_state : dict of str to torch.Tensor
        State dictionary of the COCO ``DetectionModel``.
    report : dict
        Filled in place with ``coco_share_<part>`` entries.

    Raises
    ------
    RuntimeError
        Propagated from :func:`check_initialisation`, stopping training.
    """
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    shares = coco_share_by_part(model, coco_state)
    report.update({f"coco_share_{part}": value for part, value in shares.items()})
    print(f"[{variant}] share of tensors identical to COCO: "
          + ", ".join(f"{part}={value:.2f}" for part, value in shares.items()))
    check_initialisation(variant, shares)


def set_backbone_lr_multiplier(trainer, multiplier: float) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    backbone_ids = {id(param) for layer in model.model[:10] for param in layer.parameters()}
    optimizer = trainer.optimizer
    for group in list(optimizer.param_groups):
        backbone, other = [], []
        for param in group["params"]:
            (backbone if id(param) in backbone_ids else other).append(param)
        if not backbone:
            continue
        if other:
            group["params"] = other
            backbone_group = {key: value for key, value in group.items() if key != "params"}
            backbone_group["params"] = backbone
            optimizer.add_param_group(backbone_group)
            group = optimizer.param_groups[-1]
        group["lr"] *= multiplier
        if "initial_lr" in group:
            group["initial_lr"] *= multiplier
    scheduler = getattr(trainer, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
        if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
            scheduler.lr_lambdas.extend([scheduler.lr_lambdas[-1]] *
                                        (len(optimizer.param_groups) - len(scheduler.lr_lambdas)))


def apply_backbone_freeze_warmup(trainer, freeze_epochs: int) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    frozen = int(trainer.epoch) < freeze_epochs
    for layer in model.model[:10]:
        layer.requires_grad_(not frozen)
        if frozen:
            layer.eval()


def finetune_detector(init_weights: str, data_yaml: Path, cfg: DetectionEvalConfig,
                      run_name: str, device: str,
                      coco_state: Optional[Dict[str, torch.Tensor]] = None,
                      audit_variant: Optional[str] = None,
                      epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None
                      ) -> Tuple[Path, Dict[str, float]]:
    """Fine-tune a YOLOv8 detector, auditing its starting weights.

    When ``coco_state`` is given, the model Ultralytics trains is compared
    with COCO at the start of training (:func:`record_initialisation`), and
    training stops if the neck or head of a backbone-only variant is not
    random.

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
    coco_state : dict of str to torch.Tensor or None, default=None
        COCO ``DetectionModel`` state dictionary used for the audit.
    audit_variant : str or None, default=None
        Variant whose expected initialisation is checked; defaults to
        ``run_name``.

    epoch_callback : callable or None, default=None
        Called as ``epoch_callback(epoch, metrics)`` after each epoch's
        validation (``metrics`` is the Ultralytics metrics dictionary); it may
        raise to stop training (used for pruning).

    Returns
    -------
    run_dir : pathlib.Path
        The run folder (contains ``weights/best.pt`` and ``results.csv``).
    init_report : dict of str to float
        Share of tensors identical to COCO per part (empty without audit).
    """
    if str(device) == "mps":
        run_assigner_on_cpu()
    model = YOLO(init_weights)
    init_report: Dict[str, float] = {}
    if cfg.backbone_lr_mult != 1.0:
        model.add_callback("on_train_start", partial(set_backbone_lr_multiplier,
                                                     multiplier=cfg.backbone_lr_mult))
    if cfg.freeze_backbone_epochs:
        model.add_callback("on_train_epoch_start", partial(apply_backbone_freeze_warmup,
                                                            freeze_epochs=cfg.freeze_backbone_epochs))
    if epoch_callback is not None:
        model.add_callback("on_fit_epoch_end",
                           lambda trainer: epoch_callback(int(trainer.epoch), dict(trainer.metrics)))
    if coco_state is not None:
        model.add_callback("on_train_start", partial(record_initialisation, variant=audit_variant or run_name,
                                                     coco_state=coco_state, report=init_report))
    model.train(
        data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.image_size, batch=cfg.batch_size,
        device=device, optimizer=cfg.optimizer, lr0=cfg.lr0, lrf=cfg.lrf,
        weight_decay=cfg.weight_decay, freeze=cfg.freeze, close_mosaic=cfg.close_mosaic,
        patience=cfg.patience, workers=cfg.workers, amp=cfg.amp, seed=cfg.seed,
        project=str(cfg.runs_dir), name=run_name, exist_ok=True, plots=True, val=True,
        **cfg.train_overrides,
    )
    return Path(model.trainer.save_dir), init_report


def validate_detector(weights: Path, data_yaml: Path, cfg: DetectionEvalConfig,
                      run_name: str, device: str, split: str = "val") -> Dict[str, float]:
    """Compute detection metrics on the validation (or test) split.

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
    split : {"val", "test"}, default="val"
        Split to score.

    Returns
    -------
    dict of str to float
        ``precision``, ``recall``, ``mAP50`` and ``mAP50-95``.
    """
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), imgsz=cfg.image_size, batch=cfg.batch_size, device=device,
        split=split, plots=False, project=str(cfg.runs_dir), name=f"{run_name}_{split}", exist_ok=True,
    )
    box = metrics.box
    return {"precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50-95": float(box.map),
            "mAP75": float(box.all_ap[:, 5].mean()) if getattr(box, "all_ap", None) is not None else float("nan")}


BOX_COLUMNS = ("image_id", "status", "class", "confidence", "iou",
               "gt_x1", "gt_y1", "gt_x2", "gt_y2", "pred_x1", "pred_y1", "pred_x2", "pred_y2")


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Intersection over union of every pair of boxes.

    Parameters
    ----------
    boxes_a : numpy.ndarray
        Shape ``(n, 4)`` rows of ``x1 y1 x2 y2``.
    boxes_b : numpy.ndarray
        Shape ``(m, 4)`` rows of ``x1 y1 x2 y2``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, m)`` IoU matrix.
    """
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).prod(axis=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def match_boxes(gt_boxes: np.ndarray, gt_classes: np.ndarray, pred_boxes: np.ndarray,
                pred_classes: np.ndarray, scores: np.ndarray,
                iou_threshold: float = 0.5) -> List[dict]:
    """Greedily match predictions to ground truth, highest confidence first.

    A prediction matches the unmatched same-class ground-truth box of highest
    IoU if that IoU reaches ``iou_threshold``.

    Parameters
    ----------
    gt_boxes : numpy.ndarray
        Shape ``(g, 4)`` pixel ``x1 y1 x2 y2``.
    gt_classes : numpy.ndarray
        Shape ``(g,)`` class ids.
    pred_boxes : numpy.ndarray
        Shape ``(p, 4)`` pixel ``x1 y1 x2 y2``.
    pred_classes : numpy.ndarray
        Shape ``(p,)`` class ids.
    scores : numpy.ndarray
        Shape ``(p,)`` confidences.
    iou_threshold : float, default=0.5
        Minimum IoU of a match.

    Returns
    -------
    list of dict
        One record per matched pair (``TP``), unmatched prediction (``FP``)
        or unmatched ground truth (``FN``); keys are :data:`BOX_COLUMNS`
        without ``image_id``.
    """
    ious = pairwise_iou(gt_boxes, pred_boxes) if len(gt_boxes) and len(pred_boxes) else np.zeros((len(gt_boxes), len(pred_boxes)))
    taken, records = set(), []
    for p in np.argsort(-scores):
        candidates = [(ious[g, p], g) for g in range(len(gt_boxes))
                      if g not in taken and gt_classes[g] == pred_classes[p]]
        best_iou, best_gt = max(candidates, default=(0.0, None))
        pred = dict(zip(("pred_x1", "pred_y1", "pred_x2", "pred_y2"), map(float, pred_boxes[p])))
        if best_gt is not None and best_iou >= iou_threshold:
            taken.add(best_gt)
            gt = dict(zip(("gt_x1", "gt_y1", "gt_x2", "gt_y2"), map(float, gt_boxes[best_gt])))
            records.append({"status": "TP", "class": int(pred_classes[p]), "confidence": float(scores[p]),
                            "iou": float(best_iou), **gt, **pred})
        else:
            records.append({"status": "FP", "class": int(pred_classes[p]), "confidence": float(scores[p]),
                            "iou": np.nan, **pred})
    for g in range(len(gt_boxes)):
        if g not in taken:
            records.append({"status": "FN", "class": int(gt_classes[g]), "confidence": np.nan, "iou": np.nan,
                            **dict(zip(("gt_x1", "gt_y1", "gt_x2", "gt_y2"), map(float, gt_boxes[g])))})
    return records


def save_box_predictions(weights: Path, image_dir: Path, label_dir: Path, out_path: Path, image_size: int,
                         conf: float, device: str, iou_threshold: float = 0.5) -> Path:
    """Save predicted and ground-truth boxes of every image of a split.

    Images with neither boxes nor predictions get one row with status
    ``"EMPTY"`` so every image id appears in the file.

    Parameters
    ----------
    weights : pathlib.Path
        Trained detector (``best.pt``).
    image_dir : pathlib.Path
        Images of the evaluation split.
    label_dir : pathlib.Path
        YOLO labels of the split.
    out_path : pathlib.Path
        Destination ``.csv`` with :data:`BOX_COLUMNS`; boxes are pixel
        ``x1 y1 x2 y2``, ``image_id`` is the image file name.
    image_size : int
        Inference size.
    conf : float
        Confidence threshold of the saved predictions.
    device : str
        Ultralytics device string.
    iou_threshold : float, default=0.5
        Minimum IoU for a TP match.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    model = YOLO(str(weights))
    rows = []
    for image_path in list_images(image_dir):
        width, height = load_rgb(image_path).size
        labels = read_yolo_labels(Path(label_dir) / f"{image_path.stem}.txt")
        result = model.predict(str(image_path), imgsz=image_size, conf=conf, device=device, verbose=False)[0]
        boxes = result.boxes
        records = match_boxes(yolo_to_xyxy(labels, width, height), labels[:, 0].astype(int),
                              boxes.xyxy.cpu().numpy().reshape(-1, 4), boxes.cls.cpu().numpy().astype(int),
                              boxes.conf.cpu().numpy(), iou_threshold)
        rows.extend({"image_id": image_path.name, **record}
                    for record in records or [{"status": "EMPTY"}])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=list(BOX_COLUMNS)).to_csv(out_path, index=False)
    return out_path


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
ANNOTATION_NAME_KEYS = ("file_name", "filename", "image", "image_name", "name", "path")
ANNOTATION_LABEL_KEYS = ("label", "status", "class", "category", "plot_status")


def normalize_label(text: str) -> str:
    """Canonical form of a class name for tolerant matching.

    Parameters
    ----------
    text : str
        Raw label such as ``"In Plot"`` or ``"between-plots"``.

    Returns
    -------
    str
        Lower-case label with spaces and hyphens replaced by underscores.
    """
    return str(text).strip().lower().replace(" ", "_").replace("-", "_")


def find_annotation_file(image_dir: Path) -> Path:
    """Find the JSON annotation file inside a classification dataset folder.

    Parameters
    ----------
    image_dir : pathlib.Path
        Dataset folder.

    Returns
    -------
    pathlib.Path
        ``annotations.json`` if present, otherwise the only ``.json`` file.

    Raises
    ------
    FileNotFoundError
        If no JSON file, or several without an ``annotations.json``, exist.
    """
    candidates = sorted(Path(image_dir).rglob("*.json"))
    preferred = [p for p in candidates if p.name.lower() == "annotations.json"]
    if preferred:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected one JSON annotation file in {image_dir}, found {len(candidates)}: "
                            f"{[p.name for p in candidates]}. Set annotation_path explicitly.")


def first_present(record: dict, keys: Sequence[str]) -> Optional[str]:
    """Return the value of the first key of ``keys`` present in ``record``.

    Parameters
    ----------
    record : dict
        Annotation record.
    keys : sequence of str
        Candidate keys in priority order.

    Returns
    -------
    str or None
        The value, or ``None`` if no key is present.
    """
    return next((record[k] for k in keys if k in record), None)


def load_label_annotations(annotation_path: Path) -> Dict[str, str]:
    """Read image-level labels from a JSON file.

    Supported shapes: ``{"img.jpg": "in_plot"}``, ``{"img.jpg": {"label":
    "in_plot"}}`` and ``[{"file_name": "img.jpg", "label": "in_plot"}, ...]``
    (see :data:`ANNOTATION_NAME_KEYS` and :data:`ANNOTATION_LABEL_KEYS`).

    Parameters
    ----------
    annotation_path : pathlib.Path
        JSON annotation file.

    Returns
    -------
    dict of str to str
        Image file name (without folders) to raw label.

    Raises
    ------
    ValueError
        If the JSON structure is not recognised.
    """
    data = json.loads(Path(annotation_path).read_text())
    if isinstance(data, dict):
        records = [(name, value if not isinstance(value, dict) else first_present(value, ANNOTATION_LABEL_KEYS))
                   for name, value in data.items()]
    elif isinstance(data, list) and all(isinstance(r, dict) for r in data):
        records = [(first_present(r, ANNOTATION_NAME_KEYS), first_present(r, ANNOTATION_LABEL_KEYS)) for r in data]
    else:
        raise ValueError(f"Unrecognised annotation structure in {annotation_path}")
    return {Path(str(name)).name: str(label) for name, label in records if name is not None and label is not None}


class PlotStatusDataset(Dataset):
    """Images with one class label each, read from a JSON annotation file.

    Parameters
    ----------
    image_dir : pathlib.Path
        Folder searched recursively for images.
    annotation_path : pathlib.Path or None
        JSON annotation file; ``None`` uses :func:`find_annotation_file`.
    label_mapping : dict of str to int
        Class name to class index (matched with :func:`normalize_label`).
    transform : callable
        Transform applied to each PIL image.
    """

    def __init__(self, image_dir: Path, annotation_path: Optional[Path], label_mapping: Dict[str, int],
                 transform) -> None:
        annotation_path = Path(annotation_path) if annotation_path else find_annotation_file(image_dir)
        annotations = {name: normalize_label(label) for name, label in load_label_annotations(annotation_path).items()}
        mapping = {normalize_label(name): index for name, index in label_mapping.items()}
        images = list_images(image_dir)
        self.transform = transform
        self.samples = [(path, mapping[annotations[path.name]]) for path in images
                        if annotations.get(path.name) in mapping]
        if not self.samples:
            raise RuntimeError(
                f"No annotated images found: {len(images)} images in {image_dir}, {len(annotations)} "
                f"annotations in {annotation_path.name}, labels found {sorted(set(annotations.values()))}, "
                f"expected {sorted(mapping)}.")
        print(f"Plot status data: {len(self.samples)} labelled images of {len(images)} "
              f"(annotations: {annotation_path.name}), class counts {dict(Counter(self.targets))}.")

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
        Folder with plot-status images (searched recursively).
    output_dir : str or pathlib.Path
        Folder for histories, metrics and plots.
    weights : WeightSources
        Initial-weight locations.
    annotation_path : str, pathlib.Path or None, default=None
        JSON annotation file; ``None`` finds it inside ``image_dir``.
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
    max_train_images : int or None, default=200
        Training images for the probe, drawn with stratification after the
        split so the test set stays fixed; ``None`` uses all of them.
    num_workers : int, default=0
        DataLoader workers.
    seed : int, default=48
        Seed for splitting and training.
    device : str or None, default=None
        Torch device string; ``None`` selects automatically.
    """

    image_dir: Path
    output_dir: Path
    weights: WeightSources
    annotation_path: Optional[Path] = None
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
    max_train_images: Optional[int] = 200
    num_workers: int = 0
    seed: int = 48
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.image_dir = Path(self.image_dir)
        self.annotation_path = Path(self.annotation_path) if self.annotation_path else None
        self.output_dir = Path(self.output_dir)
        self.variants = tuple(validate_variant(v) for v in self.variants)

    @property
    def class_names(self) -> List[str]:
        """list of str : Class names ordered by index."""
        return sorted(self.label_mapping, key=self.label_mapping.get)


def build_probe_loaders(cfg: ProbeEvalConfig) -> Tuple[DataLoader, DataLoader]:
    """Load the plot-status dataset, split it, and cap the training set.

    The test split is fixed by ``cfg.test_ratio`` and ``cfg.seed``; the
    training split is then reduced to ``cfg.max_train_images`` with the class
    proportions preserved.

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
    targets = np.asarray(dataset.targets)
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=cfg.test_ratio,
                                           stratify=targets, random_state=cfg.seed)
    available = len(train_idx)
    if cfg.max_train_images is not None and available > cfg.max_train_images:
        train_idx, _ = train_test_split(train_idx, train_size=cfg.max_train_images,
                                        stratify=targets[train_idx], random_state=cfg.seed)
    print(f"Probe split: {len(train_idx)} train (of {available} available) | {len(test_idx)} test")
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
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images, targets = batch[:2]
        if images.dim() == 5:
            images, targets = images.flatten(0, 1), targets.flatten()
        features.append(model.embed(images.to(device)).cpu().numpy())
        labels.append(np.asarray(targets))
    model.train(was_training)
    return np.concatenate(features), np.concatenate(labels)