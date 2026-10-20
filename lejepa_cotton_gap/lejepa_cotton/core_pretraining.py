"""LeJEPA pretraining of YOLOv8 backbone and neck features.

Two prediction-loss variants are available (``pred_loss_mode``): the original
``"cross_camera"`` loss and the ``"intra_image"`` loss that compares two mild
geometric views of the same image.

This module owns everything needed to *learn* a representation:

* discovering synchronised multi-camera frames on disk,
* the multi-camera view dataset and its augmentations,
* the randomly initialised (``yolov8n.yaml``) multi-scale backbone,
* the global-average-pooling (GAP) encoder and projectors,
* the SIGReg regulariser and both prediction losses,
* the lambda sweep helpers (checkpoint naming),
* the training loop, checkpoint I/O and export of the backbone into an
  Ultralytics detection checkpoint.

"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from ultralytics import YOLO

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_FILENAME_PATTERN = r"clip(?P<clip>\d+)_cam(?P<cam>\d+)_frame(?P<frame>\d+)"
EMPTY_MODEL_CFG = "yolov8n.yaml"
PRED_LOSS_MODES = ("cross_camera", "intra_image")
POOLING_MODES = ("gem", "gap")
TRANSFER_MODES = ("backbone", "backbone+neck")
PREDICTION_PAIRINGS = ("intra_image", "cross_camera")
DEFAULT_LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_IMAGE_SIZE = 640

CameraGroup = Dict[int, Path]
EpochCallback = Callable[[int, nn.Module, DataLoader], None]


@dataclass
class GeMConfig:
    p_init: float = 3.0
    p_min: float = 1.0
    p_max: float = 6.0
    learnable_p: bool = True

    def __post_init__(self) -> None:
        if self.p_min <= 0 or not self.p_min <= self.p_init <= self.p_max:
            raise ValueError("GeM requires 0 < p_min <= p_init <= p_max.")


@dataclass
class RotationConfig:
    enabled: bool = True
    max_deg: float = 15.0
    p: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.max_deg <= 15 or not 0.0 <= self.p <= 1.0:
            raise ValueError("rotation.max_deg must be in [0, 15] and rotation.p must lie in [0, 1].")


# --------------------------------------------------------------------------- #
# Configuration and device
# --------------------------------------------------------------------------- #
@dataclass
class PretrainConfig:
    """Hyper-parameters and paths for LeJEPA pretraining.

    Parameters
    ----------
    image_root : str or pathlib.Path
        Folder searched recursively for multi-camera frames named like
        ``clip<n>_cam<k>_frame<m>.jpg``.
    output_dir : str or pathlib.Path
        Folder that receives the checkpoint, loss history and plots.
    cameras : tuple of int, default=(1, 2, 4)
        Zero-indexed camera ids that form the views of one sample. At least
        two cameras are required for the cross-camera prediction loss.
    max_samples : int or None, default=None
        Maximum number of synchronised frame groups. ``None`` uses all.
    views_per_camera : int, default=1
        Independent augmentations drawn from each camera image.
    image_size : int, default=640
        Side length of the square images fed to the backbone. Must be a
        multiple of 32 and should equal the detection fine-tuning size.
    batch_size : int, default=16
        Number of frame groups per optimisation step.
    epochs : int, default=100
        Number of passes over the dataset.
    proj_dim : int, default=128
        Output width of each projector.
    hidden_dim : int, default=256
        Hidden width of each projector MLP.
    lr : float, default=1e-3
        AdamW learning rate.
    weight_decay : float, default=1e-4
        AdamW weight decay.
    lam : float, default=0.2
        Weight of SIGReg; the prediction loss is weighted by ``1 - lam``.
        Used by a single run; a sweep overrides it with each value of
        ``lambdas``.
    lambdas : tuple of float, default=(0.1, 0.2, 0.3, 0.4, 0.5)
        SIGReg weights pretrained one after another by the lambda sweep.
    pred_loss_mode : {"cross_camera", "intra_image"} or None
        Compatibility alias for ``prediction_pairing``.
    pooling : {"gem", "gap"}, default="gem"
        Spatial pooling applied independently to each selected scale.
    pool_cameras : tuple of int, default=(1, 2, 5, 6)
        Camera ids (as written in the file names) whose images form the pool
        of the ``"intra_image"`` mode.
    overwrite : bool, default=False
        Retrain a lambda even when its checkpoint already exists.
    checkpoint_dir_name : str, default="checkpoints"
        Sub-folder of ``output_dir`` receiving the per-lambda artefacts.
    sigreg_knots : int, default=17
        Integration knots of the SIGReg characteristic-function test.
    sigreg_slices : int, default=256
        Random 1-D projections used by SIGReg per step.
    model_cfg : str, default="yolov8n.yaml"
        Ultralytics model definition. A ``.yaml`` file gives empty
        (randomly initialised) weights.
    filename_pattern : str, default=DEFAULT_FILENAME_PATTERN
        Regular expression with named groups ``clip``, ``cam`` and ``frame``.
    num_workers : int, default=0
        DataLoader worker processes (0 is safest on macOS).
    seed : int, default=42
        Seed for sampling, shuffling and initialisation.
    device : str or None, default=None
        Torch device string. ``None`` picks MPS, then CUDA, then CPU.
    log_every_n_steps : int, default=0
        Optional interval for learned-GeM logging; zero disables step logging.
    debug_every_n_steps : int, default=0
        Optional interval for activation debug grids; zero disables them.
    checkpoint_name : str, default="gap_lejepa_yolov8n.pth"
        File name of the checkpoint written inside ``output_dir``.
    """

    image_root: Path
    output_dir: Path
    cameras: Tuple[int, ...] = (1, 2, 4)
    max_samples: Optional[int] = None
    views_per_camera: int = 1
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 16
    epochs: int = 100
    proj_dim: int = 128
    hidden_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lam: float = 0.2
    lambdas: Tuple[float, ...] = DEFAULT_LAMBDAS
    pred_loss_mode: Optional[str] = None
    prediction_pairing: str = "intra_image"
    pool_cameras: Tuple[int, ...] = (1, 2, 5, 6)
    pooling: str = "gem"
    gem: GeMConfig = field(default_factory=GeMConfig)
    pretrain_neck: bool = True
    scale_weights: Dict[str, float] = field(default_factory=lambda: {"p3": 1.0, "p4": 0.0, "p5": 0.0})
    transfer: str = "backbone"
    rotation: RotationConfig = field(default_factory=RotationConfig)
    global_scale: Tuple[float, float] = (0.4, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.25)
    log_every_n_steps: int = 0
    debug_every_n_steps: int = 0
    overwrite: bool = False
    checkpoint_dir_name: str = "checkpoints"
    sigreg_knots: int = 17
    sigreg_slices: int = 256
    model_cfg: str = EMPTY_MODEL_CFG
    filename_pattern: str = DEFAULT_FILENAME_PATTERN
    num_workers: int = 0
    seed: int = 42
    device: Optional[str] = None
    checkpoint_name: str = "gap_lejepa_yolov8n.pth"

    def __post_init__(self) -> None:
        if isinstance(self.gem, dict):
            self.gem = GeMConfig(**self.gem)
        if isinstance(self.rotation, dict):
            self.rotation = RotationConfig(**self.rotation)
        self.image_root = Path(self.image_root)
        self.output_dir = Path(self.output_dir)
        self.cameras = tuple(int(c) for c in self.cameras)
        self.pool_cameras = tuple(int(c) for c in self.pool_cameras)
        self.lambdas = tuple(float(v) for v in self.lambdas)
        if self.pred_loss_mode is not None and self.pred_loss_mode not in PRED_LOSS_MODES:
            raise ValueError(f"pred_loss_mode must be one of {PRED_LOSS_MODES}, got '{self.pred_loss_mode}'.")
        if self.prediction_pairing not in PREDICTION_PAIRINGS:
            raise ValueError(f"prediction_pairing must be one of {PREDICTION_PAIRINGS}.")
        if self.pred_loss_mode is not None:
            self.prediction_pairing = self.pred_loss_mode
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got '{self.pooling}'.")
        if self.transfer not in TRANSFER_MODES:
            raise ValueError(f"transfer must be one of {TRANSFER_MODES}, got '{self.transfer}'.")
        if self.transfer == "backbone+neck" and not self.pretrain_neck:
            raise ValueError("transfer='backbone+neck' requires pretrain_neck=True.")
        self.scale_weights = {str(k).lower(): float(v) for k, v in self.scale_weights.items()}
        if set(self.scale_weights) - {"p3", "p4", "p5"} or any(v < 0 for v in self.scale_weights.values()):
            raise ValueError("scale_weights must contain non-negative p3, p4 and/or p5 weights.")
        if self.pooling == "gap" and not self.pretrain_neck and not self.rotation.enabled:
            if self.pred_loss_mode is None and self.prediction_pairing == "intra_image":
                self.prediction_pairing = "cross_camera"
            if self.scale_weights == {"p3": 1.0, "p4": 0.0, "p5": 0.0}:
                self.scale_weights = {"p3": 1 / 3, "p4": 1 / 3, "p5": 1 / 3}
        if not any(self.scale_weights.get(k, 0.0) > 0 for k in ("p3", "p4", "p5")):
            raise ValueError("At least one scale weight must be positive.")
        if self.prediction_pairing == "cross_camera" and len(self.cameras) < 2:
            raise ValueError("Cross-camera prediction needs at least two cameras.")
        if self.prediction_pairing == "intra_image" and not self.pool_cameras:
            raise ValueError("Intra-image prediction needs at least one pool camera.")
        self.global_scale = tuple(float(v) for v in self.global_scale)
        self.local_scale = tuple(float(v) for v in self.local_scale)
        for name, scale in (("global_scale", self.global_scale), ("local_scale", self.local_scale)):
            if len(scale) != 2 or not 0 < scale[0] <= scale[1] <= 1:
                raise ValueError(f"{name} must be a pair satisfying 0 < min <= max <= 1.")
        if self.log_every_n_steps < 0 or self.debug_every_n_steps < 0:
            raise ValueError("Logging and debug intervals must be non-negative.")
        validate_image_size(self.image_size)
        if not all(0.0 <= v <= 1.0 for v in (self.lam, *self.lambdas)):
            raise ValueError("lam and every value of lambdas must lie in [0, 1].")

    @property
    def checkpoint_path(self) -> Path:
        """pathlib.Path : Location of the pretrained encoder checkpoint."""
        return self.output_dir / self.checkpoint_name

    @property
    def imgsz(self) -> int:
        """int : Alias of ``image_size`` matching the Ultralytics argument name."""
        return self.image_size

    def to_dict(self) -> dict:
        """Serialise the configuration with paths converted to strings.

        Returns
        -------
        dict
            Plain-Python representation safe for ``torch.save``.
        """
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def validate_image_size(image_size: int, stride: int = 32) -> int:
    """Check that an input size suits the YOLOv8 stride.

    Parameters
    ----------
    image_size : int
        Candidate side length in pixels.
    stride : int, default=32
        Largest stride of the network.

    Returns
    -------
    int
        The same size.

    Raises
    ------
    ValueError
        If ``image_size`` is not a positive multiple of ``stride``.
    """
    if image_size <= 0 or image_size % stride:
        raise ValueError(f"image size must be a positive multiple of {stride}, got {image_size}.")
    return image_size


def lambda_tag(lam: float) -> str:
    """Format a lambda for file and model names.

    Parameters
    ----------
    lam : float
        SIGReg weight.

    Returns
    -------
    str
        ``"0.1"`` for ``0.1`` (shortest exact form).
    """
    return f"{lam:g}"


def lambda_artifact_paths(cfg: "PretrainConfig", lam: float) -> Dict[str, Path]:
    """Paths of every artefact of one lambda run (single naming rule).

    Parameters
    ----------
    cfg : PretrainConfig
        Configuration providing ``output_dir`` and ``checkpoint_dir_name``.
    lam : float
        SIGReg weight of the run.

    Returns
    -------
    dict of str to pathlib.Path
        ``checkpoint`` (``backbone_lambda_<lam>.pt``), ``config``
        (``..._config.json``) and ``history`` (``..._history.csv``), all in
        ``output_dir / checkpoint_dir_name``.
    """
    folder = cfg.output_dir / cfg.checkpoint_dir_name
    stem = f"backbone_lambda_{lambda_tag(lam)}"
    return {"checkpoint": folder / f"{stem}.pt", "config": folder / f"{stem}_config.json",
            "history": folder / f"{stem}_history.csv"}


def config_for_lambda(cfg: "PretrainConfig", lam: float) -> "PretrainConfig":
    """Copy a configuration with a different SIGReg weight.

    Parameters
    ----------
    cfg : PretrainConfig
        Base configuration.
    lam : float
        SIGReg weight of the copy.

    Returns
    -------
    PretrainConfig
        New configuration; ``cfg`` is unchanged.
    """
    return replace(cfg, lam=lam)


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Pick the compute device.

    Parameters
    ----------
    preferred : str or None, default=None
        Explicit device string such as ``"cpu"``, ``"mps"`` or ``"cuda:0"``.

    Returns
    -------
    torch.device
        ``preferred`` when given, otherwise MPS, then CUDA, then CPU.
    """
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Files and sampling
# --------------------------------------------------------------------------- #
def list_images(root: Path, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[Path]:
    """Recursively list image files under a folder.

    Parameters
    ----------
    root : pathlib.Path
        Folder to search.
    extensions : sequence of str, default=IMAGE_EXTENSIONS
        Lower-case suffixes accepted as images.

    Returns
    -------
    list of pathlib.Path
        Sorted image paths.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def load_rgb(path: Path) -> Image.Image:
    """Open an image file as an RGB PIL image.

    Parameters
    ----------
    path : pathlib.Path
        Image file.

    Returns
    -------
    PIL.Image.Image
        Image converted to RGB.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def subsample(items: Sequence, max_items: Optional[int], seed: int) -> list:
    """Randomly keep at most ``max_items`` elements, preserving order.

    Parameters
    ----------
    items : sequence
        Elements to sample from.
    max_items : int or None
        Upper bound. ``None`` keeps everything.
    seed : int
        Seed of the NumPy generator.

    Returns
    -------
    list
        Selected elements in their original order.
    """
    if max_items is None or len(items) <= max_items:
        return list(items)
    keep = np.sort(np.random.default_rng(seed).choice(len(items), size=max_items, replace=False))
    return [items[i] for i in keep]


def discover_camera_groups(
    root: Path,
    cameras: Sequence[int],
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
) -> List[CameraGroup]:
    """Group images of the same clip and frame captured by different cameras.

    Parameters
    ----------
    root : pathlib.Path
        Folder searched recursively for images.
    cameras : sequence of int
        Camera ids that must all be present for a group to be kept.
    filename_pattern : str, default=DEFAULT_FILENAME_PATTERN
        Regex with named groups ``clip``, ``cam`` and ``frame``.

    Returns
    -------
    list of dict
        One ``{camera_id: path}`` mapping per synchronised frame, sorted by
        clip then frame.

    Raises
    ------
    RuntimeError
        If no frame is seen by all requested cameras.
    """
    regex = re.compile(filename_pattern, re.IGNORECASE)
    wanted = set(cameras)
    groups: Dict[Tuple[int, int], CameraGroup] = defaultdict(dict)
    for path in list_images(root):
        match = regex.search(path.name)
        if match and int(match["cam"]) in wanted:
            groups[(int(match["clip"]), int(match["frame"]))][int(match["cam"])] = path
    complete = [groups[key] for key in sorted(groups) if len(groups[key]) == len(wanted)]
    if not complete:
        raise RuntimeError(f"No frame in {root} is captured by all cameras {sorted(wanted)}.")
    return complete


# --------------------------------------------------------------------------- #
# Augmentation and dataset
# --------------------------------------------------------------------------- #
class RandomRotationReflect:
    """Sample an independent, reflect-padded rotation for each transformed view."""

    def __init__(self, max_deg: float = 15.0, p: float = 0.5, enabled: bool = True) -> None:
        if not 0 <= max_deg <= 15 or not 0 <= p <= 1:
            raise ValueError("Rotation must use max_deg in [0, 15] and p in [0, 1].")
        self.max_deg = float(max_deg)
        self.p = float(p)
        self.enabled = bool(enabled)
        self.last_angle = 0.0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.last_angle = 0.0
        if not self.enabled or torch.rand(()) >= self.p:
            return image
        self.last_angle = float(torch.empty(()).uniform_(-self.max_deg, self.max_deg))
        radians = np.deg2rad(self.last_angle)
        theta = image.new_tensor([[
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
        ]])
        grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False).squeeze(0)


def build_train_transform(image_size: int, rotation: Optional[RotationConfig] = None) -> v2.Compose:
    """Resize and augment every camera view independently.

    Rotations use reflect padding so image corners do not reveal augmentation
    labels. Color changes are not applied during pretraining.

    Parameters
    ----------
    image_size : int
        Output side length.

    Returns
    -------
    torchvision.transforms.v2.Compose
        PIL image to normalised ``float32`` tensor pipeline.
    """
    if isinstance(rotation, dict):
        rotation = RotationConfig(**rotation)
    rotation = rotation or RotationConfig(enabled=False)
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        RandomRotationReflect(rotation.max_deg, rotation.p, rotation.enabled),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int) -> v2.Compose:
    """Deterministic resize and normalisation used for evaluation.

    Parameters
    ----------
    image_size : int
        Output side length.

    Returns
    -------
    torchvision.transforms.v2.Compose
        PIL image to normalised ``float32`` tensor pipeline.
    """
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiCameraViewDataset(Dataset):
    """Dataset whose views of one sample come from different cameras.

    Parameters
    ----------
    groups : list of dict
        Synchronised ``{camera_id: path}`` mappings from
        :func:`discover_camera_groups`.
    cameras : sequence of int
        Camera order used to stack views (camera-major).
    transform : callable
        Augmentation applied to every view.
    views_per_camera : int, default=1
        Augmentations drawn from each camera image.
    """

    def __init__(self, groups: List[CameraGroup], cameras: Sequence[int], transform: Callable,
                 views_per_camera: int = 1) -> None:
        self.groups = groups
        self.cameras = tuple(cameras)
        self.transform = transform
        self.views_per_camera = views_per_camera

    def __len__(self) -> int:
        """Return the number of synchronised frame groups.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and augment all camera views of one frame.

        Parameters
        ----------
        idx : int
            Group index.

        Returns
        -------
        views : torch.Tensor
            Shape ``(V, 3, H, W)`` ordered camera-major.
        camera_ids : torch.Tensor
            Shape ``(V,)`` camera id of each view.
        image_ids : torch.Tensor
            Shape ``(V,)`` source-image group id for each view.
        """
        views, camera_ids = [], []
        for cam in self.cameras:
            image = load_rgb(self.groups[idx][cam])
            for _ in range(self.views_per_camera):
                views.append(self.transform(image))
                camera_ids.append(cam)
        return torch.stack(views), torch.tensor(camera_ids), torch.full((len(views),), idx)


def discover_image_pool(root: Path, cameras: Sequence[int],
                        filename_pattern: str = DEFAULT_FILENAME_PATTERN) -> List[Tuple[Path, int]]:
    """List every image captured by one of the given cameras.

    Parameters
    ----------
    root : pathlib.Path
        Folder searched recursively.
    cameras : sequence of int
        Camera ids kept (as written in the file names).
    filename_pattern : str, default=DEFAULT_FILENAME_PATTERN
        Regex with a named group ``cam``.

    Returns
    -------
    list of tuple of (pathlib.Path, int)
        ``(path, camera_id)`` pairs, sorted by path.

    Raises
    ------
    RuntimeError
        If no image matches.
    """
    regex = re.compile(filename_pattern, re.IGNORECASE)
    wanted = set(cameras)
    pool = []
    for path in list_images(root):
        match = regex.search(path.name)
        if match and int(match["cam"]) in wanted:
            pool.append((path, int(match["cam"])))
    if not pool:
        raise RuntimeError(f"No image in {root} comes from cameras {sorted(wanted)}.")
    return pool


def build_geometric_views(image: Image.Image, image_size: int, global_scale: Tuple[float, float] = (0.4, 1.0),
                          rotation: Optional[RotationConfig] = None) -> torch.Tensor:
    """Create two independently augmented global views of one image.

    Each view samples a large global crop, independent flips, and an optional
    reflect-padded rotation. This retains most of the source image content.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB source image.
    image_size : int
        Output side length.

    Returns
    -------
    torch.Tensor
        Normalised views of shape ``(2, 3, image_size, image_size)``.
    """
    if isinstance(rotation, dict):
        rotation = RotationConfig(**rotation)
    rotation = rotation or RotationConfig(enabled=False)
    transform = v2.Compose([
        v2.RandomResizedCrop((image_size, image_size), scale=global_scale),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        RandomRotationReflect(rotation.max_deg, rotation.p, rotation.enabled),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return torch.stack([transform(image), transform(image)])


class ImagePoolDataset(Dataset):
    """Dataset of single images, each returned as two geometric views.

    Parameters
    ----------
    pool : list of tuple of (pathlib.Path, int)
        ``(path, camera_id)`` pairs from :func:`discover_image_pool`.
    image_size : int
        Side length of the views.
    """

    def __init__(self, pool: List[Tuple[Path, int]], image_size: int,
                 global_scale: Tuple[float, float] = (0.4, 1.0),
                 rotation: Optional[RotationConfig] = None) -> None:
        self.pool = pool
        self.image_size = image_size
        self.global_scale = global_scale
        self.rotation = rotation or RotationConfig(enabled=False)

    def __len__(self) -> int:
        """Return the number of images in the pool.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.pool)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one image and build its two views.

        Parameters
        ----------
        idx : int
            Pool index.

        Returns
        -------
        views : torch.Tensor
            Shape ``(2, 3, H, W)``.
        camera_ids : torch.Tensor
            Shape ``(2,)``; the source camera repeated for both views.
        image_ids : torch.Tensor
            Shape ``(2,)``; source-image group id repeated for both views.
        """
        path, camera = self.pool[idx]
        views = build_geometric_views(load_rgb(path), self.image_size, self.global_scale, self.rotation)
        return views, torch.full((2,), camera), torch.full((2,), idx)


def make_loader(dataset: Dataset, cfg: PretrainConfig) -> DataLoader:
    """Wrap a dataset in the shuffled, seeded pretraining DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset yielding ``(views, camera_ids, image_ids)``.
    cfg : PretrainConfig
        Provides batch size, workers and seed.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(B, V, 3, H, W)`` views and ``(B, V)`` camera ids.
    """
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def build_pretraining_dataset(cfg: PretrainConfig, verbose: bool = True) -> Dataset:
    """Create the dataset matching ``cfg.prediction_pairing``.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    verbose : bool, default=True
        Print dataset statistics.

    Returns
    -------
    torch.utils.data.Dataset
        Yields ``(views, camera_ids, image_ids)`` with views of shape ``(V, 3, H, W)``.
    """
    if cfg.prediction_pairing == "intra_image":
        pool = subsample(discover_image_pool(cfg.image_root, cfg.pool_cameras, cfg.filename_pattern),
                         cfg.max_samples, cfg.seed)
        if verbose:
            print(f"Image pool from cameras {cfg.pool_cameras}: {len(pool)} images.")
        return ImagePoolDataset(pool, cfg.image_size, cfg.global_scale, cfg.rotation)
    groups = discover_camera_groups(cfg.image_root, cfg.cameras, cfg.filename_pattern)
    sampled = subsample(groups, cfg.max_samples, cfg.seed)
    if verbose:
        print(f"Synchronised frames for cameras {cfg.cameras}: {len(groups)} found, "
              f"{len(sampled)} used ({len(sampled) * len(cfg.cameras)} images).")
    dataset = MultiCameraViewDataset(sampled, cfg.cameras, build_train_transform(cfg.image_size, cfg.rotation),
                                     cfg.views_per_camera)
    return dataset


def build_pretraining_loader(cfg: PretrainConfig, verbose: bool = True) -> DataLoader:
    """Create the shuffled DataLoader matching ``cfg.prediction_pairing``.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    verbose : bool, default=True
        Print dataset statistics.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(views, camera_ids, image_ids)`` with shapes ``(B, V, 3, H, W)`` and
        ``(B, V)``; ``V`` is the number of camera views (``"cross_camera"``)
        or 2 (``"intra_image"``).
    """
    return make_loader(build_pretraining_dataset(cfg, verbose), cfg)


# --------------------------------------------------------------------------- #
# Backbone and encoder
# --------------------------------------------------------------------------- #
def global_average_pool(feature_map: torch.Tensor) -> torch.Tensor:
    """Global average pooling of a feature map.

    Parameters
    ----------
    feature_map : torch.Tensor
        Shape ``(N, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(N, C)``.
    """
    return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)


class GeM(nn.Module):
    """Generalized mean pooling with a bounded, optionally learnable exponent."""

    def __init__(self, p_init: float = 3.0, p_min: float = 1.0, p_max: float = 6.0,
                 learnable_p: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0 < p_min <= p_init <= p_max:
            raise ValueError("GeM requires 0 < p_min <= p_init <= p_max.")
        self.p_min, self.p_max, self.eps = float(p_min), float(p_max), float(eps)
        value = torch.tensor(float(p_init))
        if learnable_p:
            self.p = nn.Parameter(value)
        else:
            self.register_buffer("p", value)

    @property
    def p_value(self) -> torch.Tensor:
        return self.p.clamp(self.p_min, self.p_max)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        p = self.p_value
        pooled = feature_map.clamp_min(self.eps).pow(p).mean(dim=(-2, -1))
        return pooled.pow(p.reciprocal())

    @torch.no_grad()
    def clamp_parameter(self) -> None:
        self.p.clamp_(self.p_min, self.p_max)


class YOLOv8MultiScaleBackbone(nn.Module):
    """First ten YOLOv8 layers returning the P3, P4 and P5 feature maps.

    Parameters
    ----------
    weights : str, default="yolov8n.yaml"
        Ultralytics source. A ``.yaml`` definition gives empty weights; a
        ``.pt`` file (e.g. ``"yolov8n.pt"``) gives its trained weights.
    """

    NUM_LAYERS = 10
    CAPTURE = {4: "P3", 6: "P4", 9: "P5"}

    def __init__(self, weights: str = EMPTY_MODEL_CFG, pretrain_neck: bool = False) -> None:
        super().__init__()
        detection_model = YOLO(str(weights)).model
        detect_layer = detection_model.model[-1]
        detect_sources = tuple(int(i) for i in detect_layer.f)
        if len(detect_sources) != 3:
            raise ValueError(f"Expected three Detect input scales, found {detect_sources}.")
        self.pretrain_neck = bool(pretrain_neck)
        self.detect_sources = detect_sources
        self.capture_indices = dict(zip(detect_sources, ("P3", "P4", "P5"))) if pretrain_neck else self.CAPTURE
        self.saved_indices = set(detection_model.save) | set(self.capture_indices)
        end = max(detect_sources) + 1 if pretrain_neck else self.NUM_LAYERS
        self.layers = nn.ModuleList(list(detection_model.model[:end]))
        self.weights = str(weights)
        for param in self.parameters():
            param.requires_grad_(True)

    @property
    def scale_names(self) -> List[str]:
        """list of str : Names of the returned scales."""
        return list(self.capture_indices.values())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute multi-scale feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(N, 3, H, W)``.

        Returns
        -------
        dict of str to torch.Tensor
            ``{"P3": ..., "P4": ..., "P5": ...}`` each ``(N, C, h, w)``.
        """
        features = {}
        outputs = []
        for index, layer in enumerate(self.layers):
            source = layer.f
            if source != -1:
                if isinstance(source, int):
                    x = outputs[source]
                else:
                    x = [x if item == -1 else outputs[item] for item in source]
            x = layer(x)
            outputs.append(x if index in self.saved_indices else None)
            if index in self.capture_indices:
                features[self.capture_indices[index]] = x
        return features

    def forward_pooled(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute globally average-pooled multi-scale features.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(N, 3, H, W)``.

        Returns
        -------
        dict of str to torch.Tensor
            Pooled vectors of shape ``(N, C)`` per scale.
        """
        return {name: global_average_pool(f) for name, f in self.forward(x).items()}

    @torch.no_grad()
    def out_channels(self, probe_size: int = 64) -> Dict[str, int]:
        """Infer the channel width of each scale with a dummy forward pass.

        Parameters
        ----------
        probe_size : int, default=64
            Side length of the dummy image (multiple of 32).

        Returns
        -------
        dict of str to int
            Channels per scale.
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        feats = self.forward(torch.zeros(1, 3, probe_size, probe_size, device=device))
        self.train(was_training)
        return {name: f.shape[1] for name, f in feats.items()}


def build_gap_projector(in_features: int, hidden_dim: int, proj_dim: int) -> nn.Sequential:
    """Three-layer MLP projector applied to a pooled vector.

    Parameters
    ----------
    in_features : int
        Width of the pooled backbone vector.
    hidden_dim : int
        Hidden width.
    proj_dim : int
        Output width.

    Returns
    -------
    torch.nn.Sequential
        The projector.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, proj_dim),
    )


class YOLOv8GAPEncoder(nn.Module):
    """Multi-scale YOLOv8 backbone followed by GAP and one projector per scale.

    Parameters
    ----------
    model_cfg : str, default="yolov8n.yaml"
        Backbone definition (empty weights when ``.yaml``).
    proj_dim : int, default=128
        Projector output width.
    hidden_dim : int, default=256
        Projector hidden width.
    """

    def __init__(self, model_cfg: str = EMPTY_MODEL_CFG, proj_dim: int = 128, hidden_dim: int = 256,
                 pooling: str = "gap", gem: Optional[GeMConfig] = None, pretrain_neck: bool = False,
                 scale_weights: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got '{pooling}'.")
        if isinstance(gem, dict):
            gem = GeMConfig(**gem)
        gem = gem or GeMConfig()
        self.init_kwargs = {"model_cfg": str(model_cfg), "proj_dim": proj_dim, "hidden_dim": hidden_dim,
                            "pooling": pooling, "gem": asdict(gem), "pretrain_neck": pretrain_neck,
                            "scale_weights": scale_weights}
        self.pooling = pooling
        self.scale_weights = {str(k).lower(): float(v) for k, v in (scale_weights or
                              {"p3": 1.0, "p4": 1.0, "p5": 1.0}).items()}
        self.backbone = YOLOv8MultiScaleBackbone(model_cfg, pretrain_neck=pretrain_neck)
        channels = self.backbone.out_channels()
        active_scales = [name for name in self.backbone.scale_names
                         if self.scale_weights.get(name.lower(), 0.0) > 0]
        self.poolers = nn.ModuleDict({
            name: (GeM(gem.p_init, gem.p_min, gem.p_max, gem.learnable_p) if pooling == "gem"
                   else nn.Identity()) for name in active_scales
        })
        self.projectors = nn.ModuleDict({
            name: build_gap_projector(channels[name], hidden_dim, proj_dim)
            for name in active_scales
        })

    def project(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project pooled features of a flat image batch.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``(N, 3, H, W)``.

        Returns
        -------
        dict of str to torch.Tensor
            Projections of shape ``(N, proj_dim)`` per scale.
        """
        maps = self.backbone(images)
        pooled = {
            name: (global_average_pool(feature) if self.pooling == "gap"
                   else self.poolers[name](feature))
            for name, feature in maps.items() if name in self.projectors
        }
        return {name: self.projectors[name](vec) for name, vec in pooled.items()}

    def p_values(self) -> Dict[str, float]:
        return {name: float(module.p_value.detach().cpu()) for name, module in self.poolers.items()
                if isinstance(module, GeM)}

    @torch.no_grad()
    def clamp_gem_parameters(self) -> None:
        for module in self.poolers.values():
            if isinstance(module, GeM):
                module.clamp_parameter()

    def forward(self, views: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project every view of every sample.

        Parameters
        ----------
        views : torch.Tensor
            Shape ``(B, V, 3, H, W)``.

        Returns
        -------
        dict of str to torch.Tensor
            Projections of shape ``(B, V, proj_dim)`` per scale.
        """
        batch, num_views = views.shape[:2]
        projections = self.project(views.flatten(0, 1))
        return {name: z.reshape(batch, num_views, -1) for name, z in projections.items()}

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Concatenate all scale projections into one embedding.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, num_scales * proj_dim)``.
        """
        return torch.cat(list(self.project(images).values()), dim=-1)


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regulariser (Epps-Pulley statistic).

    Parameters
    ----------
    knots : int, default=17
        Integration points of the characteristic-function test on ``[0, 3]``.
    num_slices : int, default=256
        Random unit directions sampled per call.
    """

    def __init__(self, knots: int = 17, num_slices: int = 256) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_slices = num_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Measure the distance of the embeddings from an isotropic Gaussian.

        Parameters
        ----------
        proj : torch.Tensor
            Shape ``(V, N, D)`` or ``(N, D)``.

        Returns
        -------
        torch.Tensor
            Scalar statistic averaged over views and slices.
        """
        # What is tested: the projector outputs (the embeddings), per view.
        # Against what: a standard Gaussian N(0, I). A Gaussian is the target
        # because matching it spreads the embeddings isotropically, which rules
        # out the trivial collapsed solution (all images -> one point).
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)  # (N, D) -> (1, N, D): a single "view"
        n_samples, dim = proj.shape[1], proj.shape[2]
        # Sketching: a high-dimensional Gaussian test is replaced by many 1-D
        # tests along random unit directions (the Cramer-Wold idea).
        directions = torch.randn(dim, self.num_slices, device=proj.device)
        directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-12)  # unit length, (D, S)
        # A projection onto a unit direction turns each D-dimensional embedding
        # into one scalar. Repeating this for S directions gives many cheap
        # one-dimensional tests instead of estimating a D-dimensional density.
        # The Cramer-Wold principle says that matching all one-dimensional
        # projections is sufficient to identify the full distribution.
        #
        # Shape bookkeeping:
        #   proj @ directions:                  (V, N, S)
        #   multiply by every integration knot: (V, N, S, K)
        # Here V is the view, N the batch/sample count, S the random slice, and
        # K the number of points used to numerically integrate over t.
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        # A distribution's characteristic function is E[exp(i*t*x)]. Its
        # empirical real and imaginary parts are respectively mean(cos(t*x))
        # and mean(sin(t*x)) over the N samples. For a standard normal, the
        # characteristic function is exp(-t^2/2), with zero imaginary part.
        #
        # Thus `err` is the squared complex distance between the observed and
        # standard-normal characteristic functions at every view, slice, and
        # knot: (observed_real - target_real)^2 + (observed_imag - 0)^2.
        # Averaging over samples here is what makes this a distribution-level
        # test rather than a pointwise penalty on each embedding.
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        # Numerically integrate the characteristic-function discrepancy over
        # t. `self.weights` contains trapezoidal integration weights multiplied
        # by the Gaussian window, which gives more emphasis to the central
        # region. The Epps-Pulley statistic scales the integral by N; averaging
        # the resulting values across views and random slices gives one scalar
        # regularisation loss for this batch.
        return ((err @ self.weights) * n_samples).mean()


def cross_camera_prediction_loss(z: torch.Tensor, num_cameras: int) -> torch.Tensor:
    """Variant 1 of ``L_pred``: predict each camera's embedding from the other cameras' mean.

    Every view of camera ``c`` is pulled toward the average embedding of all
    views from the *other* cameras of the same frame, so the invariance being
    learned is between camera viewpoints rather than only between
    augmentations of one image.

    Parameters
    ----------
    z : torch.Tensor
        Shape ``(B, V, D)`` with views ordered camera-major.
    num_cameras : int
        Number of cameras ``C``; ``V`` must be a multiple of ``C``.

    Returns
    -------
    torch.Tensor
        Scalar mean squared error.

    Raises
    ------
    ValueError
        If fewer than two cameras are given or ``V`` is not divisible by ``C``.
    """
    batch, num_views, dim = z.shape
    if num_cameras < 2 or num_views % num_cameras:
        raise ValueError(f"{num_views} views cannot be split across {num_cameras} cameras.")
    # The loader stacks camera 0's augmentations, then camera 1's, and so on.
    # Restore that organization so the camera and augmentation axes are
    # explicit: (B, V, D) -> (B, C, views_per_camera, D).
    per_camera = z.reshape(batch, num_cameras, num_views // num_cameras, dim)
    # Collapse only the augmentation axis. Each frame now has one representative
    # embedding per camera, shape (B, C, D).
    camera_means = per_camera.mean(dim=2)
    # For camera c, form its target from the *other* cameras rather than
    # including its own embedding. Subtracting its mean from the sum avoids
    # constructing a separate masked tensor for every camera:
    #   target_c = (sum_j camera_mean_j - camera_mean_c) / (C - 1)
    # The resulting shape is (B, C, D), with one cross-view target per camera.
    other_means = (camera_means.sum(dim=1, keepdim=True) - camera_means) / (num_cameras - 1)
    # Compare every augmented view to its camera's other-camera target. The
    # mean squares average equally over frames, cameras, augmentations, and
    # embedding dimensions; gradients flow through both views and targets.
    return (per_camera - other_means.unsqueeze(2)).square().mean()


def assert_image_groups(image_ids: Optional[torch.Tensor], batch: int, num_views: int) -> None:
    """Ensure each prediction row contains views from exactly one source image."""
    if image_ids is None:
        return
    if image_ids.shape != (batch, num_views):
        raise AssertionError(f"image_ids must have shape {(batch, num_views)}, got {tuple(image_ids.shape)}.")
    if not torch.equal(image_ids, image_ids[:, :1].expand_as(image_ids)):
        raise AssertionError("Prediction groups mix views from different source images.")
    if torch.unique(image_ids[:, 0]).numel() != batch:
        raise AssertionError("A source image appears in more than one prediction group.")


def intra_image_prediction_loss(z: torch.Tensor, image_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Variant 2 of ``L_pred``: compare two geometric views of one image.

    All views are augmented copies of the *same* image. Each view is compared
    with that source image's mean embedding, never with another source image.

    Parameters
    ----------
    z : torch.Tensor
        Shape ``(B, V, D)``: all view embeddings of each source image.

    Returns
    -------
    torch.Tensor
        Scalar mean squared error between each view and its group's mean.

    Raises
    ------
    ValueError
        If ``z`` does not hold at least two views per image.
    """
    if z.dim() != 3 or z.shape[1] < 2:
        raise ValueError(f"intra-image loss needs shape (B, V>=2, D), got {tuple(z.shape)}.")
    assert_image_groups(image_ids, z.shape[0], z.shape[1])
    targets = z.mean(dim=1, keepdim=True)
    return (z - targets).square().mean()


def prediction_loss(z: torch.Tensor, mode: str, num_cameras: int = 0,
                    image_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatch to the prediction loss selected by ``pred_loss_mode``.

    Parameters
    ----------
    z : torch.Tensor
        Per-scale projections of shape ``(B, V, D)``.
    mode : {"cross_camera", "intra_image"}
        Loss variant.
    num_cameras : int, default=0
        Number of cameras; only used by ``"cross_camera"``.

    Returns
    -------
    torch.Tensor
        Scalar prediction loss.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """
    if image_ids is not None:
        assert_image_groups(image_ids, z.shape[0], z.shape[1])
    if mode == "cross_camera":
        return cross_camera_prediction_loss(z, num_cameras)
    if mode == "intra_image":
        return intra_image_prediction_loss(z, image_ids)
    raise ValueError(f"pred_loss_mode must be one of {PRED_LOSS_MODES}, got '{mode}'.")


def lejepa_loss(
    projections: Dict[str, torch.Tensor],
    sigreg: SIGReg,
    num_cameras: int,
    lam: float,
    pred_loss_mode: str = "cross_camera",
    scale_weights: Optional[Dict[str, float]] = None,
    image_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine the prediction loss and SIGReg over all scales.

    Parameters
    ----------
    projections : dict of str to torch.Tensor
        Per-scale projections of shape ``(B, V, D)``.
    sigreg : SIGReg
        Regulariser module.
    num_cameras : int
        Number of cameras in each sample.
    lam : float
        SIGReg weight in ``[0, 1]``.
    pred_loss_mode : {"cross_camera", "intra_image"}, default="cross_camera"
        Prediction-loss variant.

    Returns
    -------
    loss : torch.Tensor
        ``(1 - lam) * prediction + lam * sigreg`` averaged over scales.
    logs : dict of str to torch.Tensor
        Detached components (``total``, ``pred``, ``sigreg`` and per scale).
    """
    logs: Dict[str, torch.Tensor] = {}
    if scale_weights is None:
        scale_weights = {name.lower(): 1.0 / len(projections) for name in projections}
    active_weights = {name: float(scale_weights.get(name.lower(), 0.0)) for name in projections}
    if not any(weight > 0 for weight in active_weights.values()):
        raise ValueError("At least one projected scale must have a positive loss weight.")
    for name, z in projections.items():
        # Each backbone scale has its own projected embeddings z with shape
        # (batch, views, projection_dim). Its selected prediction objective
        # encourages agreement across cameras or between two transforms of one
        # image, depending on `pred_loss_mode`.
        logs[f"pred_{name}"] = prediction_loss(z, pred_loss_mode, num_cameras, image_ids)
        # SIGReg is independent of the prediction mode. Put views first because
        # SIGReg interprets its leading axis as separate view groups and tests
        # each group's batch distribution against N(0, I): (B, V, D) -> (V, B, D).
        logs[f"sigreg_{name}"] = sigreg(z.transpose(0, 1))
    loss = sum(
        active_weights[name] * ((1.0 - lam) * logs[f"pred_{name}"] + lam * logs[f"sigreg_{name}"])
        for name in projections
    )
    pred = sum(active_weights[name] * logs[f"pred_{name}"] for name in projections)
    sig = sum(active_weights[name] * logs[f"sigreg_{name}"] for name in projections)
    logs.update(total=loss, pred=pred, sigreg=sig)
    # Keep the training loss attached to autograd, but detach every value saved
    # for reporting so epoch-history accumulation does not retain computation
    # graphs or GPU activations.
    return loss, {k: v.detach() for k, v in logs.items()}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one_epoch(
    encoder: YOLOv8GAPEncoder,
    loader: DataLoader,
    sigreg: SIGReg,
    optimizer: torch.optim.Optimizer,
    num_cameras: int,
    lam: float,
    device: torch.device,
    desc: str = "",
    pred_loss_mode: str = "cross_camera",
    scale_weights: Optional[Dict[str, float]] = None,
    cfg: Optional[PretrainConfig] = None,
    global_step: int = 0,
    step_counter: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Run one optimisation pass over the loader.

    Parameters
    ----------
    encoder : YOLOv8GAPEncoder
        Model being trained.
    loader : torch.utils.data.DataLoader
        Yields ``(views, camera_ids)``.
    sigreg : SIGReg
        Regulariser module.
    optimizer : torch.optim.Optimizer
        Optimiser over the encoder parameters.
    num_cameras : int
        Number of cameras per sample.
    lam : float
        SIGReg weight.
    device : torch.device
        Compute device.
    desc : str, default=""
        Optional label prefix for explicitly enabled step diagnostics.
        No per-batch progress bar is displayed.
    pred_loss_mode : {"cross_camera", "intra_image"}, default="cross_camera"
        Prediction-loss variant.
    cfg : PretrainConfig or None, default=None
        Optional legacy per-step diagnostics configuration; intervals default
        to zero so they are disabled unless explicitly enabled.
    global_step : int, default=0
        Starting step index for optional diagnostics.
    step_counter : list of int or None, default=None
        Optional mutable step state updated after every batch.

    Returns
    -------
    dict of str to float
        Epoch-averaged loss components, or an empty dictionary when the loader
        yields no batches.
    """
    encoder.train()
    totals: Dict[str, torch.Tensor] = {}
    n_batches = 0
    for batch in loader:
        views, _, image_ids = batch if len(batch) == 3 else (*batch, None)
        views = views.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True) if image_ids is not None else None
        loss, logs = lejepa_loss(
            encoder(views), sigreg, num_cameras, lam, pred_loss_mode, scale_weights, image_ids
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        encoder.clamp_gem_parameters()
        for key, value in logs.items():
            value = torch.as_tensor(value, device=device).detach().float().reshape(())
            totals[key] = totals[key] + value if key in totals else value
        n_batches += 1
        global_step += 1
        if step_counter is not None:
            if step_counter:
                step_counter[0] = global_step
            else:
                step_counter.append(global_step)
        if cfg and cfg.log_every_n_steps and global_step % cfg.log_every_n_steps == 0:
            label = f"{desc} " if desc else ""
            print(f"{label}step={global_step} " + " ".join(f"{name}.p={value:.3f}"
                  for name, value in encoder.p_values().items()))
        if cfg and cfg.debug_every_n_steps and global_step % cfg.debug_every_n_steps == 0:
            debug_path = cfg.output_dir / "debug" / f"p3_activation_step_{global_step:07d}.png"
            save_activation_debug_grid(encoder, views, debug_path)
    if not totals:
        return {}
    keys = list(totals)
    averages = torch.stack([totals[key] / n_batches for key in keys]).cpu().tolist()
    return dict(zip(keys, averages))


@torch.no_grad()
def save_activation_debug_grid(encoder: YOLOv8GAPEncoder, views: torch.Tensor, output_path: Path,
                               max_views: int = 4) -> Path:
    """Save source views with their P3 GeM activation-energy overlays."""
    import matplotlib.pyplot as plt

    was_training = encoder.training
    encoder.eval()
    images = views.flatten(0, 1)[:max_views]
    maps = encoder.backbone(images)["P3"]
    pooler = encoder.poolers["P3"] if "P3" in encoder.poolers else None
    exponent = float(pooler.p_value) if isinstance(pooler, GeM) else 1.0
    energy = maps.clamp_min(1e-6).pow(exponent).mean(dim=1, keepdim=True)
    energy = F.interpolate(energy, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(images), 1, figsize=(7, 3 * len(images)), squeeze=False)
    for i, (image, heatmap) in enumerate(zip(images, energy)):
        rgb = image.detach().cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN), 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].imshow(heatmap.detach().cpu().numpy(), cmap="magma", alpha=0.45)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"P3 channel mean(x^{exponent:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    encoder.train(was_training)
    return output_path


def pretrain(
    cfg: PretrainConfig,
    loader: Optional[DataLoader] = None,
    on_epoch_end: Optional[EpochCallback] = None,
    verbose: bool = True,
) -> Tuple[YOLOv8GAPEncoder, pd.DataFrame]:
    """Pretrain an empty-weight YOLOv8 backbone with GAP LeJEPA.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    loader : torch.utils.data.DataLoader or None, default=None
        Pre-built loader; built from ``cfg`` when ``None``.
    on_epoch_end : callable or None, default=None
        Called as ``on_epoch_end(epoch, encoder, loader)`` after each epoch.
    verbose : bool, default=True
        Print one summary line per epoch; no per-batch progress bar is shown.

    Returns
    -------
    encoder : YOLOv8GAPEncoder
        Trained encoder on the selected device.
    history : pandas.DataFrame
        One row per epoch with every loss component.
    """
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    loader = build_pretraining_loader(cfg, verbose) if loader is None else loader
    try:
        batches_per_epoch = len(loader)
    except TypeError:
        batches_per_epoch = None
    if batches_per_epoch == 0:
        raise ValueError("Pretraining data loader contains no batches.")
    encoder = YOLOv8GAPEncoder(
        cfg.model_cfg, cfg.proj_dim, cfg.hidden_dim, cfg.pooling, cfg.gem,
        cfg.pretrain_neck, cfg.scale_weights,
    ).to(device)
    sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_slices).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    records = []
    step_counter = [0]
    for epoch in range(cfg.epochs):
        stats = train_one_epoch(encoder, loader, sigreg, optimizer, len(cfg.cameras), cfg.lam, device,
                                desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                                pred_loss_mode=cfg.prediction_pairing, scale_weights=cfg.scale_weights,
                                cfg=cfg, global_step=step_counter[0], step_counter=step_counter)
        if not stats:
            raise ValueError("Pretraining data loader yielded no batches.")
        stats.update({f"p_{name.lower()}": value for name, value in encoder.p_values().items()})
        records.append({"epoch": epoch + 1, **stats})
        if verbose:
            print(f"Epoch {epoch + 1:03d} | " + " | ".join(f"{k}={v:.5f}" for k, v in stats.items()))
        if on_epoch_end is not None:
            on_epoch_end(epoch, encoder, loader)
    return encoder, pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Checkpoints and export
# --------------------------------------------------------------------------- #
def save_encoder(encoder: YOLOv8GAPEncoder, path: Path, cfg: Optional[PretrainConfig] = None) -> Path:
    """Save encoder weights and the arguments needed to rebuild it.

    Parameters
    ----------
    encoder : YOLOv8GAPEncoder
        Trained encoder.
    path : pathlib.Path
        Destination ``.pth`` file.
    cfg : PretrainConfig or None, default=None
        Configuration stored alongside for provenance.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "encoder_kwargs": encoder.init_kwargs,
        "pretrain_config": cfg.to_dict() if cfg else {},
    }, path)
    return path


def load_encoder(path: Path, map_location: str = "cpu") -> YOLOv8GAPEncoder:
    """Rebuild a pretrained encoder from a checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        File written by :func:`save_encoder`.
    map_location : str, default="cpu"
        Device the tensors are loaded onto.

    Returns
    -------
    YOLOv8GAPEncoder
        Encoder with restored weights.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    encoder = YOLOv8GAPEncoder(**checkpoint["encoder_kwargs"])
    encoder.load_state_dict(checkpoint["encoder_state"])
    return encoder


def read_pretrain_config(path: Path) -> dict:
    """Read the configuration stored in an encoder checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        File written by :func:`save_encoder`.

    Returns
    -------
    dict
        The saved ``PretrainConfig.to_dict()`` (empty if none was stored).
    """
    return torch.load(Path(path), map_location="cpu", weights_only=True).get("pretrain_config", {})


def load_backbone(path: Path) -> YOLOv8MultiScaleBackbone:
    """Load only the pretrained backbone from an encoder checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        File written by :func:`save_encoder`.

    Returns
    -------
    YOLOv8MultiScaleBackbone
        Backbone with LeJEPA weights.
    """
    return load_encoder(path).backbone


def export_backbone_to_yolo(
    backbone: YOLOv8MultiScaleBackbone,
    output_path: Path,
    model_cfg: str = EMPTY_MODEL_CFG,
    transfer: str = "backbone",
) -> Path:
    """Write an Ultralytics checkpoint with pretrained backbone and optional neck.

    The Detect head keeps the empty (random) initialisation of ``model_cfg``.
    Saving a real ``.pt`` checkpoint is required because
    ``YOLO("*.yaml").train()`` rebuilds the model from scratch and would
    silently discard weights copied into it in memory.

    Parameters
    ----------
    backbone : YOLOv8MultiScaleBackbone
        Source of the backbone weights.
    output_path : pathlib.Path
        Destination ``.pt`` file.
    model_cfg : str, default="yolov8n.yaml"
        Architecture of the exported detector.
    transfer : {"backbone", "backbone+neck"}, default="backbone"
        Layers copied from the pretraining encoder.

    Returns
    -------
    pathlib.Path
        The written checkpoint, loadable with ``YOLO(path)``.
    """
    if transfer not in TRANSFER_MODES:
        raise ValueError(f"transfer must be one of {TRANSFER_MODES}, got '{transfer}'.")
    detection_model = YOLO(str(model_cfg)).model
    num_layers = YOLOv8MultiScaleBackbone.NUM_LAYERS if transfer == "backbone" else len(backbone.layers)
    if num_layers > len(backbone.layers) or num_layers >= len(detection_model.model):
        raise ValueError("Requested transfer layers are not available in the pretrained or target model.")
    source = nn.ModuleList(list(backbone.layers[:num_layers])).state_dict()
    target = nn.ModuleList(list(detection_model.model[:num_layers]))
    target_keys = target.state_dict()
    matched = sorted(set(source) & set(target_keys))
    missing = sorted(set(target_keys) - set(source))
    unexpected = sorted(set(source) - set(target_keys))
    print(f"Export key match ({transfer}): matched={len(matched)}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    target.load_state_dict(source, strict=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": deepcopy(detection_model).float(),
        "train_args": {},
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "transfer": transfer,
    }, output_path)
    return output_path
