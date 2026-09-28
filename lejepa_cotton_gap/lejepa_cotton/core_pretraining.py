"""Cross-camera LeJEPA pretraining of a YOLOv8 backbone with global average pooling.

This module owns everything needed to *learn* a representation:

* discovering synchronised multi-camera frames on disk,
* the multi-camera view dataset and its augmentations,
* the randomly initialised (``yolov8n.yaml``) multi-scale backbone,
* the global-average-pooling (GAP) encoder and projectors,
* the SIGReg regulariser and the cross-camera prediction loss,
* the training loop, checkpoint I/O and export of the backbone into an
  Ultralytics detection checkpoint.

No directory is hard coded: every path is supplied by the caller through
:class:`PretrainConfig` or through function arguments.
"""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
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

CameraGroup = Dict[int, Path]
EpochCallback = Callable[[int, nn.Module, DataLoader], None]


# --------------------------------------------------------------------------- #
# Configuration and device
# --------------------------------------------------------------------------- #
@dataclass
class PretrainConfig:
    """Hyper-parameters and paths for cross-camera GAP LeJEPA pretraining.

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
    image_size : int, default=128
        Side length of the square crops fed to the backbone.
    batch_size : int, default=16
        Number of frame groups per optimisation step.
    epochs : int, default=60
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
    checkpoint_name : str, default="gap_lejepa_yolov8n.pth"
        File name of the checkpoint written inside ``output_dir``.
    """

    image_root: Path
    output_dir: Path
    cameras: Tuple[int, ...] = (1, 2, 4)
    max_samples: Optional[int] = None
    views_per_camera: int = 1
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 60
    proj_dim: int = 128
    hidden_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lam: float = 0.2
    sigreg_knots: int = 17
    sigreg_slices: int = 256
    model_cfg: str = EMPTY_MODEL_CFG
    filename_pattern: str = DEFAULT_FILENAME_PATTERN
    num_workers: int = 0
    seed: int = 42
    device: Optional[str] = None
    checkpoint_name: str = "gap_lejepa_yolov8n.pth"

    def __post_init__(self) -> None:
        self.image_root = Path(self.image_root)
        self.output_dir = Path(self.output_dir)
        self.cameras = tuple(int(c) for c in self.cameras)
        if len(self.cameras) < 2:
            raise ValueError("Cross-camera prediction needs at least two cameras.")

    @property
    def checkpoint_path(self) -> Path:
        """pathlib.Path : Location of the pretrained encoder checkpoint."""
        return self.output_dir / self.checkpoint_name

    def to_dict(self) -> dict:
        """Serialise the configuration with paths converted to strings.

        Returns
        -------
        dict
            Plain-Python representation safe for ``torch.save``.
        """
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


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
def build_train_transform(image_size: int) -> v2.Compose:
    """Geometric augmentation applied independently to every camera view.

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
        v2.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.1),
        v2.RandomRotation(25),
        v2.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        """
        views, camera_ids = [], []
        for cam in self.cameras:
            image = load_rgb(self.groups[idx][cam])
            for _ in range(self.views_per_camera):
                views.append(self.transform(image))
                camera_ids.append(cam)
        return torch.stack(views), torch.tensor(camera_ids)


def build_pretraining_loader(cfg: PretrainConfig, verbose: bool = True) -> DataLoader:
    """Create the shuffled multi-camera DataLoader.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    verbose : bool, default=True
        Print dataset statistics.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(views, camera_ids)`` with shapes ``(B, V, 3, H, W)`` and
        ``(B, V)``.
    """
    groups = discover_camera_groups(cfg.image_root, cfg.cameras, cfg.filename_pattern)
    sampled = subsample(groups, cfg.max_samples, cfg.seed)
    if verbose:
        print(f"Synchronised frames for cameras {cfg.cameras}: {len(groups)} found, "
              f"{len(sampled)} used ({len(sampled) * len(cfg.cameras)} images).")
    dataset = MultiCameraViewDataset(sampled, cfg.cameras, build_train_transform(cfg.image_size),
                                     cfg.views_per_camera)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(cfg.seed),
    )


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

    def __init__(self, weights: str = EMPTY_MODEL_CFG) -> None:
        super().__init__()
        detection_model = YOLO(str(weights)).model
        self.layers = nn.ModuleList(list(detection_model.model[: self.NUM_LAYERS]))
        self.weights = str(weights)
        for param in self.parameters():
            param.requires_grad_(True)

    @property
    def scale_names(self) -> List[str]:
        """list of str : Names of the returned scales."""
        return list(self.CAPTURE.values())

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
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in self.CAPTURE:
                features[self.CAPTURE[index]] = x
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

    def __init__(self, model_cfg: str = EMPTY_MODEL_CFG, proj_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.init_kwargs = {"model_cfg": str(model_cfg), "proj_dim": proj_dim, "hidden_dim": hidden_dim}
        self.backbone = YOLOv8MultiScaleBackbone(model_cfg)
        self.projectors = nn.ModuleDict({
            name: build_gap_projector(channels, hidden_dim, proj_dim)
            for name, channels in self.backbone.out_channels().items()
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
        pooled = self.backbone.forward_pooled(images)
        return {name: self.projectors[name](vec) for name, vec in pooled.items()}

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
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        n_samples, dim = proj.shape[1], proj.shape[2]
        directions = torch.randn(dim, self.num_slices, device=proj.device)
        directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-12)
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * n_samples).mean()


def cross_camera_prediction_loss(z: torch.Tensor, num_cameras: int) -> torch.Tensor:
    """Predict each camera's embedding from the other cameras' mean.

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
    per_camera = z.reshape(batch, num_cameras, num_views // num_cameras, dim)
    camera_means = per_camera.mean(dim=2)
    other_means = (camera_means.sum(dim=1, keepdim=True) - camera_means) / (num_cameras - 1)
    return (per_camera - other_means.unsqueeze(2)).square().mean()


def lejepa_loss(
    projections: Dict[str, torch.Tensor],
    sigreg: SIGReg,
    num_cameras: int,
    lam: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine the cross-camera prediction loss and SIGReg over all scales.

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

    Returns
    -------
    loss : torch.Tensor
        ``(1 - lam) * prediction + lam * sigreg`` averaged over scales.
    logs : dict of str to torch.Tensor
        Detached components (``total``, ``pred``, ``sigreg`` and per scale).
    """
    logs: Dict[str, torch.Tensor] = {}
    for name, z in projections.items():
        logs[f"pred_{name}"] = cross_camera_prediction_loss(z, num_cameras)
        logs[f"sigreg_{name}"] = sigreg(z.transpose(0, 1))
    pred = torch.stack([logs[f"pred_{n}"] for n in projections]).mean()
    sig = torch.stack([logs[f"sigreg_{n}"] for n in projections]).mean()
    loss = (1.0 - lam) * pred + lam * sig
    logs.update(total=loss, pred=pred, sigreg=sig)
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
        Progress-bar label.

    Returns
    -------
    dict of str to float
        Epoch-averaged loss components.
    """
    encoder.train()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0
    for views, _ in tqdm.tqdm(loader, desc=desc, leave=False):
        loss, logs = lejepa_loss(encoder(views.to(device)), sigreg, num_cameras, lam)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for key, value in logs.items():
            totals[key] += float(value.cpu())
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


def pretrain(
    cfg: PretrainConfig,
    loader: Optional[DataLoader] = None,
    on_epoch_end: Optional[EpochCallback] = None,
    verbose: bool = True,
) -> Tuple[YOLOv8GAPEncoder, pd.DataFrame]:
    """Pretrain an empty-weight YOLOv8 backbone with cross-camera GAP LeJEPA.

    Parameters
    ----------
    cfg : PretrainConfig
        Pretraining configuration.
    loader : torch.utils.data.DataLoader or None, default=None
        Pre-built loader; built from ``cfg`` when ``None``.
    on_epoch_end : callable or None, default=None
        Called as ``on_epoch_end(epoch, encoder, loader)`` after each epoch.
    verbose : bool, default=True
        Print one summary line per epoch.

    Returns
    -------
    encoder : YOLOv8GAPEncoder
        Trained encoder on the selected device.
    history : pandas.DataFrame
        One row per epoch with every loss component.
    """
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    loader = loader or build_pretraining_loader(cfg, verbose)
    encoder = YOLOv8GAPEncoder(cfg.model_cfg, cfg.proj_dim, cfg.hidden_dim).to(device)
    sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_slices).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    records = []
    for epoch in range(cfg.epochs):
        stats = train_one_epoch(encoder, loader, sigreg, optimizer, len(cfg.cameras), cfg.lam, device,
                                desc=f"Epoch {epoch + 1}/{cfg.epochs}")
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
) -> Path:
    """Write an Ultralytics checkpoint whose layers 0-9 come from ``backbone``.

    The neck and head keep the empty (random) initialisation of
    ``model_cfg``. Saving a real ``.pt`` checkpoint is required because
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

    Returns
    -------
    pathlib.Path
        The written checkpoint, loadable with ``YOLO(path)``.
    """
    detection_model = YOLO(str(model_cfg)).model
    target = nn.ModuleList(list(detection_model.model[: len(backbone.layers)]))
    target.load_state_dict(backbone.layers.state_dict())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": deepcopy(detection_model).float(),
        "train_args": {},
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
    }, output_path)
    return output_path
