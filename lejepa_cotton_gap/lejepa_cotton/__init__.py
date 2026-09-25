"""lejepa_cotton: cross-camera GAP LeJEPA pretraining of YOLOv8 for cotton imagery.

Modules
-------
core_pretraining
    Multi-camera data, empty-weight YOLOv8 GAP encoder, SIGReg, cross-camera
    prediction loss, training loop, checkpoint I/O and export to Ultralytics.
core_evaluation
    Detection fine-tuning and frozen-backbone linear probing that compare the
    LeJEPA backbone against COCO weights.
visualization
    Loss curves, PCA embeddings, confusion matrices, metric bars and
    detection overlays.
pipeline
    Runners that combine the three modules end to end.
"""

from .core_evaluation import (
    VARIANTS,
    DetectionEvalConfig,
    LinearProbe,
    ProbeEvalConfig,
    WeightSources,
    build_backbone_for_variant,
    classification_summary,
    PlotStatusDataset,
    extract_embeddings,
    find_labeled_pairs,
    prepare_detection_split,
    verify_backbone_transfer,
)
from .core_pretraining import (
    PretrainConfig,
    SIGReg,
    YOLOv8GAPEncoder,
    YOLOv8MultiScaleBackbone,
    cross_camera_prediction_loss,
    discover_camera_groups,
    export_backbone_to_yolo,
    load_backbone,
    load_encoder,
    pretrain,
    save_encoder,
    select_device,
)
from .pipeline import run_all, run_detection_evaluation, run_pretraining, run_probe_evaluation

__version__ = "0.1.0"

__all__ = [
    "VARIANTS",
    "DetectionEvalConfig",
    "LinearProbe",
    "PretrainConfig",
    "ProbeEvalConfig",
    "SIGReg",
    "WeightSources",
    "YOLOv8GAPEncoder",
    "YOLOv8MultiScaleBackbone",
    "build_backbone_for_variant",
    "classification_summary",
    "cross_camera_prediction_loss",
    "discover_camera_groups",
    "export_backbone_to_yolo",
    "PlotStatusDataset",
    "extract_embeddings",
    "find_labeled_pairs",
    "load_backbone",
    "load_encoder",
    "prepare_detection_split",
    "pretrain",
    "run_all",
    "run_detection_evaluation",
    "run_pretraining",
    "run_probe_evaluation",
    "save_encoder",
    "select_device",
    "verify_backbone_transfer",
    "__version__",
]