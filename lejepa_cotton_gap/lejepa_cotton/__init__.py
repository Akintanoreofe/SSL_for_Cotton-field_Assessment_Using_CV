"""lejepa_cotton: GAP LeJEPA pretraining of YOLOv8 for cotton imagery.

Modules
-------
core_pretraining
    Multi-camera data, empty-weight YOLOv8 GAP encoder, SIGReg, both
    prediction losses (cross-camera and intra-image), lambda-sweep helpers,
    training loop, checkpoint I/O and export to Ultralytics.
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
    GeM,
    GeMConfig,
    PretrainConfig,
    RandomRotationReflect,
    RotationConfig,
    SIGReg,
    YOLOv8GAPEncoder,
    YOLOv8MultiScaleBackbone,
    build_geometric_views,
    cross_camera_prediction_loss,
    discover_camera_groups,
    intra_image_prediction_loss,
    lambda_artifact_paths,
    export_backbone_to_yolo,
    load_backbone,
    load_encoder,
    pretrain,
    save_encoder,
    select_device,
)
from .pipeline import (
    plot_lambda_losses,
    run_all,
    run_detection_evaluation,
    run_lambda_evaluation,
    run_lambda_sweep,
    run_pretraining,
    run_probe_evaluation,
)
from .predict_video import predict_video

__version__ = "0.1.0"

__all__ = [
    "VARIANTS",
    "DetectionEvalConfig",
    "GeM",
    "GeMConfig",
    "LinearProbe",
    "PretrainConfig",
    "ProbeEvalConfig",
    "RandomRotationReflect",
    "RotationConfig",
    "SIGReg",
    "WeightSources",
    "YOLOv8GAPEncoder",
    "YOLOv8MultiScaleBackbone",
    "build_backbone_for_variant",
    "classification_summary",
    "build_geometric_views",
    "cross_camera_prediction_loss",
    "intra_image_prediction_loss",
    "lambda_artifact_paths",
    "plot_lambda_losses",
    "run_lambda_evaluation",
    "run_lambda_sweep",
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
    "predict_video",
    "__version__",
]