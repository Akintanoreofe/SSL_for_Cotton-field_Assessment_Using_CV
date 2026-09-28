"""Run cross-camera GAP LeJEPA pretraining, evaluation and visualization.

Edit the paths below, then run ``python run_pipeline.py``. This script only
builds configurations and calls package functions; all logic lives in
``lejepa_cotton``.
"""

from pathlib import Path

from lejepa_cotton import DetectionEvalConfig, PretrainConfig, ProbeEvalConfig, WeightSources, run_all

# ---- Paths (the only place directories are defined) ------------------------
MULTI_CAMERA_ROOT = Path("/Users/akintanoreofeoluwa/Downloads/LeJEPA _pretrainining_multi_camera_boll/mars_multi_camera_boll")
DETECTION_DATA = Path("image_dataset")
PLOT_STATUS_DIR = Path("Plot Status")
OUTPUT_ROOT = Path("outputs")

if __name__ == "__main__":
    pretrain_cfg = PretrainConfig(
        image_root=MULTI_CAMERA_ROOT,
        output_dir=OUTPUT_ROOT / "pretraining",
        cameras=(1, 2, 4),          # 2nd, 3rd and 5th physical cameras (0-indexed)
        max_samples=16_667,         # synchronised triplets -> ~50k images
        image_size=128,
        batch_size=16,
        epochs=60,
        lam=0.2,
    )
    weights = WeightSources(lejepa_checkpoint=pretrain_cfg.checkpoint_path)

    detection_cfg = DetectionEvalConfig(
        source_dir=DETECTION_DATA,
        output_dir=OUTPUT_ROOT / "detection",
        weights=weights,
        variants=("lejepa", "coco"),
        epochs=40,
        device="cpu",
    )
    probe_cfg = ProbeEvalConfig(
        image_dir=PLOT_STATUS_DIR,
        annotation_path=PLOT_STATUS_DIR / "annotations.json",
        output_dir=OUTPUT_ROOT / "plot_status_probe",
        weights=weights,
        variants=("lejepa", "coco"),
    )

    results = run_all(pretrain_cfg, detection_cfg, probe_cfg)
    print(results["detection"].to_string(index=False))
    print(results["probe"].to_string(index=False))
