# SSL for Cotton Field Assessment Using Computer Vision

This repository contains a self-supervised learning pipeline for cotton imagery. It pretrains YOLOv8n with LeJEPA (learnable GeM and neck/P3 pretraining by default, with a legacy GAP mode), then evaluates the learned features on cotton-boll detection and plot-status classification.

## Start here

The runnable Python package is in [`lejepa_cotton_gap/`](lejepa_cotton_gap/). Its [README](lejepa_cotton_gap/README.md) documents the method, datasets, configuration options, evaluation variants, and generated outputs.

To install the package from a local checkout:

```bash
cd lejepa_cotton_gap
python -m pip install -e .
```

For a complete run, edit the dataset and output paths in [`run_pipeline.py`](lejepa_cotton_gap/run_pipeline.py), then run it from the package directory:

```bash
python run_pipeline.py
```

Alternatively, open [`run_gap_lejepa_pipeline.ipynb`](lejepa_cotton_gap/notebooks/run_gap_lejepa_pipeline.ipynb), update its paths and configuration, and run the notebook cells.

## Pipeline overview

1. **Pretraining:** learns GeM- or GAP-pooled backbone/neck features with configurable intra-image or cross-camera pairing and scale weights.
2. **Cotton-boll detection:** fine-tunes and compares LeJEPA, COCO-pretrained, COCO-backbone-only, and randomly initialized YOLOv8 variants; evaluation reports AP50, AP50-95 and AP75.
3. **Plot-status classification:** trains a linear probe on frozen backbone features and reports classification metrics.

The package requires Python 3.9 or newer. The default input size is 640 and must be a multiple of 32. Use absolute paths for dataset and output locations, especially in notebooks.

## Data at a glance

- **Pretraining:** images are searched recursively; the default filename pattern is `clip<n>_cam<k>_frame<m>.jpg`. Images with the same clip and frame identifiers are treated as synchronized camera views.
- **Detection:** images must have YOLO-format text labels (`class xc yc w h`). The package supports common `images/` and `labels/` directory layouts.
- **Plot status:** images are paired with JSON labels. See the package README for accepted annotation formats and label configuration.

See the [package README](lejepa_cotton_gap/README.md) for complete data requirements and examples.
