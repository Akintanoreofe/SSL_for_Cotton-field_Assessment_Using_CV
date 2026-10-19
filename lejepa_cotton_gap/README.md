# lejepa-cotton

Cross-camera **LeJEPA** self-supervised pretraining of a **YOLOv8n** backbone with **global average pooling (GAP)**, evaluated against **COCO** weights on cotton-boll detection and plot-status classification.

For a short repository overview and quick start, see the [root README](../README.md).

## Repository layout

```
lejepa_cotton_gap/
├── setup.py
├── README.md
├── run_pipeline.py                  # script: configs + calls only
├── notebooks/
│   └── run_gap_lejepa_pipeline.ipynb  # notebook: configs + calls only
└── lejepa_cotton/
    ├── __init__.py
    ├── core_pretraining.py          # data, encoder, loss, training, checkpoints
    ├── core_evaluation.py           # detection fine-tuning + linear probe
    ├── visualization.py             # every plot
    └── pipeline.py                  # runners that wire the three modules together
```

Each module has one job, and shared helpers are defined once and imported. For example, `load_rgb`, `list_images` and `build_eval_transform` live in `core_pretraining`. `read_yolo_labels` and `extract_embeddings` live in `core_evaluation`. No core module contains a hard-coded directory: paths only enter through the config dataclasses, which are filled in by the notebook or `run_pipeline.py`.

## Installation

**From GitHub (works in a VS Code "GitHub remote repository" workspace):**

```bash
python -m pip install "git+https://github.com/Akintanoreofe/SSL_for_Cotton-field_Assessment_Using_CV.git#subdirectory=lejepa_cotton_gap"
```

Restart the Jupyter kernel after installing or updating the package.

**From a local clone (recommended while developing):**

```bash
git clone https://github.com/Akintanoreofe/SSL_for_Cotton-field_Assessment_Using_CV.git
cd SSL_for_Cotton-field_Assessment_Using_CV/lejepa_cotton_gap
python -m pip install -e .
```

Python ≥ 3.9, PyTorch ≥ 2.0, torchvision ≥ 0.16, Ultralytics ≥ 8.1.

**Paths:** use absolute paths in the notebook. A Jupyter kernel launched from a virtual workspace starts in `/`, which is read-only on macOS.

## Method

**Data.** Files are named `clip<n>_cam<k>_frame<m>.jpg`. `discover_camera_groups` groups images that share the same `clip` and `frame` and keeps only frames seen by *all* requested cameras. The default cameras are `(1, 2, 4)`, which are the 2nd, 3rd and 5th physical cameras. One training sample is one synchronised frame, and its views are the camera images, each augmented independently. `views_per_camera > 1` adds extra augmentations per camera.

**Encoder.** `YOLOv8MultiScaleBackbone` takes layers 0–9 of `yolov8n.yaml`, so it starts from empty, randomly initialised weights, and returns the P3/P4/P5 feature maps. `YOLOv8GAPEncoder` global-average-pools each scale to a vector and passes it through its own 3-layer MLP projector.

**Objective** (per scale, then averaged):

- *Cross-camera prediction loss*: every view of camera *c* is regressed onto the mean embedding of the **other** cameras of the same frame. The invariance being learned is therefore between viewpoints, not only between augmentations of one image.
- *SIGReg*: a sliced Epps–Pulley test pushes the embedding distribution toward an isotropic Gaussian, which prevents collapse.
- `loss = (1 − λ)·prediction + λ·SIGReg`, with λ = 0.2 by default.

**How the losses are computed.** At each feature scale, the projector produces embeddings with shape `(batch, views, projection_dim)`. With `cross_camera`, views are grouped by camera; each camera's target is the mean embedding of the *other* cameras in the same frame, and the prediction term is the mean squared error from every view to its camera's target. With `intra_image`, the two independently flipped/rotated views of each image are compared directly with mean squared error. Both prediction losses encourage agreement, but across different sources of views.

SIGReg evaluates the embedding distribution separately for each view. It projects the batch onto many random unit directions, compares the empirical characteristic function on those one-dimensional projections with the characteristic function of a standard normal, and integrates the squared discrepancy over a fixed range. This sliced test encourages embeddings to have a spread, isotropic distribution instead of all collapsing to the same value. Prediction and SIGReg are each averaged across P3/P4/P5 before they are combined by λ. The reported component losses are detached from autograd; the combined training loss remains differentiable.

**Export.** `export_backbone_to_yolo` writes a real Ultralytics `.pt` checkpoint whose layers 0–9 are the pretrained backbone. This step matters. `YOLO("yolov8n.yaml").train()` rebuilds the network from scratch, so weights copied into a YAML-built model in memory are silently discarded.

## Evaluation variants

| variant | backbone | neck + head |
|---|---|---|
| `lejepa` | cross-camera GAP LeJEPA | random |
| `coco` | COCO `yolov8n.pt` | COCO |
| `coco_backbone` | COCO | random (like-for-like control for `lejepa`) |
| `scratch` | random (`yolov8n.yaml`) | random |

The default comparison is `("lejepa", "coco")`. In detection, `coco` also brings a pretrained neck and head, so add `coco_backbone` when you want a comparison that isolates the backbone.

1. **Detection fine-tuning** (`run_detection_evaluation`). The pipeline builds one reproducible train/val split, fine-tunes every variant with identical hyper-parameters, and reports precision, recall, mAP50 and mAP50-95. Validation uses Ultralytics' default low confidence threshold so that mAP is computed over the whole precision–recall curve. The `conf` setting only affects the drawn overlays.
2. **Linear probe** (`run_probe_evaluation`). The backbone is frozen, and its BatchNorm layers are kept in eval mode. A linear layer is trained on the concatenated P3/P4/P5 GAP vectors using a stratified split, and the pipeline reports accuracy, macro-F1 and a confusion matrix. Model selection uses the lowest test loss, which mirrors the original protocol. Use a separate validation split if you need an unbiased estimate.

## Usage

**Notebook.** Open `notebooks/run_gap_lejepa_pipeline.ipynb`, edit the *Paths* cell, and run the cells top to bottom.

**Script.** Edit the paths at the top of `run_pipeline.py`, then run:

```bash
python run_pipeline.py
```

**Python API:**

```python
from lejepa_cotton import PretrainConfig, WeightSources, DetectionEvalConfig, run_pretraining, run_detection_evaluation

pre = PretrainConfig(image_root="path/to/mars_multi_camera_boll", output_dir="outputs/pretraining", max_samples=16_667)
ckpt, history = run_pretraining(pre)

det = DetectionEvalConfig(source_dir="path/to/image_dataset", output_dir="outputs/detection",
                          weights=WeightSources(lejepa_checkpoint=ckpt), variants=("lejepa", "coco"))
summary = run_detection_evaluation(det)
```

## Lambda sweep and `intra_image` loss

`PretrainConfig` now carries the sweep and loss-variant options (defaults shown): `lambdas=(0.1, 0.2, 0.3, 0.4, 0.5)`, `epochs=100`, `image_size=640` (multiple of 32, shared with detection), `pred_loss_mode="cross_camera"` or `"intra_image"`, `pool_cameras=(1, 2, 5, 6)`, `max_rotation_deg=15.0` and `overwrite=False`.

- `"cross_camera"` (default, unchanged): each camera view is pulled toward the mean embedding of the other cameras.
- `"intra_image"`: each image from `pool_cameras` yields two views (horizontal flip, vertical flip, rotation within ±`max_rotation_deg`; no crops or colour changes) and the MSE between their multi-scale GAP embeddings is the prediction loss. SIGReg is applied to the same embeddings.

```python
from lejepa_cotton import (PretrainConfig, DetectionEvalConfig, WeightSources,
                           run_lambda_sweep, run_lambda_evaluation, plot_lambda_losses)

pre = PretrainConfig(image_root="path/to/mars_multi_camera_boll", output_dir="outputs/pretraining",
                     pred_loss_mode="intra_image")
checkpoints = run_lambda_sweep(pre)        # outputs/pretraining/checkpoints/backbone_lambda_0.1.pt, ...
plot_lambda_losses(pre)                    # total / prediction / SIGReg loss per lambda

det = DetectionEvalConfig(source_dir="path/to/image_dataset", output_dir="outputs/detection",
                          weights=WeightSources(lejepa_checkpoint=checkpoints[0.1]))
summary = run_lambda_evaluation(det, checkpoints)
```

Per lambda, `checkpoints/` also holds `backbone_lambda_<λ>_config.json` and `_history.csv`. A λ with an existing checkpoint is skipped unless `overwrite=True`. `run_lambda_evaluation` fine-tunes each λ backbone and the COCO backbone (neck/head random for all) and writes `summary_metrics.csv`, `plots/map_vs_lambda.png` (COCO as dashed line), `plots/lambda_vs_coco_bars.png` and, per model, `eval/boxes_lambda_<λ>.csv` / `eval/boxes_coco.csv` (image id, GT and predicted boxes, confidence, class, IoU, status `TP`/`FP`/`FN`/`EMPTY`). The notebook `notebooks/run_gap_lejepa_pipeline.ipynb` runs this whole flow.

## Outputs

```
outputs/
├── pretraining/
│   ├── gap_lejepa_yolov8n.pth         # encoder weights + rebuild args + config
│   ├── loss_history.csv
│   └── plots/{loss_curves.png, pca_3d/pca_epoch_XXX.html}   # PCA coloured by camera
├── detection/
│   ├── split_dataset/  init_weights/  runs/<variant>/
│   ├── detection_summary.csv
│   ├── overlays/<variant>/            # green = GT, red = prediction
│   └── plots/{detection_metrics.png, detection_curves.png}
└── plot_status_probe/
    ├── probe_summary.csv, probe_history_<variant>.csv
    └── plots/{probe_metrics.png, probe_loss_curves.png, confusion_*.png, pca2d_*.png, pca3d_*.html}
```

## Data expectations

- **Pretraining:** a folder (searched recursively) of multi-camera JPEGs named `clip<n>_cam<k>_frame<m>.jpg`. Images with the same `clip` and `frame` are assumed to be synchronised. Change `filename_pattern` if your naming differs; it must keep the named groups `clip`, `cam` and `frame`.
- **Detection:** YOLO text labels (`class xc yc w h`). Any of these layouts works, searched recursively:
  - a folder of sub-datasets, each with `images/` and `labels/` (e.g. `014/images`, `014/labels`, `ssl_active_1/...`);
  - a single `images/` + `labels/` pair;
  - labels next to their images.

  Images without boxes are skipped. Byte-identical images that appear in several sub-datasets are kept once. Split files are renamed `<subfolder>__<name>` so equal file names from different sub-datasets never overwrite each other. Stray list files (`014.txt`) and `.yaml` files in the root are ignored. Set `split_by="folder"` to keep each sub-dataset wholly in train or validation. Keep `output_dir` outside the dataset folder.
- **Plot status:** a folder with the images and a JSON annotation file, found automatically (`annotations.json` if present, otherwise the only `.json`). Accepted shapes: `{"img.jpg": "in_plot"}`, `{"img.jpg": {"label": "in_plot"}}`, or a list of `{"file_name": ..., "label": ...}` records. Labels are matched case-insensitively, with spaces or hyphens treated as underscores.

Run `find_labeled_pairs(DETECTION_DATA)` and `PlotStatusDataset(PLOT_STATUS_DIR, None, label_mapping, transform=None)` to check both datasets in seconds before training.