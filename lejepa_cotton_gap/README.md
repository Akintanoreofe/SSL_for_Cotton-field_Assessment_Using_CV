# lejepa-cotton

Cross-camera **LeJEPA** self-supervised pretraining of a **YOLOv8n** backbone with **global average pooling (GAP)**, evaluated against **COCO** weights on cotton-boll detection and plot-status classification.

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

```bash
cd lejepa_cotton_gap
pip install -e .            # add [notebook] to also install Jupyter
```

Python ≥ 3.9, PyTorch ≥ 2.0, torchvision ≥ 0.16, Ultralytics ≥ 8.1.

## Method

**Data.** Files are named `clip<n>_cam<k>_frame<m>.jpg`. `discover_camera_groups` groups images that share the same `clip` and `frame` and keeps only frames seen by *all* requested cameras. The default cameras are `(1, 2, 4)`, which are the 2nd, 3rd and 5th physical cameras. One training sample is one synchronised frame, and its views are the camera images, each augmented independently. `views_per_camera > 1` adds extra augmentations per camera.

**Encoder.** `YOLOv8MultiScaleBackbone` takes layers 0–9 of `yolov8n.yaml`, so it starts from empty, randomly initialised weights, and returns the P3/P4/P5 feature maps. `YOLOv8GAPEncoder` global-average-pools each scale to a vector and passes it through its own 3-layer MLP projector.

**Objective** (per scale, then averaged):

- *Cross-camera prediction loss*: every view of camera *c* is regressed onto the mean embedding of the **other** cameras of the same frame. The invariance being learned is therefore between viewpoints, not only between augmentations of one image.
- *SIGReg*: a sliced Epps–Pulley test pushes the embedding distribution toward an isotropic Gaussian, which prevents collapse.
- `loss = (1 − λ)·prediction + λ·SIGReg`, with λ = 0.2 by default.

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

- **Pretraining:** a folder (searched recursively) of multi-camera JPEGs. Frames with the same `clip` and `frame` numbers are assumed to be captured at the same moment. Change `filename_pattern` if your naming differs; it must keep the named groups `clip`, `cam` and `frame`.
- **Detection:** `images/` and `labels/` folders in YOLO format, where images with empty label files are skipped. Keep `output_dir` outside `source_dir`.
- **Plot status:** images plus an `annotations.json` of the form `{"file_name.jpg": "in_plot", ...}`.
