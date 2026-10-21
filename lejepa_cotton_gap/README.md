# lejepa-cotton

**LeJEPA** self-supervised pretraining for cotton-boll detection with **YOLOv8n**. The current default uses learnable GeM pooling and pretrains the PAN-FPN neck with a P3-only loss; GAP/backbone-only pretraining remains available for reproducing the legacy pipeline.

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

**Encoder.** The model is randomly initialised from `yolov8n.yaml`. `pretrain_neck=True` manually executes the YOLO graph through the neck and captures the Detect module's P3/P4/P5 inputs; `False` captures the legacy backbone stages. Each selected scale has a separate projector. Pooling can use GAP or GeM (`p` fixed at 3 by default; `learnable_p=True` learns it, clamped to [1, 6]); GeM clamps activations to `1e-6` before exponentiation.

**Objective** (weighted sum across enabled scales):

- *Intra-image prediction loss* (default): each view is regressed toward the mean embedding of the same source image's global views. Optional `prediction_pairing="cross_camera"` retains the previous synchronized-camera objective.
- *SIGReg*: a sliced Epps–Pulley test pushes the embedding distribution toward an isotropic Gaussian, which prevents collapse. It is applied to both the global and local paths.
- *Local path* (`use_local_sigreg=True`): a 1×1-conv projector on P3 gives per-location tokens; SIGReg is applied to `tokens_per_batch` (4096) randomly sampled tokens, weighted by `local_sigreg_weight`. No prediction loss on tokens. `use_local_sigreg=False` disables it.
- `loss = Σ_s scale_weight_s · ((1 − λ)·prediction_s + λ·SIGReg_s) + local_sigreg_weight·SIGReg_local`, with the single weight `SIGREG_LAMBDA = 0.05` (defined once in `core_pretraining`; no sweep).
- Per-step prediction / SIGReg / total losses are written to `step_losses.csv`; `log_every_n_steps` also prints them. Zero-weight scales are skipped entirely; SIGReg is evaluated over the full set of view embeddings at every active scale.

**Pretraining augmentation.** The intra-image path samples two global crops (`global_scale=(0.4, 1.0)`) and applies independent flips and optional rotations. Rotation uses bilinear interpolation and reflect padding (default maximum ±15°, probability 0.5). `local_scale=(0.05, 0.25)` is exposed for crop configuration. No pixel-based salience heuristic or hand-crafted spatial prior is used.

**How the losses are computed.** At each active feature scale, the projector produces embeddings with shape `(batch, views, projection_dim)`. The data loader returns a source-image group id for every view, and the prediction loss asserts groups are not mixed. Intra-image targets are the mean embeddings of an image's global views; the optional cross-camera objective uses the mean from other cameras in the same synchronized frame.

SIGReg projects the batch onto random unit directions, compares the empirical characteristic function with a standard normal, and integrates the squared discrepancy over a fixed range. The reported component losses are detached from autograd; the combined training loss remains differentiable. By default, training shows a per-batch progress bar for each epoch and prints epoch-averaged losses and learned GeM exponents once per epoch (`verbose=True`). Set `verbose=False` to disable progress output. Per-step GeM logging and activation grids are disabled (`log_every_n_steps=0`, `debug_every_n_steps=0`); set either interval to a positive value to opt into that diagnostic. Optional PCA snapshots can be enabled by passing selected zero-based epoch indices as `pca_epochs` to `run_pretraining`; none are generated by default.

**Export.** `transfer="backbone"` exports layers 0–9; `transfer="backbone+neck"` exports layers 0–21. The Detect head remains random. Export reports matched, missing and unexpected keys and writes a real Ultralytics `.pt` checkpoint so fine-tuning does not discard the pretraining weights.

## Evaluation variants

| variant | backbone | neck + head |
|---|---|---|
| `lejepa` | configured LeJEPA transfer (`backbone` by default) | random Detect head; neck random for backbone-only transfer |
| `coco` | COCO `yolov8n.pt` | COCO |
| `coco_backbone` | COCO | random (like-for-like control for `lejepa`) |
| `scratch` | random (`yolov8n.yaml`) | random |

The default comparison is `("lejepa", "coco")`. In detection, `coco` also brings a pretrained neck and head, so add `coco_backbone` when you want a comparison that isolates the backbone.

1. **Detection fine-tuning** (`run_detection_evaluation`). The pipeline builds one reproducible train/val split, fine-tunes every variant with identical hyper-parameters, and reports precision, recall, AP50, AP50-95 and AP75. Validation uses Ultralytics' default low confidence threshold so that AP is computed over the whole precision–recall curve. The `conf` setting only affects the drawn overlays.
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

## Pretraining configuration examples

The dataclasses below expose the same options used in the notebook. The first example reproduces the legacy GAP/backbone/cross-camera behavior; the second is the GeM + neck + P3 default.

```python
from lejepa_cotton import GeMConfig, PretrainConfig, RotationConfig

# Legacy GAP, backbone-only, no rotation.
legacy = PretrainConfig(
    image_root="path/to/images", output_dir="outputs/legacy",
    pooling="gap", pretrain_neck=False,
    scale_weights={"p3": 1 / 3, "p4": 1 / 3, "p5": 1 / 3},
    prediction_pairing="cross_camera", rotation=RotationConfig(enabled=False),
    transfer="backbone",
)

# New default: fixed-p GeM, neck attached, P3-only loss.
modern = PretrainConfig(
    image_root="path/to/images", output_dir="outputs/gem_neck",
    pooling="gem", gem=GeMConfig(p_init=3.0, p_min=1.0, p_max=6.0, learnable_p=False),
    pretrain_neck=True, scale_weights={"p3": 1.0, "p4": 0.0, "p5": 0.0},
    transfer="backbone", prediction_pairing="intra_image",
    rotation=RotationConfig(enabled=True, max_deg=15, p=0.5),
    global_scale=(0.4, 1.0), local_scale=(0.05, 0.25),
)
```

Set `transfer="backbone+neck"` to export the pretrained PAN-FPN layers as well. `DetectionEvalConfig` provides `freeze_backbone_epochs` (default 0) and `backbone_lr_mult` (default 1.0); these settings are applied to every compared initialization. Detection metrics include AP50, AP50-95 and AP75.

For video inference from a fine-tuned detector:

```bash
python -m lejepa_cotton.predict_video --weights outputs/detection/runs/lejepa/weights/best.pt \
  --video path/to/cotton_field.mp4 --out-dir outputs/video
```

## Pretraining options and Optuna tuning

`PretrainConfig` options: `epochs=100`, `image_size=640` (multiple of 32, shared with detection), `pooling="gem"` or `"gap"`, `prediction_pairing="intra_image"` (default) or `"cross_camera"`, `use_local_sigreg`, `local_sigreg_weight`, `tokens_per_batch`, `pool_cameras=(1, 2, 5, 6)`. The legacy `pred_loss_mode` argument remains a compatibility alias.

```python
from lejepa_cotton import (PretrainConfig, DetectionEvalConfig, WeightSources, TuningConfig,
                           run_pretraining, run_tuning, retrain_best, print_best_params,
                           plot_optimization_history, summary_table)

pre = PretrainConfig(image_root="path/to/mars_multi_camera_boll", output_dir="outputs/pretraining")
run_pretraining(pre)

det = DetectionEvalConfig(source_dir="path/to/image_dataset", output_dir="outputs/detection",
                          weights=WeightSources(lejepa_checkpoint=pre.checkpoint_path),
                          val_ratio=0.2, test_ratio=0.2)
tcfg = TuningConfig(n_trials=20)           # 30-epoch proxy trials, same budget for both arms
studies = run_tuning(det, tcfg)            # TPE + median pruner, val mAP50-95; resumable (SQLite)
print_best_params(studies); plot_optimization_history(studies)
runs = retrain_best(det, tcfg, studies)    # best config per arm, 120 epochs x 3 seeds, test split
print(summary_table(runs))                 # mean ± std of test mAP50 / mAP50-95
```

Tuning compares the LeJEPA-init and COCO-backbone-init arms with an identical search space (lr0, lrf, weight_decay, optimizer, warmup_epochs, backbone freeze epochs, backbone LR multiplier, mosaic, close_mosaic, mixup). Studies are stored in `detection/optuna/studies.db`; the test split is only read by `retrain_best`. `run_detection_evaluation` still fine-tunes the variants with default settings and writes `eval/boxes_<variant>.csv`. The notebook `notebooks/run_gap_lejepa_pipeline.ipynb` runs this whole flow.

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