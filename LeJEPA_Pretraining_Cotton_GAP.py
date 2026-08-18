import os
import re
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tqdm
from torchvision.transforms import v2

from sklearn.decomposition import PCA
import plotly.express as px
from ultralytics import YOLO

# DEVICE SETUP (Apple Silicon MPS Acceleration)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f" Device: Apple Silicon GPU via MPS ({DEVICE})")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f" Device: CUDA ({DEVICE})")
else:
    DEVICE = torch.device("cpu")
    print(f"Device: CPU ({DEVICE})")


# CONFIGURATION & HYPERPARAMETERS

JPEG_ROOT = r"/Users/akintanoreofeoluwa/Downloads/LeJEPA _pretrainining_multi_camera_boll/mars_multi_camera_boll"
CHECKPOINT_DIR = r"./checkpoints_cam235_50k_gap_lejepa"

# 0-Indexed Filename Mapping for 2nd, 3rd, and 5th Physical Cameras:
TARGET_CAMERAS_0INDEX = {1, 2, 4}

TARGET_DATASET_SIZE = 50000
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 60
V = 2                          # Number of augmented views generated per sample

PROJ_DIM = 128                 # Global vector projection channel width
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Regularization balancing parameter.
LAMBDA = 0.2

# macOS DataLoader stability setting
NUM_WORKERS = 0
PIN_MEMORY = False

# Log 3D PCA embeddings at these epochs
PCA_EPOCHS = {0, 29, EPOCHS - 1}



# SIGREG REGULARIZATION MODULE 

class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # Expects (V, N, D): V views, N independent batch samples, D channels.
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        Vv, N, D = proj.shape
        A = torch.randn(D, 256, device=proj.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-12)

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * N
        return statistic.mean()


# DATASET MODULE (0-INDEXED CAMERA 50k SAMPLER)

class Cotton50kSamplerDataset(Dataset):
    """
    Parses MARS robot dataset filename format: clip<num>_cam<num>_frame<num>.jpg
    Handles 0-indexed camera tags (0-5) and samples ~50,000 images equally across 3 camera views.
    """
    def __init__(self, root, target_cameras={1, 2, 4}, target_size=50000, seed=42):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"JPEG_ROOT path not found: {root}")

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        all_image_paths = [Path(p) for p in root.rglob("*") if p.suffix.lower() in valid_exts]

        camera_buckets = {cam_id: [] for cam_id in target_cameras}

        for path in all_image_paths:
            match = re.search(r"_cam([0-5])_", path.name.lower())
            if not match:
                match = re.search(r"cam(?:era)?([0-5])", str(path).lower())
            if match:
                cam_num = int(match.group(1))
                if cam_num in target_cameras:
                    camera_buckets[cam_num].append(str(path))

        total_matched = sum(len(paths) for paths in camera_buckets.values())
        print(f"\n🔍 Discovered {total_matched} total images matching Camera IDs {sorted(list(target_cameras))}:")
        
        if total_matched == 0:
            raise RuntimeError(f"No matching camera images found in {root}")

        rng = np.random.default_rng(seed)
        sampled_paths = []

        if total_matched <= target_size:
            for paths in camera_buckets.values():
                sampled_paths.extend(paths)
        else:
            per_camera_target = target_size // len(target_cameras)
            for cam_id, paths in camera_buckets.items():
                if len(paths) >= per_camera_target:
                    chosen = rng.choice(paths, size=per_camera_target, replace=False).tolist()
                else:
                    chosen = paths
                sampled_paths.extend(chosen)

            remaining_slots = target_size - len(sampled_paths)
            if remaining_slots > 0:
                all_matched = sum(camera_buckets.values(), [])
                unused = list(set(all_matched) - set(sampled_paths))
                extra = rng.choice(unused, size=min(remaining_slots, len(unused)), replace=False).tolist()
                sampled_paths.extend(extra)

        self.paths = sampled_paths
        print(f"Sampled balanced dataset: {len(self.paths)} images for optimized pre-training.\n")

        # GEOMETRIC AUGMENTATION PIPELINE (Updated)
        self.aug = v2.Compose([
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.1),
            v2.RandomRotation(25),
            v2.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15)
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        views = [self.aug(img) for _ in range(V)]
        return torch.stack(views, dim=0)


# MULTI-SCALE YOLOv8 BACKBONE

class YOLOv8MultiScaleBackbone(nn.Module):
    """
    Wraps the first 10 layers of a YOLOv8 model and captures intermediate 
    feature maps at the standard P3, P4, and P5 tap points.
    """
    CAPTURE_INDICES = (4, 6, 9)
    SCALE_NAMES = {4: "P3", 6: "P4", 9: "P5"}

    def __init__(self, weights="yolov8n.pt"):
        super().__init__()
        yolo_model = YOLO(weights).model
        self.layers = nn.ModuleList(list(yolo_model.model[:10]))

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.CAPTURE_INDICES:
                feats[self.SCALE_NAMES[i]] = x
        return feats  # {"P3": ..., "P4": ..., "P5": ...}



# GAP PROJECTOR (MLP for flattened global vectors)

def build_gap_projector(in_channels, proj_dim):
    return nn.Sequential(
        nn.Linear(in_channels, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, proj_dim)
    )



# GAP ENCODER — YOLOv8 Multi-Scale + Global Average Pooling

class YOLOv8DenseEncoder(nn.Module):
    # Changed default weights to .yaml to ensure random initialization
    def __init__(self, weights="yolov8n.yaml", proj_dim=128):
        super().__init__()
        self.backbone = YOLOv8MultiScaleBackbone(weights)

        with torch.no_grad():
            # Ensure IMAGE_SIZE is defined in your scope, e.g., 224 or 512
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            feats = self.backbone(dummy)

        self.projectors = nn.ModuleDict()
        for name, f in feats.items():
            c_in = f.shape[1]
            self.projectors[name] = build_dense_projector(c_in, proj_dim)
            print(f"{name} | channels: {c_in} -> {proj_dim} | spatial: {tuple(f.shape[-2:])}")

    def forward(self, x):
        # x: (B, V, C, H, W)
        B, Vv = x.shape[:2]
        x = x.flatten(0, 1)  # (B*V, C, H, W)

        feats = self.backbone(x)

        dense_projections = {}
        for name, f in feats.items():
            p = self.projectors[name](f)             # (B*V, proj_dim, Hs, Ws)
            _, D, Hs, Ws = p.shape
            dense_projections[name] = p.view(B, Vv, D, Hs, Ws)

        return dense_projections



# GLOBAL LeJEPA LOSSES

def lejepa_prediction_loss(proj: torch.Tensor) -> torch.Tensor:
    """
    Computes difference between the embedding created from original image 
    and augmented version (using mean view embedding as target).
    proj shape: (B, V, D)
    """
    mu = proj.mean(dim=1, keepdim=True)
    dif = mu - proj
    return dif.square().mean()


def compute_multiscale_gap_losses(gap_projections, sigreg):
    pred_by_scale, sig_by_scale = {}, {}
    
    for name, proj in gap_projections.items():
        # proj is shape: (B, V, D)
        pred_by_scale[name] = lejepa_prediction_loss(proj)
        
        # SIGReg expects (V, Batch, D)
        sig_input = proj.transpose(0, 1).contiguous()
        sig_by_scale[name] = sigreg(sig_input)

    pred_loss = torch.stack(list(pred_by_scale.values())).mean()
    sig_loss = torch.stack(list(sig_by_scale.values())).mean()
    
    return pred_loss, sig_loss, pred_by_scale, sig_by_scale



# 3D PCA EMBEDDING LOGGING

@torch.no_grad()
def save_interactive_pca(net, loader, epoch, out_dir: Path, max_batches=60):
    net.eval()
    feats = []
    batches_seen = 0

    for vs in loader:
        vs = vs.to(DEVICE)
        gap_projections = net(vs)

        pooled_scales = []
        for name, proj in gap_projections.items():
            # proj is already GAP pooled: (B, V, D). 
            # We log the mean across views to get 1 representative vector per image (B, D)
            avg_view = proj.mean(dim=1) 
            pooled_scales.append(avg_view)

        # Concat P3, P4, P5 into one large vector per image
        combined = torch.cat(pooled_scales, dim=-1)
        feats.append(combined.cpu().numpy())
        batches_seen += 1
        if batches_seen >= max_batches:
            break

    X = np.concatenate(feats, axis=0)

    print(f"\n--- [EMBEDDINGS INSPECTION - EPOCH {epoch + 1}] ---")
    print(f"Embeddings Matrix Shape: {X.shape}")
    print(f"Embeddings Mean: {X.mean():.4f} | Std Dev: {X.std():.4f}")

    Z = PCA(n_components=3, random_state=0).fit_transform(X)

    fig = px.scatter_3d(
        x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
        opacity=0.7,
        title=f"YOLOv8 GAP LeJEPA (Cam 2,3,5) — Epoch {epoch + 1}",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pca_epoch_{epoch+1:03d}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f" Saved 3D PCA visualization: {out_path}")


def save_loss_history(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    out_path = out_dir / "loss_history_50k_cam235_gap.csv"
    df.to_csv(out_path, index=False)
    print(f" Saved loss history CSV: {out_path}")



# BACKBONE EXPORT

def load_pretrained_backbone_into_yolo(checkpoint_path, yolo_weights="yolov8n.pt", proj_dim=PROJ_DIM):
    encoder = YOLOv8GAPEncoder(weights=yolo_weights, proj_dim=proj_dim)
    state = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(state)

    yolo = YOLO(yolo_weights)
    for i in range(10):
        yolo.model.model[i].load_state_dict(encoder.backbone.layers[i].state_dict())

    print(f"Loaded GAP-pretrained backbone (layers 0-9) from {checkpoint_path} into fresh YOLO model.")
    return yolo



# MAIN EXECUTION LOOP

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_dir = Path(CHECKPOINT_DIR)
    plots_dir = ckpt_dir / "interactive_plots"
    pca_dir = plots_dir / "pca_3d"

    torch.manual_seed(42)

    dataset = Cotton50kSamplerDataset(
        root=JPEG_ROOT,
        target_cameras=TARGET_CAMERAS_0INDEX,
        target_size=TARGET_DATASET_SIZE,
        seed=42
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    net = YOLOv8GAPEncoder(weights="yolov8n.pt", proj_dim=PROJ_DIM).to(DEVICE)
    sigreg = SIGReg().to(DEVICE)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scale_names = list(YOLOv8MultiScaleBackbone.SCALE_NAMES.values())
    history = {"lejepa_total": [], "pred_avg": [], "sigreg_avg": []}
    for name in scale_names:
        history[f"pred_{name}"] = []
        history[f"sigreg_{name}"] = []

    for epoch in range(EPOCHS):
        net.train()
        ep_total, ep_pred, ep_sig, n_batches = 0.0, 0.0, 0.0, 0
        ep_pred_scale = {name: 0.0 for name in scale_names}
        ep_sig_scale = {name: 0.0 for name in scale_names}

        pbar = tqdm.tqdm(loader, desc=f"Pre-training Epoch {epoch+1}/{EPOCHS}")
        for vs in pbar:
            vs = vs.to(DEVICE)

            gap_proj = net(vs)
            pred_loss, sig_loss, pred_by_scale, sig_by_scale = compute_multiscale_gap_losses(gap_proj, sigreg)

            loss = (1.0 - LAMBDA) * pred_loss + LAMBDA * sig_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_total += float(loss.detach().cpu())
            ep_pred += float(pred_loss.detach().cpu())
            ep_sig += float(sig_loss.detach().cpu())
            for name in scale_names:
                ep_pred_scale[name] += float(pred_by_scale[name].detach().cpu())
                ep_sig_scale[name] += float(sig_by_scale[name].detach().cpu())
            n_batches += 1

        ep_total /= max(1, n_batches)
        ep_pred /= max(1, n_batches)
        ep_sig /= max(1, n_batches)
        for name in scale_names:
            ep_pred_scale[name] /= max(1, n_batches)
            ep_sig_scale[name] /= max(1, n_batches)

        history["lejepa_total"].append(ep_total)
        history["pred_avg"].append(ep_pred)
        history["sigreg_avg"].append(ep_sig)
        for name in scale_names:
            history[f"pred_{name}"].append(ep_pred_scale[name])
            history[f"sigreg_{name}"].append(ep_sig_scale[name])

        scale_summary = " | ".join(f"{n}: pred={ep_pred_scale[n]:.5f} sig={ep_sig_scale[n]:.5f}" for n in scale_names)
        print(
            f"Epoch {epoch+1:03d} | Total: {ep_total:.6f} | Pred(avg): {ep_pred:.6f} | "
            f"SIGReg(avg): {ep_sig:.6f} | {scale_summary}"
        )

        if epoch in PCA_EPOCHS:
            save_interactive_pca(net, loader, epoch, pca_dir)

    ckpt_path = ckpt_dir / "gap_lejepa_yolov8_cotton_50k_checkpoint.pth"
    torch.save(net.state_dict(), ckpt_path)
    print(f"Optimized pre-trained checkpoint saved to: {ckpt_path}")

    save_loss_history(history, plots_dir)


if __name__ == "__main__":
    main()