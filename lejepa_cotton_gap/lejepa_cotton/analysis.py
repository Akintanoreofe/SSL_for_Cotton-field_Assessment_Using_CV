"""Overlap/occlusion analysis, train-vs-eval diagnostics and the LeJEPA-vs-COCO report.

The evaluation split (test when held out, else validation) is the primary
benchmark. The training split is scored only as a diagnostic. Everything is
written inside ``<output_dir>/analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import ImageDraw
from ultralytics import YOLO

from .core_evaluation import (
    DetectionEvalConfig,
    display_variant_name,
    load_training_log,
    match_boxes,
    pairwise_iou,
    read_yolo_labels,
    ultralytics_device,
    validate_detector,
    yolo_to_xyxy,
)
from .core_pretraining import list_images, load_rgb, select_device
from .tuning import TUNING_ARMS, TuningConfig, evaluation_split

METRICS = ("precision", "recall", "f1", "mAP50", "mAP50-95")


@dataclass
class OverlapConfig:
    """Definition of an overlapping / heavily occluded image.

    Parameters
    ----------
    iou_threshold : float, default=0.3
        Two ground-truth boxes overlap when their IoU reaches this value.
    occlusion_threshold : float, default=0.5
        A box is heavily occluded when another box covers at least this
        fraction of its area.
    conf : float, default=0.25
        Confidence threshold used for per-image matching and visualisations.
    match_iou : float, default=0.5
        IoU of a prediction-to-truth match.
    max_examples : int, default=6
        Images in the qualitative figure.
    """

    iou_threshold: float = 0.3
    occlusion_threshold: float = 0.5
    conf: float = 0.25
    match_iou: float = 0.5
    max_examples: int = 6


def occlusion_scores(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-box maximum IoU and maximum covered fraction with respect to the other boxes."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    iou = pairwise_iou(boxes, boxes)
    top_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area = np.clip((boxes[:, 2:] - boxes[:, :2]).prod(axis=1), 1e-9, None)
    covered = inter / area[:, None]
    np.fill_diagonal(iou, 0.0)
    np.fill_diagonal(covered, 0.0)
    return iou.max(axis=1), covered.max(axis=1)


def find_overlap_images(image_dir: Path, label_dir: Path, ocfg: OverlapConfig) -> pd.DataFrame:
    """Flag images holding overlapping or heavily occluded ground-truth boxes."""
    rows = []
    for path in list_images(image_dir):
        width, height = load_rgb(path).size
        boxes = yolo_to_xyxy(read_yolo_labels(Path(label_dir) / f"{path.stem}.txt"), width, height)
        max_iou, covered = occlusion_scores(boxes)
        occluded = (max_iou >= ocfg.iou_threshold) | (covered >= ocfg.occlusion_threshold)
        rows.append({"image": path.name, "n_boxes": len(boxes), "n_overlapping": int(occluded.sum()),
                     "max_iou": float(max_iou.max(initial=0.0)), "max_covered": float(covered.max(initial=0.0)),
                     "overlap": bool(occluded.any())})
    return pd.DataFrame(rows)


def build_overlap_subset(split_dir: Path, split: str, out_dir: Path, ocfg: OverlapConfig) -> Tuple[Path, pd.DataFrame]:
    """Create a dataset (symlinks) holding only the overlap images of ``split``.

    Returns the dataset yaml (the subset is exposed as ``val`` and ``test``)
    and the per-image overlap table of the whole split.
    """
    image_dir, label_dir = split_dir / "images" / split, split_dir / "labels" / split
    table = find_overlap_images(image_dir, label_dir, ocfg)
    chosen = table.loc[table["overlap"], "image"]
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*"), reverse=True):
            path.unlink() if path.is_file() or path.is_symlink() else path.rmdir()
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for name in chosen:
        (out_dir / "images" / "val" / name).symlink_to((image_dir / name).resolve())
        label = label_dir / f"{Path(name).stem}.txt"
        if label.exists():
            (out_dir / "labels" / "val" / label.name).symlink_to(label.resolve())
    base = yaml.safe_load((split_dir / "dataset.yaml").read_text())
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()), "train": "images/val", "val": "images/val", "test": "images/val",
        "names": base["names"]}))
    return yaml_path, table


def predict_images(weights: Path, image_dir: Path, label_dir: Path, names: Sequence[str],
                   ocfg: OverlapConfig, imgsz: int, device: str) -> Dict[str, dict]:
    """Match predictions to truth per image; flag which truth boxes are occluded."""
    model = YOLO(str(weights))
    out = {}
    for name in names:
        path = Path(image_dir) / name
        width, height = load_rgb(path).size
        labels = read_yolo_labels(Path(label_dir) / f"{path.stem}.txt")
        truth = yolo_to_xyxy(labels, width, height)
        max_iou, covered = occlusion_scores(truth)
        occluded = (max_iou >= ocfg.iou_threshold) | (covered >= ocfg.occlusion_threshold)
        boxes = model.predict(str(path), imgsz=imgsz, conf=ocfg.conf, device=device, verbose=False)[0].boxes
        pred = boxes.xyxy.cpu().numpy().reshape(-1, 4)
        records = match_boxes(truth, labels[:, 0].astype(int), pred, boxes.cls.cpu().numpy().astype(int),
                              boxes.conf.cpu().numpy(), ocfg.match_iou)
        tp = sum(r["status"] == "TP" for r in records)
        fp = sum(r["status"] == "FP" for r in records)
        fn = sum(r["status"] == "FN" for r in records)
        found = {(r["gt_x1"], r["gt_y1"]) for r in records if r["status"] == "TP"}
        hit = np.array([(float(b[0]), float(b[1])) in found for b in truth], dtype=bool)
        out[name] = {"tp": tp, "fp": fp, "fn": fn, "truth": truth, "pred": pred,
                     "scores": boxes.conf.cpu().numpy(), "occluded": occluded, "hit": hit}
    return out


def image_scores(record: dict) -> Dict[str, float]:
    """Precision, recall and F1 of one image at the fixed confidence threshold."""
    tp, fp, fn = record["tp"], record["fp"], record["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def save_overlap_examples(predictions: Dict[str, Dict[str, dict]], image_dir: Path, table: pd.DataFrame,
                          out_path: Path, ocfg: OverlapConfig) -> Optional[Path]:
    """Figure: one row per overlap image, one column per model, with per-image metrics.

    Truth boxes are green, predictions red. Images with the most overlapping
    boxes are shown first.
    """
    arms = list(predictions)
    chosen = table[table["overlap"]].sort_values(["n_overlapping", "max_covered"], ascending=False)
    chosen = chosen["image"].head(ocfg.max_examples).tolist()
    if not chosen:
        return None
    fig, axes = plt.subplots(len(chosen), len(arms), figsize=(5.5 * len(arms), 4.2 * len(chosen)), squeeze=False)
    for row, name in enumerate(chosen):
        for col, arm in enumerate(arms):
            record = predictions[arm][name]
            image = load_rgb(Path(image_dir) / name)
            draw = ImageDraw.Draw(image)
            for box in record["truth"]:
                draw.rectangle(box.tolist(), outline=(0, 200, 0), width=2)
            for box, score in zip(record["pred"], record["scores"]):
                draw.rectangle(box.tolist(), outline=(220, 0, 0), width=2)
                draw.text((box[0], box[3] + 2), f"{score:.2f}", fill=(220, 0, 0))
            s = image_scores(record)
            ax = axes[row, col]
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"{display_variant_name(TUNING_ARMS[arm])} | {name}\n"
                         f"P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f} "
                         f"(TP {record['tp']}, FP {record['fp']}, FN {record['fn']})", fontsize=8)
    fig.suptitle("Overlapping / occluded boxes: truth (green) vs prediction (red)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _final_weights(cfg: DetectionEvalConfig, arm: str, seed: int) -> Path:
    return cfg.runs_dir / f"final_{arm}_seed{seed}" / "weights" / "best.pt"


def _markdown(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown()
    except ImportError:
        return "```\n" + table.to_string() + "\n```"


def overlap_comparison(cfg: DetectionEvalConfig, tcfg: TuningConfig = TuningConfig(),
                       ocfg: OverlapConfig = OverlapConfig()) -> Dict[str, object]:
    """Compare LeJEPA and COCO backbones on overlapping/occluded evaluation images.

    Returns
    -------
    dict
        ``metrics`` (one row per arm with precision, recall, F1, mAP50 and
        mAP50-95 on the overlap subset), ``image_table``, ``occlusion``
        (recall on occluded vs unoccluded boxes), ``wins`` (per-image F1
        comparison), ``figure`` and ``text`` (the printed section).
    """
    split = evaluation_split(cfg)
    device = ultralytics_device(select_device(cfg.device))
    folder = cfg.output_dir / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    data_yaml, table = build_overlap_subset(cfg.split_dir, split, folder / "overlap_subset", ocfg)
    table.to_csv(folder / "overlap_images.csv", index=False)
    names = table.loc[table["overlap"], "image"].tolist()
    if not names:
        text = f"No overlapping or occluded images in the {split} split; comparison skipped."
        print(text)
        return {"metrics": pd.DataFrame(), "image_table": table, "text": text}
    seed = tcfg.seeds[0]
    image_dir, label_dir = cfg.split_dir / "images" / split, cfg.split_dir / "labels" / split
    rows, predictions, occlusion = [], {}, []
    for arm in TUNING_ARMS:
        weights = _final_weights(cfg, arm, seed)
        scores = validate_detector(weights, data_yaml, cfg, f"overlap_{arm}", device, split="val")
        rows.append({"arm": arm, "images": len(names), **{m: scores[m] for m in METRICS}})
        predictions[arm] = predict_images(weights, image_dir, label_dir, names, ocfg, cfg.image_size, device)
        hit = np.concatenate([p["hit"] for p in predictions[arm].values()])
        occ = np.concatenate([p["occluded"] for p in predictions[arm].values()])
        occlusion.append({"arm": arm, "occluded_boxes": int(occ.sum()),
                          "recall_occluded": float(hit[occ].mean()) if occ.any() else np.nan,
                          "free_boxes": int((~occ).sum()),
                          "recall_unoccluded": float(hit[~occ].mean()) if (~occ).any() else np.nan})
    metrics = pd.DataFrame(rows).set_index("arm")
    occ_table = pd.DataFrame(occlusion).set_index("arm")
    per_image = pd.DataFrame({arm: {n: image_scores(r)["f1"] for n, r in p.items()}
                              for arm, p in predictions.items()})
    diff = per_image["lejepa_init"] - per_image["coco_init"]
    wins = {"lejepa_better": int((diff > 1e-9).sum()), "coco_better": int((diff < -1e-9).sum()),
            "tie": int((diff.abs() <= 1e-9).sum())}
    metrics.to_csv(folder / "overlap_metrics.csv")
    occ_table.to_csv(folder / "overlap_occlusion_recall.csv")
    per_image.to_csv(folder / "overlap_per_image_f1.csv")
    figure = save_overlap_examples(predictions, image_dir, table, folder / "overlap_examples.png", ocfg)
    text = _overlap_text(metrics, occ_table, wins, ocfg, split, figure, len(table))
    (folder / "overlap_comparison.md").write_text(text)
    print(text)
    return {"metrics": metrics, "image_table": table, "occlusion": occ_table, "wins": wins,
            "figure": figure, "text": text}


def _overlap_text(metrics: pd.DataFrame, occ: pd.DataFrame, wins: Dict[str, int], ocfg: OverlapConfig,
                  split: str, figure: Optional[Path], n_total: int) -> str:
    lejepa, coco = metrics.loc["lejepa_init"], metrics.loc["coco_init"]
    shown = metrics[list(METRICS)].rename(index={a: display_variant_name(v) for a, v in TUNING_ARMS.items()})
    delta = (lejepa[list(METRICS)] - coco[list(METRICS)]).astype(float)
    better = [m for m in METRICS if delta[m] > 0.005]
    worse = [m for m in METRICS if delta[m] < -0.005]
    lines = [
        "## LeJEPA vs COCO pretraining on overlapping / occluded objects",
        "",
        f"Evaluation set: **{split}** split. Overlap images: {int(metrics['images'].iloc[0])} of {n_total} "
        f"(a box pair has IoU >= {ocfg.iou_threshold} or a box is >= {ocfg.occlusion_threshold:.0%} covered).",
        "",
        _markdown(shown.round(4)),
        "",
        "Difference (LeJEPA - COCO): " + ", ".join(f"{m} {delta[m]:+.4f}" for m in METRICS),
        "",
        f"Recall at conf {ocfg.conf} on occluded vs unoccluded truth boxes of these images:",
        "",
        _markdown(occ.rename(index={a: display_variant_name(v) for a, v in TUNING_ARMS.items()}).round(4)),
        "",
        f"Per-image F1: LeJEPA better on {wins['lejepa_better']}, COCO better on {wins['coco_better']}, "
        f"tied on {wins['tie']} images.",
        "",
        "### Summary",
        f"- LeJEPA improves: {', '.join(better) if better else 'none of the metrics'}.",
        f"- LeJEPA does not improve (worse than COCO): {', '.join(worse) if worse else 'none of the metrics'}.",
    ]
    occ_delta = occ.loc["lejepa_init", "recall_occluded"] - occ.loc["coco_init", "recall_occluded"]
    if np.isfinite(occ_delta):
        verdict = "higher" if occ_delta > 0.005 else "lower" if occ_delta < -0.005 else "similar"
        lines.append(f"- Recall on occluded boxes is {verdict} for LeJEPA ({occ_delta:+.4f}).")
    lines.append("- Differences below 0.005 are treated as ties; with one seed and a small subset they "
                 "may fall within noise.")
    if figure is not None:
        lines += ["", f"Qualitative examples with per-image metrics: `{figure}`"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Training-set diagnostics
# --------------------------------------------------------------------------- #
def label_quality(split_dir: Path, split: str) -> Dict[str, float]:
    """Simple annotation statistics of a split."""
    image_dir, label_dir = split_dir / "images" / split, split_dir / "labels" / split
    n_images = empty = tiny = duplicates = out_of_bounds = total = 0
    areas: List[float] = []
    per_image: List[int] = []
    for path in list_images(image_dir):
        n_images += 1
        labels = read_yolo_labels(label_dir / f"{path.stem}.txt")
        if not len(labels):
            empty += 1
            per_image.append(0)
            continue
        per_image.append(len(labels))
        total += len(labels)
        cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
        areas.extend((w * h).tolist())
        tiny += int(((w * 640 < 4) | (h * 640 < 4)).sum())
        out_of_bounds += int(((cx - w / 2 < -0.01) | (cy - h / 2 < -0.01) |
                              (cx + w / 2 > 1.01) | (cy + h / 2 > 1.01)).sum())
        boxes = yolo_to_xyxy(labels, 1, 1)
        if len(boxes) > 1:
            iou = pairwise_iou(boxes, boxes)
            np.fill_diagonal(iou, 0.0)
            duplicates += int((np.triu(iou) > 0.95).sum())
    return {"images": n_images, "boxes": total, "boxes_per_image": total / max(n_images, 1),
            "empty_label_images": empty, "median_box_area": float(np.median(areas)) if areas else 0.0,
            "tiny_boxes": tiny, "near_duplicate_boxes": duplicates, "out_of_bounds_boxes": out_of_bounds}


def confident_false_positive_rate(weights: Path, split_dir: Path, split: str, imgsz: int, device: str,
                                  conf: float = 0.7, max_images: int = 100) -> float:
    """Confident unmatched predictions per truth box; high values hint at missing annotations."""
    image_dir, label_dir = split_dir / "images" / split, split_dir / "labels" / split
    model = YOLO(str(weights))
    fp = truth_total = 0
    for path in list_images(image_dir)[:max_images]:
        width, height = load_rgb(path).size
        labels = read_yolo_labels(label_dir / f"{path.stem}.txt")
        boxes = model.predict(str(path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0].boxes
        records = match_boxes(yolo_to_xyxy(labels, width, height), labels[:, 0].astype(int),
                              boxes.xyxy.cpu().numpy().reshape(-1, 4), boxes.cls.cpu().numpy().astype(int),
                              boxes.conf.cpu().numpy())
        fp += sum(r["status"] == "FP" for r in records)
        truth_total += len(labels)
    return fp / max(truth_total, 1)


def curve_summary(run_dir: Path) -> Dict[str, float]:
    """Loss and mAP trends of a training log."""
    log = load_training_log(run_dir)
    train_loss = log[[c for c in log.columns if c.startswith("train/")]].sum(axis=1)
    val_loss = log[[c for c in log.columns if c.startswith("val/")]].sum(axis=1)
    k = max(2, len(log) // 5)
    map_col = "metrics/mAP50-95(B)"
    return {"epochs": len(log), "train_loss_first": float(train_loss.iloc[0]), "train_loss_last": float(train_loss.iloc[-1]),
            "train_loss_tail_drop": float(train_loss.iloc[:k].mean() - train_loss.iloc[-k:].mean()) if len(log) > k else 0.0,
            "last_epoch_train_drop": float(train_loss.iloc[-k] - train_loss.iloc[-1]),
            "val_loss_min_epoch": int(val_loss.idxmin()) + 1, "val_loss_last": float(val_loss.iloc[-1]),
            "val_loss_min": float(val_loss.min()), "best_map_epoch": int(log[map_col].idxmax()) + 1,
            "best_val_map": float(log[map_col].max())}


def diagnose(row_train: pd.Series, row_eval: pd.Series, curves: Dict[str, float], quality: Dict[str, float],
             confident_fp: float) -> List[str]:
    """Rule-based reading of the train-vs-eval metrics; thresholds are heuristics."""
    findings = []
    train_map, eval_map = float(row_train["mAP50-95"]), float(row_eval["mAP50-95"])
    gap = train_map - eval_map
    if train_map < 0.3:
        findings.append(f"UNDERFITTING: training mAP50-95 is only {train_map:.3f}; the model does not even fit "
                        "the data it sees. Train longer, raise capacity/resolution or revisit the learning rate.")
    if gap > 0.15 and train_map >= 0.3:
        findings.append(f"OVERFITTING: training mAP50-95 {train_map:.3f} exceeds evaluation {eval_map:.3f} by "
                        f"{gap:.3f}. Use more data/augmentation or stronger regularisation.")
    if curves["val_loss_last"] > 1.15 * curves["val_loss_min"] and curves["val_loss_min_epoch"] < curves["epochs"] - 3:
        findings.append(f"Validation loss bottomed at epoch {curves['val_loss_min_epoch']} then rose "
                        f"({curves['val_loss_min']:.3f} -> {curves['val_loss_last']:.3f}): overfitting signal.")
    if curves["last_epoch_train_drop"] > 0.05 * max(curves["train_loss_last"], 1e-9):
        findings.append("OPTIMIZATION: training loss is still falling at the end of training, so the run is "
                        "under-trained (30 epochs may be too few) or the schedule ends too early.")
    if train_map < 0.3 and curves["train_loss_last"] > 0.8 * curves["train_loss_first"]:
        findings.append("OPTIMIZATION: training loss barely decreased; check learning rate, optimizer and "
                        "backbone freezing.")
    if confident_fp > 0.3:
        findings.append(f"ANNOTATION QUALITY: {confident_fp:.2f} confident unmatched predictions per truth box "
                        "on training images; objects may be unlabelled or boxes misplaced. Inspect them.")
    issues = quality["near_duplicate_boxes"] + quality["out_of_bounds_boxes"] + quality["tiny_boxes"]
    if issues > 0.02 * max(quality["boxes"], 1) or quality["empty_label_images"] > 0.1 * max(quality["images"], 1):
        findings.append(f"DATA QUALITY: {quality['near_duplicate_boxes']:.0f} duplicate, "
                        f"{quality['out_of_bounds_boxes']:.0f} out-of-bounds and {quality['tiny_boxes']:.0f} tiny boxes; "
                        f"{quality['empty_label_images']:.0f} images without labels.")
    if train_map >= 0.3 and gap <= 0.15 and eval_map < 0.3:
        findings.append("Train and evaluation are both low and close: the limit is the task/data (small, dense, "
                        "occluded bolls) or annotation noise rather than a train/eval mismatch.")
    return findings or ["No dominant failure mode detected by the heuristics."]


def training_diagnostics(cfg: DetectionEvalConfig, runs: pd.DataFrame,
                         tcfg: TuningConfig = TuningConfig()) -> str:
    """Compare training-set and evaluation-set metrics per arm and print a diagnosis.

    ``runs`` is the output of :func:`lejepa_cotton.tuning.retrain_best`. The
    evaluation rows are the primary benchmark; training rows are diagnostic.
    """
    device = ultralytics_device(select_device(cfg.device))
    folder = cfg.output_dir / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    seed = tcfg.seeds[0]
    split = evaluation_split(cfg)
    metrics = runs[runs["seed"] == seed].set_index(["arm", "split"])[list(METRICS)]
    metrics.to_csv(folder / "train_vs_eval_metrics.csv")
    lines = ["## Training-set diagnostic (training split vs evaluation split)", "",
             f"Primary benchmark: **{split}** split. Training-split metrics are diagnostic only.", "",
             _markdown(metrics.rename(index={a: display_variant_name(v) for a, v in TUNING_ARMS.items()},
                                      level=0).round(4)), ""]
    quality = {s: label_quality(cfg.split_dir, s) for s in ("train", split)}
    quality_table = pd.DataFrame(quality)
    quality_table.to_csv(folder / "label_quality.csv")
    lines += ["Annotation statistics:", "", _markdown(quality_table.round(4)), ""]
    for arm, variant in TUNING_ARMS.items():
        weights = _final_weights(cfg, arm, seed)
        curves = curve_summary(weights.parents[1])
        confident = confident_false_positive_rate(weights, cfg.split_dir, "train", cfg.image_size, device)
        findings = diagnose(metrics.loc[(arm, "train")], metrics.loc[(arm, "eval")], curves,
                            quality["train"], confident)
        lines += [f"### {display_variant_name(variant)}",
                  f"- epochs {curves['epochs']}, best val mAP50-95 {curves['best_val_map']:.3f} at epoch "
                  f"{curves['best_map_epoch']}; train loss {curves['train_loss_first']:.3f} -> "
                  f"{curves['train_loss_last']:.3f}; confident-FP rate on train {confident:.2f}"]
        lines += [f"- {f}" for f in findings]
        lines.append("")
    text = "\n".join(lines)
    (folder / "training_diagnostics.md").write_text(text)
    print(text)
    return text


def run_final_report(cfg: DetectionEvalConfig, runs: pd.DataFrame, tcfg: TuningConfig = TuningConfig(),
                     ocfg: OverlapConfig = OverlapConfig()) -> str:
    """Print and save the evaluation table, overlap comparison and training diagnostics.

    Writes ``<output_dir>/analysis/final_report.md``.
    """
    split = evaluation_split(cfg)
    folder = cfg.output_dir / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    seed = tcfg.seeds[0]
    primary = runs[(runs["split"] == "eval") & (runs["seed"] == seed)].set_index("arm")[list(METRICS)]
    primary = primary.rename(index={a: display_variant_name(v) for a, v in TUNING_ARMS.items()})
    sections = [f"## Primary benchmark: {split} split", "",
                _markdown(primary.round(4)), ""]
    print("\n".join(sections))
    overlap = overlap_comparison(cfg, tcfg, ocfg)
    diagnostics = training_diagnostics(cfg, runs, tcfg)
    report = "\n".join(sections + [overlap["text"], "", diagnostics])
    path = folder / "final_report.md"
    path.write_text(report)
    print(f"Report saved to {path}")
    return report
