"""Run YOLO boll detection or tracking on a video and save annotated outputs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO


def predict_video(
    weights: Union[str, Path],
    video_path: Union[str, Path],
    out_dir: Union[str, Path],
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    frame_stride: int = 1,
    track: bool = False,
    device: Optional[str] = None,
) -> dict:
    """Detect or track bolls frame-by-frame and write video plus CSV results."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open or read video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out = out_dir / f"{video_path.stem}_annotated.mp4"
    csv_out = out_dir / f"{video_path.stem}_counts.csv"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"),
                             source_fps / frame_stride, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create output video: {video_out}")

    fields = ["frame_idx", "timestamp_s", "n_bolls", "mean_conf"]
    if track:
        fields.append("unique_tracks")
    counts, track_ids = [], set()
    processed, frame_idx = 0, -1
    start = time.perf_counter()
    try:
        model = YOLO(str(weights))
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            rows = csv.DictWriter(handle, fieldnames=fields)
            rows.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % frame_stride:
                    continue
                kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
                result = (model.track(frame, tracker="bytetrack.yaml", persist=True, **kwargs)[0]
                          if track else model.predict(frame, **kwargs)[0])
                boxes = result.boxes
                confidences = boxes.conf.detach().cpu().numpy() if boxes is not None else np.empty(0)
                n_bolls = int(len(confidences))
                counts.append(n_bolls)
                annotated = result.plot()
                cv2.putText(annotated, f"Bolls: {n_bolls}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)
                writer.write(annotated)
                row = {
                    "frame_idx": frame_idx,
                    "timestamp_s": f"{frame_idx / source_fps:.3f}",
                    "n_bolls": n_bolls,
                    "mean_conf": f"{float(confidences.mean()) if n_bolls else 0.0:.6f}",
                }
                if track:
                    ids = getattr(boxes, "id", None) if boxes is not None else None
                    if ids is not None:
                        track_ids.update(int(item) for item in ids.detach().cpu().tolist())
                    row["unique_tracks"] = len(track_ids)
                rows.writerow(row)
                processed += 1
    finally:
        capture.release()
        writer.release()

    if processed == 0:
        video_out.unlink(missing_ok=True)
        raise ValueError(f"Video contains no readable frames: {video_path}")

    elapsed = max(time.perf_counter() - start, 1e-9)
    summary = {
        "frames_processed": processed,
        "mean_bolls_per_frame": float(np.mean(counts)) if counts else 0.0,
        "max_bolls_per_frame": max(counts, default=0),
        "processing_fps": processed / elapsed,
        "video_path": video_out,
        "csv_path": csv_out,
    }
    if track:
        summary["unique_tracks"] = len(track_ids)
    print(f"Frames processed: {processed} | mean/max bolls: "
          f"{summary['mean_bolls_per_frame']:.2f}/{summary['max_bolls_per_frame']} | "
          f"{'unique tracks: ' + str(len(track_ids)) + ' | ' if track else ''}"
          f"processing FPS: {summary['processing_fps']:.2f}")
    print(f"Annotated video: {video_out}\nCounts CSV: {csv_out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    predict_video(args.weights, args.video, args.out_dir, args.conf, args.iou, args.imgsz,
                  args.frame_stride, args.track, args.device)


if __name__ == "__main__":
    main()
