"""Tuning-time NMS controls for Ultralytics validation (timeouts, candidate cap, CPU NMS on MPS).

Ultralytics' validator only exposes ``conf``, ``iou`` and ``max_det``; ``max_nms`` and
``max_time_img`` can only be changed by wrapping ``ultralytics.utils.nms.non_max_suppression``.
The patch is scoped by a context manager so final (paper) evaluation is never affected.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils import nms as ul_nms

from .core_evaluation import read_yolo_labels

NMS_TIMEOUT_MESSAGE = "NMS time limit"


class _TimeoutCounter(logging.Handler):
    """Counts Ultralytics 'NMS time limit exceeded' warnings (each one truncates a batch)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if NMS_TIMEOUT_MESSAGE in record.getMessage():
            self.count += 1


@contextmanager
def tuning_nms(max_nms: int, max_time_img: float, on_cpu: bool = False) -> Iterator[_TimeoutCounter]:
    """Patch NMS defaults inside the block and yield a counter of NMS timeouts.

    Parameters
    ----------
    max_nms : int
        Candidate boxes (by confidence) passed to NMS per image.
    max_time_img : float
        Seconds per image; Ultralytics' limit is ``2 + max_time_img * batch``. Set it so large
        that truncation never happens during tuning.
    on_cpu : bool, default=False
        Run NMS on CPU (one transfer per batch) instead of per-image ops on MPS.
    """
    original = ul_nms.non_max_suppression
    counter = _TimeoutCounter()

    def patched(prediction, *args, **kwargs):
        kwargs.setdefault("max_nms", max_nms)
        kwargs.setdefault("max_time_img", max_time_img)
        device = prediction[0].device if isinstance(prediction, (list, tuple)) else prediction.device
        if not (on_cpu and device.type == "mps"):
            return original(prediction, *args, **kwargs)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        out = original(prediction.cpu(), *args, **kwargs)
        return [o.to(device) for o in out]

    ul_nms.non_max_suppression = patched
    LOGGER.addHandler(counter)
    try:
        yield counter
    finally:
        ul_nms.non_max_suppression = original
        LOGGER.removeHandler(counter)


def max_boxes_per_image(label_dir: Path) -> int:
    """Largest number of ground-truth boxes in any single label file (prints it)."""
    counts: List[int] = [len(read_yolo_labels(p)) for p in sorted(Path(label_dir).glob("*.txt"))]
    worst = int(max(counts, default=0))
    print(f"{label_dir}: {len(counts)} label files, max {worst} GT boxes/image, "
          f"mean {np.mean(counts) if counts else 0:.1f}")
    return worst
