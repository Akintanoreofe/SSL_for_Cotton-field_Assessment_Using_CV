import csv

import cv2
import numpy as np
import pytest
import torch
from ultralytics import YOLO

from lejepa_cotton import (
    GeM,
    PretrainConfig,
    RandomRotationReflect,
    SIGReg,
    YOLOv8GAPEncoder,
    YOLOv8MultiScaleBackbone,
    assert_image_groups,
    cross_camera_prediction_loss,
    export_backbone_to_yolo,
    global_average_pool,
    intra_image_prediction_loss,
    lejepa_loss,
    predict_video,
)


def test_gem_p_one_matches_gap_and_handles_negative_inputs():
    pooling = GeM(p_init=1.0, p_min=1.0, p_max=6.0)
    positive = torch.rand(2, 3, 5, 7)
    torch.testing.assert_close(pooling(positive), global_average_pool(positive))
    assert torch.isfinite(pooling(torch.randn(2, 3, 5, 7))).all()


def test_gem_exponent_is_clamped():
    pooling = GeM(p_init=3.0, p_min=1.0, p_max=6.0)
    with torch.no_grad():
        pooling.p.fill_(100.0)
    pooling(torch.ones(1, 2, 2, 2))
    pooling.clamp_parameter()
    assert 1.0 <= float(pooling.p) <= 6.0


def test_neck_returns_detect_scales_at_expected_strides():
    encoder = YOLOv8MultiScaleBackbone(pretrain_neck=True).eval()
    assert encoder.detect_sources == (15, 18, 21)
    with torch.no_grad():
        features = encoder(torch.zeros(1, 3, 640, 640))
    assert {name: tuple(value.shape[-2:]) for name, value in features.items()} == {
        "P3": (80, 80), "P4": (40, 40), "P5": (20, 20)
    }


def test_gap_backbone_legacy_config_uses_all_scales_and_cross_camera(tmp_path):
    cfg = PretrainConfig(image_root=tmp_path, output_dir=tmp_path, pooling="gap",
                         pretrain_neck=False, rotation={"enabled": False})
    assert cfg.prediction_pairing == "cross_camera"
    assert cfg.scale_weights == {"p3": 1 / 3, "p4": 1 / 3, "p5": 1 / 3}
    torch.manual_seed(7)
    encoder = YOLOv8GAPEncoder(pooling="gap", pretrain_neck=False,
                               scale_weights=cfg.scale_weights).eval()
    views = torch.randn(2, 2, 3, 64, 64)
    with torch.no_grad():
        projections = encoder(views)
        legacy_maps = encoder.backbone(views.flatten(0, 1))
        legacy_pooled = {name: global_average_pool(value) for name, value in legacy_maps.items()}
        expected = {name: encoder.projectors[name](value).reshape(2, 2, -1)
                    for name, value in legacy_pooled.items()}
    for scale in expected:
        torch.testing.assert_close(projections[scale], expected[scale])
    weights = {name.lower(): 1 / 3 for name in projections}
    torch.manual_seed(13)
    actual_loss, _ = lejepa_loss(projections, SIGReg(5, 16), 2, 0.2, "cross_camera", weights)
    torch.manual_seed(13)
    legacy_sigreg = SIGReg(5, 16)
    expected_loss = sum(
        weights[name.lower()] * (0.8 * cross_camera_prediction_loss(z, 2)
                                 + 0.2 * legacy_sigreg(z.transpose(0, 1)))
        for name, z in projections.items()
    )
    torch.testing.assert_close(actual_loss, expected_loss)


def test_exported_backbone_loads_into_yolo(tmp_path):
    backbone = YOLOv8MultiScaleBackbone()
    path = export_backbone_to_yolo(backbone, tmp_path / "backbone.pt")
    detector = YOLO(str(path)).model
    expected = torch.nn.ModuleList(list(backbone.layers[:10])).state_dict()
    actual = torch.nn.ModuleList(list(detector.model[:10])).state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


def test_rotation_shape_and_sampled_angle():
    transform = RandomRotationReflect(max_deg=15, p=1, enabled=True)
    image = torch.rand(3, 40, 50)
    output = transform(image)
    assert output.shape == image.shape
    assert -15 <= transform.last_angle <= 15


def test_prediction_group_assertion_rejects_mixed_images():
    embeddings = torch.randn(2, 2, 4)
    with pytest.raises(AssertionError, match="mix views"):
        intra_image_prediction_loss(embeddings, torch.tensor([[1, 2], [3, 3]]))
    assert_image_groups(torch.tensor([[1, 1], [2, 2]]), 2, 2)


def test_predict_video_writes_video_and_csv(tmp_path):
    source = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 64))
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    weights = tmp_path / "random.pt"
    YOLO("yolov8n.yaml").save(str(weights))
    result = predict_video(weights, source, tmp_path / "out", imgsz=64, frame_stride=2)
    assert result["video_path"].is_file()
    assert result["csv_path"].is_file()
    with result["csv_path"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"frame_idx", "timestamp_s", "n_bolls", "mean_conf"} <= rows[0].keys()
