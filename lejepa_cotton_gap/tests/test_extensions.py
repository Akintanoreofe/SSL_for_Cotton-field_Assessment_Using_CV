import csv

import cv2
import numpy as np
import pytest
import torch
from ultralytics import YOLO

from lejepa_cotton import core_pretraining
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


def test_train_one_epoch_averages_metrics_without_per_step_output(monkeypatch, capsys):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, views):
            return {"feature": self.weight * views.mean()}

        def clamp_gem_parameters(self):
            pass

    def fake_lejepa_loss(projections, *_args):
        loss = projections["feature"].square()
        return loss, {"total": loss.detach()}

    monkeypatch.setattr(core_pretraining, "lejepa_loss", fake_lejepa_loss)
    encoder = Encoder()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor([[[1.0]], [[3.0]]]),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        ),
        batch_size=1,
    )
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0)

    stats = core_pretraining.train_one_epoch(
        encoder, loader, None, optimizer, 1, 0.2, torch.device("cpu")
    )

    # The batch losses are (2 * 1)^2 = 4 and (2 * 3)^2 = 36.
    assert stats["total"] == pytest.approx(20.0)
    assert capsys.readouterr().out == ""


def test_train_one_epoch_returns_empty_stats_for_empty_loader():
    encoder = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.empty((0, 1, 1)),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        ),
        batch_size=1,
    )

    stats = core_pretraining.train_one_epoch(
        encoder, loader, None, optimizer, 1, 0.2, torch.device("cpu")
    )

    assert stats == {}


def test_train_one_epoch_enables_step_diagnostics_when_configured(monkeypatch, capsys, tmp_path):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, views):
            return {"feature": self.weight * views.mean()}

        def clamp_gem_parameters(self):
            pass

        def p_values(self):
            return {"P3": 2.0}

    def fake_lejepa_loss(projections, *_args):
        loss = projections["feature"].square()
        return loss, {"total": loss.detach()}

    debug_paths = []
    monkeypatch.setattr(core_pretraining, "lejepa_loss", fake_lejepa_loss)
    monkeypatch.setattr(
        core_pretraining,
        "save_activation_debug_grid",
        lambda _encoder, _views, path: debug_paths.append(path),
    )
    cfg = PretrainConfig(
        image_root=tmp_path,
        output_dir=tmp_path,
        log_every_n_steps=1,
        debug_every_n_steps=1,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor([[[1.0]], [[3.0]]]),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        ),
        batch_size=1,
    )
    encoder = Encoder()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0)

    core_pretraining.train_one_epoch(
        encoder, loader, None, optimizer, 1, 0.2, torch.device("cpu"),
        desc="train", cfg=cfg,
    )

    output = capsys.readouterr().out
    assert "train step=1 P3.p=2.000" in output
    assert "train step=2 P3.p=2.000" in output
    assert len(debug_paths) == 2


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
    assert cfg.log_every_n_steps == cfg.debug_every_n_steps == 0
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
