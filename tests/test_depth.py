import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.data import demo_scene
import app.depth as depth_module
from app.depth import METHODS, _denoise_and_fill, _sample_sparse_depth, _training_features, calculate_metrics, reconstruct
from app.model import ResidualUNet, load_checkpoint, train_checkpoint
from app.main import app


def test_training_sparse_samples_only_controlled_grid_with_noise_and_dropout():
    target = np.full((64, 64), 2.0, dtype=np.float32)

    sparse, sample_mask, observed_mask = _sample_sparse_depth(target, seed=2, scale=4)

    assert sample_mask.sum() == pytest.approx(target.size / 16, rel=0.15)
    sample_rows, sample_columns = np.nonzero(sample_mask)
    assert np.all(sample_rows % 4 == sample_rows[0] % 4)
    assert np.all(sample_columns % 4 == sample_columns[0] % 4)
    assert np.all(sparse[~sample_mask] == 0)
    assert np.all(np.isfinite(sparse[observed_mask]))
    assert np.all(sparse[observed_mask] > 0)
    assert np.any(sparse[observed_mask] != target[observed_mask])
    assert np.any(sample_mask & ~observed_mask)
    assert np.all(sparse[~observed_mask] == 0)


@pytest.mark.parametrize("seed", (100, 101, 102))
def test_training_features_and_targets_are_finite_for_each_sampling_scale(seed):
    features, target = _training_features(seed)

    assert np.isfinite(features).all()
    assert np.isfinite(target).all()


def test_training_features_do_not_leak_unobserved_ground_truth(monkeypatch):
    seed = 123
    rgb, target = depth_module._scene(seed, 64)
    scale = (2, 4, 8)[seed % 3]
    _, sample_mask, _ = _sample_sparse_depth(target, seed, scale)
    changed_target = target.copy()
    changed_target[~sample_mask] += 0.25

    monkeypatch.setattr(depth_module, "_scene", lambda _seed, _size: (rgb.copy(), target.copy()))
    features, labels = _training_features(seed)
    monkeypatch.setattr(depth_module, "_scene", lambda _seed, _size: (rgb.copy(), changed_target.copy()))
    changed_features, changed_labels = _training_features(seed)

    assert features.shape == (5, 64, 64)
    assert np.isfinite(features).all()
    assert np.array_equal(features, changed_features)
    assert not np.array_equal(labels, changed_labels)


def test_preprocessing_preserves_raw_mask_and_rejects_an_isolated_depth_spike():
    depth = np.full((9, 9), 2.0, dtype=np.float32)
    depth[4, 4] = 3.0
    depth[0, 0] = np.nan

    filled, original_mask, cleaned_mask = _denoise_and_fill(depth)

    assert original_mask[4, 4] == 1
    assert cleaned_mask[4, 4] == 0
    assert original_mask[0, 0] == 0
    assert np.isfinite(filled).all()
    assert filled[4, 4] == pytest.approx(2.0)


@pytest.mark.parametrize("method", ("guided", "cnn"))
def test_missing_and_noisy_depth_inputs_produce_finite_predictions(method):
    rgb, sparse, _ = demo_scene()
    noisy_depth = sparse.copy()
    noisy_depth[1, 1] = np.nan
    noisy_depth[2, 2] = np.inf
    noisy_depth[3, 3] = 0
    noisy_depth[4, 4] += 35

    prediction, _ = reconstruct(rgb, noisy_depth, method, scale=4)

    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("scale", (2, 4, 8))
@pytest.mark.parametrize("method", METHODS)
def test_reconstruction_returns_finite_scale_aligned_depth(method, scale):
    rgb, sparse, truth = demo_scene()

    prediction, latency = reconstruct(rgb, sparse, method, scale=scale)

    assert prediction.shape == (
        min(rgb.shape[0], sparse.shape[0] * scale),
        min(rgb.shape[1], sparse.shape[1] * scale),
    )
    assert np.isfinite(prediction).all()
    assert prediction.min() >= 0
    assert latency >= 0
    assert truth.shape == rgb.shape[:2]


def test_residual_unet_trains_saves_reloads_and_is_cached(tmp_path, monkeypatch):
    checkpoint = tmp_path / "residual_unet.pt"

    training = train_checkpoint(checkpoint, epochs=2, patience=1)
    loaded = load_checkpoint(checkpoint)
    depth_module._model.cache_clear()
    monkeypatch.setattr(depth_module, "DEFAULT_CHECKPOINT", checkpoint)
    monkeypatch.setattr(
        depth_module,
        "train_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("A valid saved checkpoint should not be retrained."),
    )
    cached_first = depth_module._model()
    cached_second = depth_module._model()
    depth_module._model.cache_clear()
    rgb, sparse, _ = demo_scene()
    filled, valid_mask, _ = _denoise_and_fill(sparse)
    low, high = float(np.min(filled)), float(np.max(filled))
    initial = cv2.resize(
        depth_module._to_unit(filled, low, high),
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    mask = cv2.resize(valid_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    residual = loaded.predict(rgb, np.clip(initial, 0, 1), mask)

    assert checkpoint.is_file()
    assert training["best_epoch"] in (1, 2)
    assert np.isfinite(training["validation_loss"])
    assert residual.shape == rgb.shape[:2]
    assert np.isfinite(residual).all()
    assert cached_first is cached_second


def test_missing_depth_values_are_filled_and_empty_input_is_rejected():
    rgb, sparse, _ = demo_scene()
    sparse[0, 0] = 0
    prediction, _ = reconstruct(rgb, sparse, "guided", scale=4)
    assert np.isfinite(prediction).all()

    with pytest.raises(ValueError, match="no positive"):
        reconstruct(rgb, np.zeros_like(sparse), "guided", scale=4)


def test_metrics_are_finite_for_exact_prediction():
    rgb, _, truth = demo_scene()
    metrics = calculate_metrics(truth, truth, rgb)

    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert 0 <= metrics["edge_preservation"] <= 1
    assert metrics["boundary_rmse"] == 0
    assert metrics["boundary_mae"] == 0


def test_boundary_errors_use_twice_dilated_ground_truth_edge_band():
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    truth = np.zeros((24, 24), dtype=np.float32)
    truth[:, 12:] = 10
    prediction = truth.copy()
    prediction[9:15, 9:15] += 2
    metrics = calculate_metrics(prediction, truth, rgb)

    sobel_x = cv2.Sobel(truth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(truth, cv2.CV_32F, 0, 1, ksize=3)
    boundary = (np.hypot(sobel_x, sobel_y) > 0.8).astype(np.uint8)
    band = cv2.dilate(boundary, np.ones((3, 3), dtype=np.uint8), iterations=2).astype(bool)
    residual = (prediction - truth)[band]

    assert metrics["boundary_rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert metrics["boundary_mae"] == pytest.approx(np.mean(np.abs(residual)))


def test_process_api_returns_downloadable_outputs_and_metrics():
    rgb, sparse, truth = demo_scene()
    images = []
    for array in (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), sparse.astype(np.uint16), truth.astype(np.uint16)):
        ok, payload = cv2.imencode(".png", array)
        assert ok
        images.append(payload.tobytes())

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={
                "rgb_file": ("rgb.png", images[0], "image/png"),
                "depth_file": ("depth.png", images[1], "image/png"),
                "ground_truth_file": ("truth.png", images[2], "image/png"),
            },
            data={"method": "hybrid", "scale": "4", "radius": "5", "epsilon": "0.0001"},
        )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["method"] == "hybrid"
    assert result["metrics_available"] is True
    assert result["metrics"]["rmse"] >= 0
    assert result["metrics"]["boundary_rmse"] >= 0
    assert result["metrics"]["boundary_mae"] >= 0
    assert 0 <= result["metrics"]["edge_f1"] <= 1
    assert result["depth_png"]
    assert result["edge_png"]
    assert result["process_rss_mb"] > 0
    assert result["error_png"]


@pytest.mark.parametrize(
    ("epsilon", "expected_status"),
    (("0.0001", 200), ("0.000001", 400)),
)
def test_process_api_validates_epsilon_range(epsilon, expected_status):
    rgb, sparse, _ = demo_scene()
    rgb_ok, rgb_bytes = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth_ok, depth_bytes = cv2.imencode(".png", sparse.astype(np.uint16))
    assert rgb_ok and depth_ok

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={
                "rgb_file": ("rgb.png", rgb_bytes.tobytes(), "image/png"),
                "depth_file": ("depth.png", depth_bytes.tobytes(), "image/png"),
            },
            data={
                "method": "hybrid",
                "scale": "4",
                "radius": "5",
                "epsilon": epsilon,
            },
        )

    assert response.status_code == expected_status


def test_process_api_reports_invalid_methods():
    rgb, sparse, _ = demo_scene()
    rgb_ok, rgb_bytes = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth_ok, depth_bytes = cv2.imencode(".png", sparse.astype(np.uint16))
    assert rgb_ok and depth_ok

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={
                "rgb_file": ("rgb.png", rgb_bytes.tobytes(), "image/png"),
                "depth_file": ("depth.png", depth_bytes.tobytes(), "image/png"),
            },
            data={"method": "unknown", "scale": "4", "radius": "5", "epsilon": "0.0001"},
        )

    assert response.status_code == 400
    assert "supported method" in response.json()["detail"]


def test_process_api_marks_supervised_metrics_unavailable_without_ground_truth():
    rgb, sparse, _ = demo_scene()
    rgb_ok, rgb_bytes = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth_ok, depth_bytes = cv2.imencode(".png", sparse.astype(np.uint16))
    assert rgb_ok and depth_ok

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={
                "rgb_file": ("rgb.png", rgb_bytes.tobytes(), "image/png"),
                "depth_file": ("depth.png", depth_bytes.tobytes(), "image/png"),
            },
            data={"method": "nearest", "scale": "4", "radius": "5", "epsilon": "0.0001"},
        )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["metrics_available"] is False
    assert "rmse" not in result["metrics"]
    assert result["metrics"]["edge_preservation"] >= 0
    assert result["error_png"] is None


def test_process_api_can_simulate_sparse_depth_without_rgb_or_reference_upload():
    _, _, truth = demo_scene()
    ok, depth_bytes = cv2.imencode(".png", truth.astype(np.uint16))
    assert ok

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"depth_file": ("dense-depth.png", depth_bytes.tobytes(), "image/png")},
            data={"method": "nearest", "scale": "4", "simulate": "true"},
        )

    assert response.status_code == 200
    payload = response.json()
    result = payload["results"][0]
    assert payload["input"]["simulated"] is True
    assert payload["input"]["has_ground_truth"] is True
    assert payload["input"]["width"] == truth.shape[1]
    assert payload["input"]["height"] == truth.shape[0]
    assert result["metrics_available"] is True
    assert np.isfinite(result["metrics"]["boundary_rmse"])
    assert np.isfinite(result["metrics"]["boundary_mae"])
    assert result["depth_png"]


def test_process_api_generates_rgb_guide_when_rgb_is_omitted():
    _, sparse, _ = demo_scene()
    ok, depth_bytes = cv2.imencode(".png", sparse.astype(np.uint16))
    assert ok

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"depth_file": ("sparse-depth.png", depth_bytes.tobytes(), "image/png")},
            data={"method": "nearest", "scale": "4"},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["metrics_available"] is False


def test_hybrid_variant_flags_skip_disabled_components(monkeypatch):
    rgb, sparse, _ = demo_scene()
    calls = {"guided": 0, "unet": 0, "refine": 0}

    def guided(guide, source, radius, epsilon):
        calls["guided"] += 1
        return source

    def predict(guide, source, mask):
        calls["unet"] += 1
        return np.zeros_like(source)

    def refine(guide, prediction, base):
        calls["refine"] += 1
        return prediction, np.zeros_like(prediction, dtype=np.uint8)

    monkeypatch.setattr(depth_module, "_guided_filter", guided)
    monkeypatch.setattr(depth_module, "_model", lambda: type("Model", (), {"predict": staticmethod(predict)})())
    monkeypatch.setattr(depth_module, "_edge_aware_refine", refine)

    reconstruct(rgb, sparse, "hybrid", variant={})
    assert calls == {"guided": 1, "unet": 1, "refine": 1}

    reconstruct(
        rgb,
        sparse,
        "hybrid",
        variant={"use_guided": False, "use_unet": False, "use_refine": False},
    )
    assert calls == {"guided": 1, "unet": 1, "refine": 1}
