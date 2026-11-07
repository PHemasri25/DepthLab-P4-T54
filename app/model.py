from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODEL_FORMAT_VERSION = 1
MODEL_INPUT_CHANNELS = 5
MODEL_BASE_CHANNELS = 8
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "residual_unet.pt"


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs) + self.skip(inputs))


class ResidualUNet(nn.Module):
    """Compact RGB/depth/mask residual U-Net with encoder-decoder skip connections."""

    def __init__(self, base_channels: int = MODEL_BASE_CHANNELS) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoder1 = ResidualBlock(MODEL_INPUT_CHANNELS, channels[0])
        self.encoder2 = ResidualBlock(channels[0], channels[1])
        self.encoder3 = ResidualBlock(channels[1], channels[2])
        self.bottleneck = ResidualBlock(channels[2], channels[3])
        self.decoder3 = ResidualBlock(channels[3] + channels[2], channels[2])
        self.decoder2 = ResidualBlock(channels[2] + channels[1], channels[1])
        self.decoder1 = ResidualBlock(channels[1] + channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(F.avg_pool2d(skip1, kernel_size=2, ceil_mode=True))
        skip3 = self.encoder3(F.avg_pool2d(skip2, kernel_size=2, ceil_mode=True))
        encoded = self.bottleneck(F.avg_pool2d(skip3, kernel_size=2, ceil_mode=True))

        decoded = F.interpolate(encoded, size=skip3.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder3(torch.cat((decoded, skip3), dim=1))
        decoded = F.interpolate(decoded, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder2(torch.cat((decoded, skip2), dim=1))
        decoded = F.interpolate(decoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder1(torch.cat((decoded, skip1), dim=1))
        return self.output(decoded)

    @torch.inference_mode()
    def predict(
        self,
        rgb: np.ndarray,
        initial_depth: np.ndarray,
        validity_mask: np.ndarray,
    ) -> np.ndarray:
        rgb_unit = rgb.astype(np.float32) / 255.0
        inputs = np.concatenate(
            (
                np.moveaxis(rgb_unit, -1, 0),
                initial_depth[None].astype(np.float32),
                validity_mask[None].astype(np.float32),
            ),
            axis=0,
        )
        device = next(self.parameters()).device
        tensor = torch.from_numpy(np.ascontiguousarray(inputs)).unsqueeze(0).to(device)
        residual = self(tensor)[0, 0].cpu().numpy()
        return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .depth import _training_features

    training = [_training_features(seed) for seed in range(100, 120)]
    validation = [_training_features(seed) for seed in range(120, 124)]

    def stack(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.stack([features for features, _ in items]).astype(np.float32)
        targets = np.stack([target for _, target in items]).astype(np.float32)
        return inputs, targets

    train_x, train_y = stack(training)
    validation_x, validation_y = stack(validation)
    return train_x, train_y, validation_x, validation_y


def train_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    epochs: int = 24,
    patience: int = 5,
) -> dict[str, float | int | str]:
    if epochs < 1 or patience < 1:
        raise ValueError("Training epochs and patience must be positive.")
    torch.manual_seed(2026)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    train_x, train_y, validation_x, validation_y = _training_arrays()
    model = ResidualUNet().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_inputs = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_y)
    validation_inputs = torch.from_numpy(validation_x)
    validation_targets = torch.from_numpy(validation_y)
    rng = np.random.default_rng(2026)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    batch_size = 4

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_inputs))
        for start in range(0, len(order), batch_size):
            batch_indices = torch.from_numpy(order[start : start + batch_size].copy())
            batch_inputs = train_inputs[batch_indices]
            batch_targets = train_targets[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_inputs)
            loss = F.smooth_l1_loss(prediction, batch_targets, beta=0.02)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_inputs)
            validation_loss = float(F.smooth_l1_loss(validation_prediction, validation_targets, beta=0.02))
        if not np.isfinite(validation_loss):
            raise RuntimeError("Residual U-Net validation loss became non-finite.")
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Residual U-Net training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "base_channels": MODEL_BASE_CHANNELS,
        "input_channels": MODEL_INPUT_CHANNELS,
        "model_state": model.state_dict(),
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "training_seeds": list(range(100, 120)),
        "validation_seeds": list(range(120, 124)),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.stem}-",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"best_epoch": best_epoch, "validation_loss": best_loss, "checkpoint": str(checkpoint_path)}


def load_checkpoint(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> ResidualUNet:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained Residual U-Net checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Could not load Residual U-Net checkpoint: {checkpoint_path}") from error
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}.")
    model = ResidualUNet(base_channels=int(payload["base_channels"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model
