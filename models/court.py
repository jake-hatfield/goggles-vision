"""Court keypoint inference with the project-trained MobileNetV4 model."""

from concurrent.futures import Future
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inference import CudaInferenceExecutor
from tracking.court import COURT_KEYPOINT_COUNT


INPUT_WIDTH = 896
INPUT_HEIGHT = 512
HEATMAP_STRIDE = 4
DECODE_RADIUS = 5
DECODE_TEMPERATURE = 0.5
MIN_HEATMAP_PEAK_PROBABILITY = 0.055
MIN_COURT_VISIBILITY_SCORE = 0.005480526255443699
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class _SepConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        return self.act(self.bn2(self.pointwise(x)))


class _ConvBnAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn1(self.conv(x)))


class _ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _UniversalInvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        dw_start: int,
        dw_mid: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        mid_channels = in_channels * expansion
        self.dw_start = (
            _ConvNormAct(
                in_channels,
                in_channels,
                dw_start,
                groups=in_channels,
                activation=False,
            )
            if dw_start
            else nn.Identity()
        )
        self.pw_exp = _ConvNormAct(in_channels, mid_channels, 1)
        self.dw_mid = (
            _ConvNormAct(
                mid_channels,
                mid_channels,
                dw_mid,
                stride=stride,
                groups=mid_channels,
            )
            if dw_mid
            else nn.Identity()
        )
        self.se = nn.Identity()
        self.pw_proj = _ConvNormAct(mid_channels, out_channels, 1, activation=False)
        self.dw_end = nn.Identity()
        self.layer_scale = nn.Identity()
        self.drop_path = nn.Identity()
        self.has_skip = in_channels == out_channels and stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        return self.drop_path(x) + residual if self.has_skip else x


class _MobileNetV4ConvSmall(nn.Module):
    """Local feature extractor for the project's timm-trained MobileNetV4 checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            nn.Sequential(_ConvBnAct(32, 32, 3, 2), _ConvBnAct(32, 32, 1, 1)),
            nn.Sequential(_ConvBnAct(32, 96, 3, 2), _ConvBnAct(96, 64, 1, 1)),
            nn.Sequential(
                _UniversalInvertedResidual(64, 96, 3, 5, 5, 2),
                *[_UniversalInvertedResidual(96, 96, 2, 0, 3) for _ in range(4)],
                _UniversalInvertedResidual(96, 96, 4, 3, 0),
            ),
            nn.Sequential(
                _UniversalInvertedResidual(96, 128, 6, 3, 3, 2),
                _UniversalInvertedResidual(128, 128, 4, 5, 5),
                _UniversalInvertedResidual(128, 128, 4, 0, 5),
                _UniversalInvertedResidual(128, 128, 3, 0, 5),
                *[_UniversalInvertedResidual(128, 128, 4, 0, 3) for _ in range(2)],
            ),
            nn.Sequential(_ConvBnAct(128, 960, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.act1(self.bn1(self.conv_stem(x)))
        features = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index != 3:
                features.append(x)
        return features


class _CourtHeatmapModel(nn.Module):
    def __init__(self, backbone: str) -> None:
        super().__init__()
        if backbone != "mobilenetv4_conv_small.e3600_r256_in1k":
            raise ValueError(f"unsupported court backbone: {backbone}")
        self.encoder = _MobileNetV4ConvSmall()
        channels = (32, 64, 96, 960)
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, 64, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.SiLU(inplace=True),
                )
                for channel in channels
            ]
        )
        self.smooth = nn.ModuleList([_SepConv(64) for _ in range(3)])
        self.head = nn.Sequential(_SepConv(64), nn.Conv2d(64, COURT_KEYPOINT_COUNT, 1))
        self.visibility_head = nn.Linear(64, COURT_KEYPOINT_COUNT)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        x = self.lateral[3](features[3])
        for index in (2, 1, 0):
            x = self.lateral[index](features[index]) + F.interpolate(
                x, size=features[index].shape[-2:], mode="bilinear", align_corners=False
            )
            if index < 3:
                x = self.smooth[index](x)
        heatmaps = self.head(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            visibility_logits = self.visibility_head(x.float().mean(dim=(-2, -1)))
        return heatmaps, visibility_logits


class CourtPoseDetector:
    """Predict court keypoints and preserve the shared asynchronous model API."""

    def __init__(self, checkpoint: str | Path, device: int | str) -> None:
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state.get("width", INPUT_WIDTH),
            state.get("height", INPUT_HEIGHT),
        ) != (INPUT_WIDTH, INPUT_HEIGHT):
            raise ValueError(
                f"court checkpoint must use {INPUT_WIDTH}x{INPUT_HEIGHT} input"
            )
        self.model = _CourtHeatmapModel(state["backbone"]).to(
            device=self.device, memory_format=torch.channels_last
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        offsets = torch.arange(-DECODE_RADIUS, DECODE_RADIUS + 1, device=self.device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        self.offset_x = offset_x.reshape(1, 1, -1)
        self.offset_y = offset_y.reshape(1, 1, -1)
        self.inference = CudaInferenceExecutor(device, self.detect)

    def _prepare(self, frame: np.ndarray) -> tuple[torch.Tensor, float, int, int]:
        height, width = frame.shape[:2]
        ratio = min(INPUT_WIDTH / width, INPUT_HEIGHT / height)
        resized_width = int(round(width * ratio))
        resized_height = int(round(height * ratio))
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        horizontal = INPUT_WIDTH - resized_width
        vertical = INPUT_HEIGHT - resized_height
        left = int(round(horizontal / 2 - 0.1))
        top = int(round(vertical / 2 - 0.1))
        image = cv2.copyMakeBorder(
            resized,
            top,
            vertical - top,
            left,
            horizontal - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
        )
        inputs.div_(255.0).sub_(self.mean).div_(self.std)
        return inputs, ratio, left, top

    def _decode(self, logits: torch.Tensor) -> torch.Tensor:
        _, _, height, width = logits.shape
        heatmaps = logits.float().reshape(1, COURT_KEYPOINT_COUNT, -1)
        indices = heatmaps.argmax(dim=-1)
        x = (indices % width).unsqueeze(-1) + self.offset_x
        y = (indices // width).unsqueeze(-1) + self.offset_y
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
        values = heatmaps.gather(-1, indices).masked_fill(~valid, -1e4)
        weights = torch.softmax(values / DECODE_TEMPERATURE, dim=-1)
        return torch.stack(
            (
                (weights * x).sum(-1) * HEATMAP_STRIDE,
                (weights * y).sum(-1) * HEATMAP_STRIDE,
            ),
            dim=-1,
        )[0]

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs, ratio, pad_x, pad_y = self._prepare(frame)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            heatmaps, visibility_logits = self.model(inputs)
            heatmaps = heatmaps.float()
            points = self._decode(heatmaps).cpu().numpy()
            peak_probability = torch.softmax(heatmaps.flatten(2), dim=-1).amax(-1)[0]
            visibility_probability = torch.sigmoid(visibility_logits.float())[0]
        points[:, 0] = (points[:, 0] - pad_x) / ratio
        points[:, 1] = (points[:, 1] - pad_y) / ratio
        scores = (
            (
                peak_probability.ge(MIN_HEATMAP_PEAK_PROBABILITY)
                & visibility_probability.ge(MIN_COURT_VISIBILITY_SCORE)
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return points.astype(np.float32), scores

    def start(self) -> None:
        self.inference.start()

    def submit(self, frame: np.ndarray) -> Future[Any] | None:
        return self.inference.submit(frame)

    def result(
        self,
        future: Future[Any] | None,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.inference.result(future, frame)

    def close(self) -> None:
        self.inference.close()


__all__ = ["CourtPoseDetector"]
