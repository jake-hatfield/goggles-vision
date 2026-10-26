"""Standalone PyTorch inference for the OpenMMLab RTMPose-S AIC/COCO model.

The network blocks and SimCC decoder are adapted from OpenMMLab MMPose under
Apache-2.0. The implementation uses only PyTorch, NumPy, and OpenCV, all of
which are already project dependencies. See licenses/OpenMMLab-Apache-2.0.txt.
"""

from concurrent.futures import Future
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inference import CudaInferenceExecutor


INPUT_WIDTH = 192
INPUT_HEIGHT = 256
KEYPOINT_COUNT = 17
SIMCC_SPLIT_RATIO = 2
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)


class _ConvModule(nn.Module):
    """Conv2d, BatchNorm, and SiLU with OpenMMLab-compatible names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # The checkpoint config uses SyncBN with PyTorch's defaults.
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class _DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.depthwise_conv = _ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise_conv = _ConvModule(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.global_avgpool(x)))


class _CSPNeXtBlock(nn.Module):
    def __init__(self, channels: int, *, add_identity: bool) -> None:
        super().__init__()
        self.conv1 = _ConvModule(channels, channels, 3, padding=1)
        self.conv2 = _DepthwiseSeparableConvModule(channels, channels, 5, padding=2)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.conv1(x))
        return output + x if self.add_identity else output


class _CSPLayer(nn.Module):
    def __init__(self, channels: int, num_blocks: int, *, add_identity: bool, channel_attention: bool) -> None:
        super().__init__()
        hidden_channels = channels // 2
        self.main_conv = _ConvModule(channels, hidden_channels, 1)
        self.short_conv = _ConvModule(channels, hidden_channels, 1)
        self.final_conv = _ConvModule(2 * hidden_channels, channels, 1)
        self.blocks = nn.Sequential(*[
            _CSPNeXtBlock(hidden_channels, add_identity=add_identity)
            for _ in range(num_blocks)
        ])
        self.channel_attention = channel_attention
        if channel_attention:
            self.attention = _ChannelAttention(2 * hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.blocks(self.main_conv(x))
        short = self.short_conv(x)
        output = torch.cat((main, short), dim=1)
        if self.channel_attention:
            output = self.attention(output)
        return self.final_conv(output)


class _SPPBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = channels // 2
        self.conv1 = _ConvModule(channels, hidden_channels, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            for size in (5, 9, 13)
        ])
        self.conv2 = _ConvModule(hidden_channels * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat([x, *(pool(x) for pool in self.poolings)], dim=1))


class _CSPNeXtBackbone(nn.Module):
    """CSPNeXt-S backbone (P5, width .5, depth .33)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _ConvModule(3, 16, 3, stride=2, padding=1),
            _ConvModule(16, 16, 3, padding=1),
            _ConvModule(16, 32, 3, padding=1),
        )
        self.stage1 = nn.Sequential(
            _ConvModule(32, 64, 3, stride=2, padding=1),
            _CSPLayer(64, 1, add_identity=True, channel_attention=True),
        )
        self.stage2 = nn.Sequential(
            _ConvModule(64, 128, 3, stride=2, padding=1),
            _CSPLayer(128, 2, add_identity=True, channel_attention=True),
        )
        self.stage3 = nn.Sequential(
            _ConvModule(128, 256, 3, stride=2, padding=1),
            _CSPLayer(256, 2, add_identity=True, channel_attention=True),
        )
        self.stage4 = nn.Sequential(
            _ConvModule(256, 512, 3, stride=2, padding=1),
            _SPPBottleneck(512),
            _CSPLayer(512, 1, add_identity=False, channel_attention=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)


class _ScaleNorm(nn.Module):
    def __init__(self, dimensions: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = dimensions**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class _Scale(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dimensions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _RTMCCBlock(nn.Module):
    """The self-attention GAU block used by the RTMPose SimCC head."""

    def __init__(self) -> None:
        super().__init__()
        self.s = 128
        self.e = 512
        self.o = nn.Linear(512, 256, bias=False)
        self.uv = nn.Linear(256, 2 * self.e + self.s, bias=False)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.ln = _ScaleNorm(256)
        self.act_fn = nn.SiLU(inplace=True)
        self.shortcut = True
        self.res_scale = _Scale(256)
        self.sqrt_s = float(np.sqrt(self.s))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        uv = self.act_fn(self.uv(self.ln(inputs)))
        u, v, base = torch.split(uv, (self.e, self.e, self.s), dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        query, key = torch.unbind(base, dim=2)
        kernel = torch.square(F.relu(torch.bmm(query, key.transpose(1, 2)) / self.sqrt_s))
        output = self.o(u * torch.bmm(kernel, v))
        return self.res_scale(inputs) + output


class _RTMCCHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.final_layer = nn.Conv2d(512, KEYPOINT_COUNT, kernel_size=7, stride=1, padding=3)
        self.mlp = nn.Sequential(
            _ScaleNorm(48),
            nn.Linear(48, 256, bias=False),
        )
        self.gau = _RTMCCBlock()
        self.cls_x = nn.Linear(256, INPUT_WIDTH * SIMCC_SPLIT_RATIO, bias=False)
        self.cls_y = nn.Linear(256, INPUT_HEIGHT * SIMCC_SPLIT_RATIO, bias=False)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.final_layer(features), 2)
        features = self.gau(self.mlp(features))
        return self.cls_x(features), self.cls_y(features)


class RTMPoseS(nn.Module):
    """RTMPose-S AIC/COCO checkpoint architecture for 192x256 crops."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _CSPNeXtBackbone()
        self.head = _RTMCCHead()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(inputs))


def _bbox_affine(frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make an aspect-preserving, 1.25x top-down crop from a detector box."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    center = np.asarray(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float32)
    scale = np.asarray((max(1.0, x2 - x1), max(1.0, y2 - y1)), dtype=np.float32) * 1.25
    aspect = INPUT_WIDTH / INPUT_HEIGHT
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect

    src_dir = np.asarray((0.0, -0.5 * scale[0]), dtype=np.float32)
    dst_dir = np.asarray((0.0, -0.5 * INPUT_WIDTH), dtype=np.float32)
    src_points = np.asarray((center, center + src_dir, center + (-src_dir[1], src_dir[0])), dtype=np.float32)
    dst_center = np.asarray((0.5 * INPUT_WIDTH, 0.5 * INPUT_HEIGHT), dtype=np.float32)
    dst_points = np.asarray((dst_center, dst_center + dst_dir, dst_center + (-dst_dir[1], dst_dir[0])), dtype=np.float32)
    matrix = cv2.getAffineTransform(src_points, dst_points)
    crop = cv2.warpAffine(frame, matrix, (INPUT_WIDTH, INPUT_HEIGHT), flags=cv2.INTER_LINEAR)
    return np.ascontiguousarray(crop), center, scale


class RTMPosePlayerEstimator:
    """Batch detector-provided player ROIs and return full-frame COCO-17 poses."""

    def __init__(self, checkpoint: str | Path, batch_size: int, device: int | str) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint}")
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.batch_size = max(1, int(batch_size))
        self.model = RTMPoseS()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = saved.get("state_dict", saved)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.inference = CudaInferenceExecutor(device, self.detect)

    def start(self) -> None:
        self.inference.start()

    def submit(self, frame: np.ndarray, boxes: np.ndarray) -> Future[Any] | None:
        return self.inference.submit(frame, boxes)

    def result(
        self,
        future: Future[Any] | None,
        frame: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.inference.result(future, frame, boxes)

    def detect(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pose_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        pose_count = len(pose_boxes)
        if pose_count == 0:
            return (
                pose_boxes,
                np.empty((0,), dtype=np.float32),
                np.empty((0, KEYPOINT_COUNT, 2), dtype=np.float32),
                np.empty((0, KEYPOINT_COUNT), dtype=np.float32),
            )

        crops, centers, scales = [], [], []
        for box in pose_boxes:
            crop, center, scale = _bbox_affine(frame, box)
            crops.append(crop)
            centers.append(center)
            scales.append(scale)

        points = np.empty((pose_count, KEYPOINT_COUNT, 2), dtype=np.float32)
        scores = np.empty((pose_count, KEYPOINT_COUNT), dtype=np.float32)
        mean = torch.as_tensor(_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std = torch.as_tensor(_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        with torch.inference_mode():
            for start in range(0, pose_count, self.batch_size):
                end = min(start + self.batch_size, pose_count)
                batch_array = np.stack([
                    crop[:, :, ::-1].transpose(2, 0, 1)
                    for crop in crops[start:end]
                ]).copy()
                batch = torch.from_numpy(batch_array).to(self.device, dtype=torch.float32)
                batch = (batch - mean) / std
                simcc_x, simcc_y = self.model(batch)
                x_scores, x_indices = simcc_x.max(dim=-1)
                y_scores, y_indices = simcc_y.max(dim=-1)
                batch_scores = torch.minimum(x_scores, y_scores)
                batch_points = torch.stack((x_indices, y_indices), dim=-1).to(torch.float32)
                batch_points /= SIMCC_SPLIT_RATIO
                points[start:end] = batch_points.cpu().numpy()
                scores[start:end] = batch_scores.cpu().numpy()

        points = (
            points / np.asarray((INPUT_WIDTH, INPUT_HEIGHT), dtype=np.float32)
            * np.asarray(scales, dtype=np.float32)[:, None, :]
            + np.asarray(centers, dtype=np.float32)[:, None, :]
            - 0.5 * np.asarray(scales, dtype=np.float32)[:, None, :]
        )
        box_scores = np.ones((pose_count,), dtype=np.float32)
        return pose_boxes, box_scores, points, scores

    def close(self) -> None:
        self.inference.close()


__all__ = ["RTMPosePlayerEstimator", "RTMPoseS"]
