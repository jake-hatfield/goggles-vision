from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image


_INPUT_SHAPES = ((64, 64), (96, 48), (112, 40), (128, 32))  # width, height
_CTC_DIGITS = "0123456789"
_BLANK_INDEX = 0
_BATCH_SIZE = 16


class _DropPath(nn.Module):
    def __init__(self, drop_probability):
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, inputs):
        if self.drop_probability == 0.0 or not self.training:
            return inputs
        keep_probability = 1.0 - self.drop_probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep_probability)
        return inputs * mask.div_(keep_probability)


class _Mlp(nn.Module):
    def __init__(self, channels, hidden_channels, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(inputs)))))


class _Attention(nn.Module):
    def __init__(self, channels, heads, qkv_bias, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_channels = channels // heads
        self.scale = self.head_channels**-0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs):
        batch, length, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, length, 3, self.heads, self.head_channels
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        outputs = (attention @ value).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(outputs))


class _EncoderBlock(nn.Module):
    def __init__(self, channels, heads, qkv_bias=True, drop_path=0.0, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=eps)
        self.mixer = _Attention(channels, heads, qkv_bias)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(channels, eps=eps)
        self.mlp = _Mlp(channels, channels * 4)

    def forward(self, inputs):
        inputs = self.norm1(inputs + self.drop_path(self.mixer(inputs)))
        return self.norm2(inputs + self.drop_path(self.mlp(inputs)))


class _DecoderBlock(nn.Module):
    def __init__(self, channels, heads, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = _Attention(channels, heads, qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = _Mlp(channels, channels * 4)

    def forward(self, inputs):
        inputs = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        return inputs + self.drop_path(self.mlp(self.norm2(inputs)))


class _ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 bias=False, groups=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        return self.act(self.norm(self.conv(inputs)))


class _ConvBlock(nn.Module):
    def __init__(self, channels, heads, num_conv, kernel_size=3, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=eps)
        self.mixer = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2,
                      groups=heads)
            for _ in range(num_conv)
        ])
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(channels, eps=eps)
        self.mlp = _Mlp(channels, channels * 4)

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        inputs = inputs + self.mixer(inputs)
        inputs = self.norm1(inputs.flatten(2).transpose(1, 2))
        inputs = self.norm2(inputs + self.mlp(inputs))
        return inputs.transpose(1, 2).reshape(batch, channels, height, width)


class _FlattenTranspose(nn.Module):
    def forward(self, inputs):
        return inputs.flatten(2).transpose(1, 2)


class _SubSample2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, inputs, size):
        inputs = self.conv(inputs)
        batch, channels, height, width = inputs.shape
        inputs = self.norm(inputs.flatten(2).transpose(1, 2))
        inputs = inputs.transpose(1, 2).reshape(batch, channels, height, width)
        return inputs, [height, width]


class _SubSample1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, inputs, size):
        channels = inputs.shape[-1]
        inputs = inputs.transpose(1, 2).reshape(-1, channels, size[0], size[1])
        inputs = self.conv(inputs)
        batch, channels, height, width = inputs.shape
        inputs = self.norm(inputs.flatten(2).transpose(1, 2))
        return inputs, [height, width]


class _IdentitySize(nn.Module):
    def forward(self, inputs, size):
        return inputs, size


class _SVTRStage(nn.Module):
    def __init__(self, channels, out_channels, mixers, heads, stride, drop_paths,
                 num_convs, downsample, eps=1e-6):
        super().__init__()
        self.blocks = nn.Sequential()
        for index, mixer in enumerate(mixers):
            if mixer == "Conv":
                block = _ConvBlock(channels, heads, num_convs[index], eps=eps)
                self.blocks.append(block)
                continue
            if mixer == "FGlobal":
                self.blocks.append(_FlattenTranspose())
            self.blocks.append(
                _EncoderBlock(channels, heads, drop_path=drop_paths[index], eps=eps)
            )

        if not downsample:
            self.downsample = _IdentitySize()
        elif mixers[-1] == "Conv":
            self.downsample = _SubSample2D(channels, out_channels, stride)
        else:
            self.downsample = _SubSample1D(channels, out_channels, stride)

    def forward(self, inputs, size):
        return self.downsample(self.blocks(inputs), size)


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Sequential(
            _ConvBNLayer(3, 64, stride=2, padding=1),
            _ConvBNLayer(64, 128, stride=2, padding=1),
        )

    def forward(self, inputs):
        height, width = inputs.shape[-2:]
        return self.patch_embed(inputs), [height // 4, width // 4]


class _Feat2D(nn.Module):
    def forward(self, inputs, size):
        channels = inputs.shape[-1]
        return inputs.transpose(1, 2).reshape(-1, channels, size[0], size[1]), size


class _SVTRv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pope = _PatchEmbed()
        dimensions = (128, 256, 384)
        depths = (6, 6, 6)
        heads = (4, 8, 12)
        mixers = (
            ("Conv",) * 6,
            ("Conv", "Conv", "FGlobal", "Global", "Global", "Global"),
            ("Global",) * 6,
        )
        strides = ((1, 1), (2, 1), (-1, -1))
        drop_paths = np.linspace(0.0, 0.1, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for index, depth in enumerate(depths):
            downsample = index < len(depths) - 1
            num_convs = [2] * depth
            stage = _SVTRStage(
                dimensions[index],
                dimensions[index + 1] if downsample else dimensions[index],
                mixers[index],
                heads[index],
                strides[index],
                drop_paths[offset:offset + depth],
                num_convs,
                downsample,
            )
            self.stages.append(stage)
            offset += depth
        self.stages.append(_Feat2D())

    def forward(self, inputs):
        if inputs.ndim == 5:
            inputs = inputs.flatten(0, 1)
        inputs, size = self.pope(inputs)
        for stage in self.stages:
            inputs, size = stage(inputs, size)
        return inputs


class _RCTCDecoder(nn.Module):
    def __init__(self, channels=384, classes=95):
        super().__init__()
        self.char_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.fc = nn.Linear(channels, classes)
        self.fc_kv = nn.Linear(channels, channels * 2)
        self.w_atten_block = _DecoderBlock(channels, channels // 32)
        self.out_channels = classes
        self.return_feats = False

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        inputs = self.w_atten_block(
            inputs.permute(0, 2, 3, 1).reshape(-1, width, channels)
        ).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        key_value = self.fc_kv(inputs.flatten(2).transpose(1, 2)).reshape(
            batch, height * width, 2, channels
        ).permute(2, 0, 3, 1)
        keys, values = key_value.unbind(0)
        attention = self.char_token.tile((batch, 1, 1)) @ keys
        attention = attention.reshape(-1, 1, height, width).softmax(dim=2)
        attention = attention.permute(0, 3, 1, 2)
        values = values.reshape(batch, channels, height, width)
        features = (attention @ values.permute(0, 3, 2, 1)).squeeze(2)
        return self.fc(features)


class _SVTRv2Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_decoder = _RCTCDecoder()

    def forward(self, inputs):
        return self.ctc_decoder(inputs)


class SVTRv2JerseyModel(nn.Module):
    """SVTRv2-LNConv-Two33 with the fine-tuned 95-class CTC head."""

    def __init__(self):
        super().__init__()
        self.encoder = _SVTRv2Encoder()
        self.decoder = _SVTRv2Decoder()

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class JerseyNumberReader:
    """Read jersey numbers with the fine-tuned SVTRv2-B model."""

    def __init__(self, parameters, device):
        self.parameters = parameters or {}
        self.device = torch.device(device)
        self.model = None
        self.error = None
        self.observations = 0
        self.accepted = 0

    def _load(self):
        if self.model is not None:
            return True
        if self.error is not None or not self.parameters.get("enabled", True):
            return False
        try:
            checkpoint = Path(self.parameters["checkpoint"]).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"SVTRv2 jersey checkpoint not found: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            state = {
                key: value for key, value in state.items()
                if not key.startswith("decoder.gtc_decoder.")
            }
            self.model = SVTRv2JerseyModel()
            self.model.load_state_dict(state, strict=True)
            self.model = self.model.eval().to(self.device)
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"warning: jersey-number OCR is unavailable: {self.error}")
            return False

    @staticmethod
    def _bucket(crop):
        ratio = crop.shape[1] / max(1, crop.shape[0])
        return min(4, max(1, round(ratio))) - 1

    @staticmethod
    def _preprocess(crop, bucket):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        width, height = _INPUT_SHAPES[bucket]
        resized = Image.fromarray(rgb).resize(
            (width, height), Image.Resampling.BICUBIC
        )
        pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy()
        return torch.from_numpy(pixels).div_(127.5).sub_(1.0)

    @staticmethod
    def _decode(probabilities):
        indices = probabilities.argmax(axis=-1)
        scores = probabilities.max(axis=-1)
        digits = []
        digit_scores = []
        previous = _BLANK_INDEX
        invalid_character = False
        for index, score in zip(indices, scores):
            index = int(index)
            if index != _BLANK_INDEX and index != previous:
                if 1 <= index <= len(_CTC_DIGITS):
                    digits.append(_CTC_DIGITS[index - 1])
                    digit_scores.append(float(score))
                else:
                    invalid_character = True
            previous = index
        number = "".join(digits)
        if invalid_character or not 1 <= len(number) <= 2:
            return "", 0.0
        return number, float(np.mean(digit_scores))

    def predict(self, crops):
        if not crops or not self._load():
            return []

        result = [{} for _ in crops]
        buckets = {}
        for index, row in enumerate(crops):
            crop = row.get("full")
            if crop is None or not crop.size:
                continue
            bucket = self._bucket(crop)
            buckets.setdefault(bucket, []).append((index, self._preprocess(crop, bucket)))
        if not buckets:
            return result

        self.observations += len(crops)
        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.inference_mode():
            for entries in buckets.values():
                for start in range(0, len(entries), _BATCH_SIZE):
                    batch_entries = entries[start:start + _BATCH_SIZE]
                    batch = torch.stack([image for _, image in batch_entries]).to(
                        self.device, non_blocking=True
                    )
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        logits = self.model(batch)
                    probabilities = logits.float().softmax(-1).cpu().numpy()
                    for (index, _), prediction in zip(batch_entries, probabilities):
                        number, confidence = self._decode(prediction)
                        threshold = float(self.parameters.get("min_read_confidence", 0.55))
                        if number and confidence >= threshold:
                            result[index] = {"number": number, "confidence": confidence}
                            accepted_threshold = float(self.parameters.get("confidence", 0.80))
                            self.accepted += confidence >= accepted_threshold
        return result
