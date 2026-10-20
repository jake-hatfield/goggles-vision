from typing import Any

import numpy as np
import supervision as sv


def predict_shape(frame: np.ndarray, short_side: int | None = None) -> tuple[int, int]:
    """Choose detector input dimensions while preserving the frame aspect ratio."""
    short_side = 512 if short_side is None else short_side
    stride = 32
    height, width = frame.shape[:2]
    scale = min(1.0, short_side / min(height, width), 1333 / max(height, width))
    return max(stride, int(height * scale) // stride * stride), max(
        stride, int(width * scale) // stride * stride
    )


def resolve_detector_class_ids(model: Any) -> dict[str, tuple[int, ...]]:
    """Map normalized detector class names to all matching model class IDs."""
    names = getattr(model, "class_names", {})
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    class_map = {}
    for class_id, name in entries:
        key = str(name).lower().replace("_", " ").replace("-", " ").strip()
        class_map.setdefault(key, []).append(int(class_id))
    return {name: tuple(ids) for name, ids in class_map.items()}


def _class_mask(class_ids, ids, name):
    target_ids = class_ids.get(name, ())
    if not target_ids:
        return np.zeros(ids.shape, dtype=bool)
    return ids == target_ids[0] if len(target_ids) == 1 else np.isin(ids, target_ids)


def _class_thresholds(
    player: float,
    ball: float,
    rim: float,
    holder: float,
    scorer: float,
    referee: float,
    ball_in_basket: float,
) -> dict[str, float]:
    return {
        "player": player,
        "ball": ball,
        "rim": rim,
        "holder": holder,
        "scorer": scorer,
        "referee": referee,
        "ball in basket": ball_in_basket,
    }


def _partition_detections(
    detections: Any,
    class_map: dict[str, tuple[int, ...]],
    thresholds: dict[str, float],
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if detections is None or len(detections) == 0 or detections.class_id is None:
        empty = sv.Detections.empty()
        return empty, empty, empty, empty, empty, empty, empty
    detected_ids = np.asarray(detections.class_id, dtype=np.int32)
    scores = (
        np.ones(len(detected_ids), dtype=np.float32)
        if detections.confidence is None
        else np.asarray(detections.confidence, dtype=np.float32)
    )
    names = ("player", "ball", "rim", "holder", "scorer", "referee", "ball in basket")
    return tuple(
        detections[
            _class_mask(class_map, detected_ids, name)
            & (scores >= float(thresholds[name]))
        ]
        for name in names
    )


def detect(
    model: Any,
    frame_rgb: np.ndarray,
    shape: tuple[int, int],
    class_map: dict[str, tuple[int, ...]],
    player_threshold: float,
    ball_threshold: float,
    rim_threshold: float = 0.5,
    holder_threshold: float = 0.5,
    scorer_threshold: float = 0.5,
    referee_threshold: float = 0.5,
    ball_in_basket_threshold: float = 0.5,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Run detector inference and partition results by basketball class."""
    detections = model.predict(
        frame_rgb, threshold=0.1, shape=shape, include_source_image=False
    )
    return _partition_detections(
        detections,
        class_map,
        _class_thresholds(
            player_threshold,
            ball_threshold,
            rim_threshold,
            holder_threshold,
            scorer_threshold,
            referee_threshold,
            ball_in_basket_threshold,
        ),
    )


def detect_batch(
    model: Any,
    frames_rgb: list[np.ndarray],
    shape: tuple[int, int],
    class_map: dict[str, tuple[int, ...]],
    player_threshold: float,
    ball_threshold: float,
    rim_threshold: float,
    holder_threshold: float,
    scorer_threshold: float,
    referee_threshold: float,
    ball_in_basket_threshold: float,
) -> list[tuple[Any, Any, Any, Any, Any, Any, Any]]:
    """Run batched detector inference and partition each frame's results."""
    predictions = model.predict(
        frames_rgb, threshold=0.1, shape=shape, include_source_image=False
    )
    thresholds = _class_thresholds(
        player_threshold,
        ball_threshold,
        rim_threshold,
        holder_threshold,
        scorer_threshold,
        referee_threshold,
        ball_in_basket_threshold,
    )
    return [
        _partition_detections(prediction, class_map, thresholds)
        for prediction in predictions
    ]


__all__ = ["detect", "detect_batch", "predict_shape", "resolve_detector_class_ids"]

