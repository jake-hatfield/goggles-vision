"""Court geometry and player-position mapping for basketball tracking."""

from typing import Any

import cv2
import numpy as np

from tracking.pose import get_player_ground_point, validated_pose


COURT_KEYPOINT_COUNT = 33
COURT_LENGTH_M = 28.65
COURT_WIDTH_M = 15.24
COURT_MAP_MAX_AGE_FRAMES = 5
COURT_GEOMETRY = {
    "lane_half_width_m": 2.44,
    "baseline_offset_m": 0.91,
    "baseline_key_distance_m": 4.24,
    "free_throw_line_m": 5.79,
    "free_throw_radius_m": 1.83,
    "basket_offset_m": 1.6,
    "restricted_area_radius_m": 1.22,
    "three_point_radius_m": 7.24,
    "paint_width_m": 4.88,
    "three_point_corner_x_m": 8.35,
    "three_point_center_x_m": 8.84,
    "center_circle_samples": 72,
    "free_throw_circle_samples": 36,
    "three_point_arc_samples": 80,
}


def _arc(center, radius, start, end, samples):
    angles = np.deg2rad(np.linspace(start, end, samples))
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles))).astype(
        np.float32
    )


def court_paths() -> tuple[np.ndarray, ...]:
    """Return shared NBA court line paths in metric coordinates."""
    geometry = COURT_GEOMETRY
    length = COURT_LENGTH_M
    width = COURT_WIDTH_M
    center = width * 0.5
    top = center - geometry["lane_half_width_m"]
    bottom = center + geometry["lane_half_width_m"]
    angle = np.degrees(
        np.arctan2(
            center - geometry["baseline_offset_m"], geometry["baseline_key_distance_m"] - geometry["basket_offset_m"]
        )
    )
    paths = [
        np.asarray(((0, 0), (length, 0), (length, width), (0, width), (0, 0)), np.float32),
        np.asarray(((length * 0.5, 0), (length * 0.5, width)), np.float32),
        _arc(
            (length * 0.5, center),
            geometry["free_throw_radius_m"],
            0,
            360,
            geometry["center_circle_samples"],
        ),
    ]
    for right in (False, True):
        x = (lambda value: length - value) if right else (lambda value: value)
        paths.extend(
            (
                np.asarray(
                    (
                        (x(0), top),
                        (x(geometry["free_throw_line_m"]), top),
                        (x(geometry["free_throw_line_m"]), bottom),
                        (x(0), bottom),
                        (x(0), top),
                    ),
                    np.float32,
                ),
                _arc(
                    (x(geometry["free_throw_line_m"]), center),
                    geometry["free_throw_radius_m"],
                    0,
                    360,
                    geometry["free_throw_circle_samples"],
                ),
                np.asarray(
                    (
                        (x(0), geometry["baseline_offset_m"]),
                        (x(geometry["baseline_key_distance_m"]), geometry["baseline_offset_m"]),
                    ),
                    np.float32,
                ),
                np.asarray(
                    (
                        (x(0), width - geometry["baseline_offset_m"]),
                        (x(geometry["baseline_key_distance_m"]), width - geometry["baseline_offset_m"]),
                    ),
                    np.float32,
                ),
                _arc(
                    (x(geometry["basket_offset_m"]), center),
                    geometry["restricted_area_radius_m"],
                    90 if right else -90,
                    270 if right else 90,
                    geometry["free_throw_circle_samples"],
                ),
                _arc(
                    (x(geometry["basket_offset_m"]), center),
                    geometry["three_point_radius_m"],
                    180 - angle if right else -angle,
                    180 + angle if right else angle,
                    geometry["three_point_arc_samples"],
                ),
            )
        )
    return tuple(paths)
COURT_PATHS = court_paths()


def default_nba_court_vertices() -> np.ndarray:
    """Return image-correspondence points in the court model's keypoint order."""
    geometry = COURT_GEOMETRY
    length = COURT_LENGTH_M
    width = COURT_WIDTH_M
    center = width * 0.5
    paint_left = (width - geometry["paint_width_m"]) * 0.5
    paint_right = width - paint_left
    left = np.asarray(
        [
            (0, 0),
            (0, geometry["baseline_offset_m"]),
            (0, paint_left),
            (0, paint_right),
            (0, width - geometry["baseline_offset_m"]),
            (0, width),
            (geometry["baseline_key_distance_m"], geometry["baseline_offset_m"]),
            (geometry["basket_offset_m"], center),
            (geometry["baseline_key_distance_m"], width - geometry["baseline_offset_m"]),
            (geometry["free_throw_line_m"], paint_left),
            (geometry["free_throw_line_m"], center),
            (geometry["free_throw_line_m"], paint_right),
            (geometry["three_point_center_x_m"], center),
            (geometry["three_point_corner_x_m"], 0),
            (geometry["three_point_corner_x_m"], width),
        ],
        dtype=np.float32,
    )
    middle = np.asarray(((length * 0.5, 0), (length * 0.5, center), (length * 0.5, width)), dtype=np.float32)
    right = np.asarray((length, width), dtype=np.float32) - left
    return np.concatenate((left, middle, right), axis=0)


class CourtMapper:
    """Estimate a stable image-to-court homography across frames."""

    def __init__(self, vertices: np.ndarray, max_pending_gap: int | None = None) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32)
        if self.vertices.shape != (COURT_KEYPOINT_COUNT, 2):
            raise ValueError(
                f"expected {COURT_KEYPOINT_COUNT} court vertices, got {self.vertices.shape}"
            )
        if max_pending_gap is None:
            max_pending_gap = 1
        self.max_pending_gap = max(1, int(max_pending_gap))
        self.homography = None
        self.last_valid_frame = -(10**9)
        self.last_keypoint_frame = -(10**9)
        self.last_camera_motion_frame = -(10**9)
        self.last_stats = self._stats("uninitialized")
        self.model_update_attempts = 0
        self.model_update_acceptances = 0
        self.model_update_rejections = {}
        self.court_min = self.vertices.min(axis=0)
        self.court_max = self.vertices.max(axis=0)
        anchors = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.5], [0.75, 0.5]],
            np.float32,
        )
        self.anchor_points = anchors * np.asarray((COURT_LENGTH_M, COURT_WIDTH_M), np.float32)
        self.pending = None

    def reset(self) -> None:
        self.homography = None
        self.last_valid_frame = -(10**9)
        self.last_keypoint_frame = -(10**9)
        self.last_camera_motion_frame = -(10**9)
        self.last_stats = self._stats("uninitialized")
        self.pending = None

    @staticmethod
    def _stats(
        reason,
        valid=0,
        inliers=0,
        error=None,
        reacquired=False,
    ):
        return {
            "valid": int(valid),
            "inliers": int(inliers),
            "median_error": error,
            "reason": reason,
            "reacquired": bool(reacquired),
        }

    def _reject(self, reason, **updates):
        self.last_stats["reason"] = reason
        self.last_stats.update(updates)
        self.model_update_rejections[reason] = self.model_update_rejections.get(reason, 0) + 1
        return False

    @staticmethod
    def _project_court_to_image(homography, points):
        ok, inverse = cv2.invert(np.asarray(homography, dtype=np.float64))
        if not ok or not np.isfinite(inverse).all():
            return None
        projected = cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
        return projected if np.isfinite(projected).all() else None

    @classmethod
    def _is_direction_flip(cls, current_anchors, previous_anchors, frame_shape):
        visible = cls._visible_anchor_mask(current_anchors, previous_anchors, frame_shape)
        compared = visible if int(visible.sum()) >= 2 else np.ones(len(current_anchors), dtype=bool)
        direct_shift = float(
            np.median(np.linalg.norm(current_anchors[compared] - previous_anchors[compared], axis=1))
        )
        orders = ([1, 0, 3, 2, 4, 6, 5], [2, 3, 0, 1, 4, 6, 5])
        flipped_shifts = []
        for order in orders:
            flipped_anchors = previous_anchors[list(order)]
            flipped_visible = cls._visible_anchor_mask(current_anchors, flipped_anchors, frame_shape)
            flipped_compared = (
                flipped_visible
                if int(flipped_visible.sum()) >= 2
                else np.ones(len(current_anchors), dtype=bool)
            )
            flipped_shifts.append(
                float(
                    np.median(
                        np.linalg.norm(
                            current_anchors[flipped_compared] - flipped_anchors[flipped_compared],
                            axis=1,
                        )
                    )
                )
            )
        flipped_shift = min(flipped_shifts)
        return flipped_shift + 25.0 < direct_shift

    def _confirm_reacquire(self, anchors, frame_index, frame_shape):
        confirmations = 1
        if self.pending is not None:
            previous, previous_frame, previous_count = self.pending
            anchors = np.asarray(anchors, dtype=np.float32)
            visible = self._visible_anchor_mask(anchors, previous, frame_shape)
            compared = visible if int(visible.sum()) >= 2 else np.ones(len(anchors), dtype=bool)
            shift = np.linalg.norm(anchors[compared] - previous[compared], axis=1)
            scale = max(frame_shape[:2])
            consistent = frame_index - previous_frame <= self.max_pending_gap and np.isfinite(shift).all()
            consistent &= np.median(shift) <= max(
                32.0,
                0.04 * scale,
            )
            consistent &= np.percentile(shift, 90) <= max(
                80.0,
                0.1 * scale,
            )
            confirmations = previous_count + 1 if consistent else 1
        self.pending = (np.asarray(anchors, np.float32).copy(), int(frame_index), confirmations)
        confirmed = confirmations >= 3
        if confirmed:
            self.pending = None
        return confirmed

    @staticmethod
    def _visible_anchor_mask(current, previous, frame_shape):
        height, width = frame_shape[:2]
        margin_x = 0.05 * width
        margin_y = 0.05 * height

        def inside(points):
            return (
                (points[:, 0] >= -margin_x)
                & (points[:, 0] <= width + margin_x)
                & (points[:, 1] >= -margin_y)
                & (points[:, 1] <= height + margin_y)
            )

        return inside(np.asarray(current)) | inside(np.asarray(previous))

    def update(
        self,
        keypoints: np.ndarray | None,
        scores: np.ndarray | None,
        frame_index: int,
        frame_shape: tuple[int, ...],
        continuity_data: tuple[Any, ...] | list[Any] = (),
    ) -> bool:
        """Update the homography from one court-keypoint observation."""
        frame_index = int(frame_index)
        map_was_recent = self.is_recent(frame_index)
        self.model_update_attempts += 1
        self.last_stats = self._stats("no_input")
        if keypoints is None or scores is None:
            return self._reject("no_input")
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if (
            keypoints.shape != (COURT_KEYPOINT_COUNT, 2)
            or scores.shape != (COURT_KEYPOINT_COUNT,)
        ):
            return self._reject("invalid_input_shape")
        height, width = frame_shape[:2]
        valid = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores)
        valid &= scores >= 0.3
        valid &= (keypoints[:, 0] >= 0) & (keypoints[:, 0] < width)
        valid &= (keypoints[:, 1] >= 0) & (keypoints[:, 1] < height)
        image_points = keypoints[valid]
        court_points = self.vertices[valid]
        self.last_stats = self._stats("too_few_points", len(image_points))
        if len(image_points) < 6:
            return self._reject("too_few_points", valid=len(image_points))
        frame_area = max(1.0, float(width * height))
        if (
            abs(float(cv2.contourArea(cv2.convexHull(image_points))))
            < 0.001 * frame_area
        ):
            return self._reject("degenerate_image_layout")
        cv2.setRNGSeed(42)
        homography, mask = cv2.findHomography(
            image_points, court_points, cv2.RANSAC, 0.5
        )
        if homography is None or mask is None or not np.isfinite(homography).all():
            return self._reject("homography_failed")
        inliers = mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if (
            inlier_count < 6
            or inlier_count / len(image_points) < 0.7
        ):
            return self._reject("insufficient_consensus", inliers=inlier_count)
        inlier_points = image_points[inliers]
        spread = np.linalg.svd(inlier_points - inlier_points.mean(axis=0), compute_uv=False)
        inlier_area = abs(float(cv2.contourArea(cv2.convexHull(inlier_points))))
        if (
            inlier_area < 0.0005 * frame_area
            or spread[0] <= 1e-06
            or spread[1] / spread[0] < 0.025
        ):
            return self._reject("degenerate_inlier_layout")
        projected = cv2.perspectiveTransform(inlier_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        error = float(np.median(np.linalg.norm(projected - court_points[inliers], axis=1)))
        self.last_stats = self._stats("candidate", len(image_points), inlier_count, error)
        if not np.isfinite(error) or error > 0.5:
            return self._reject("large_reprojection_error")
        if abs(float(homography[2, 2])) > 1e-09:
            homography /= homography[2, 2]
        anchors = self._project_court_to_image(homography, self.anchor_points)
        if anchors is None or np.any(
            np.abs(anchors) > 8.0 * max(height, width)
        ):
            return self._reject("unstable_anchor_projection")
        keypoint_gap = frame_index - self.last_keypoint_frame
        reacquired = self.homography is None or not map_was_recent
        previous = (
            None if self.homography is None else self._project_court_to_image(self.homography, self.anchor_points)
        )
        if self.homography is not None and previous is None:
            self.pending = None
            return self._reject("invalid_previous_anchor_projection")
        if previous is not None and not reacquired and self._is_direction_flip(anchors, previous, frame_shape):
            self.pending = None
            return self._reject("direction_flip")
        if previous is not None and 0 <= keypoint_gap <= 5:
            visible = self._visible_anchor_mask(anchors, previous, frame_shape)
            compared = visible if int(visible.sum()) >= 2 else np.ones(len(anchors), dtype=bool)
            shift = np.linalg.norm(anchors[compared] - previous[compared], axis=1)
            limit = 25.0 * max(1, keypoint_gap)
            if (
                not np.isfinite(shift).all()
                or np.median(shift) > limit
                or np.percentile(shift, 90)
                > max(
                    4.0 * limit,
                    100.0,
                )
            ):
                self.pending = None
                return self._reject("temporal_jump")
        if reacquired and len(continuity_data) >= 3:
            continuity = np.asarray(continuity_data, dtype=np.float32).reshape(-1, 5)
            mapped = cv2.perspectiveTransform(continuity[:, :2].reshape(-1, 1, 2), homography).reshape(-1, 2)
            ratio = np.linalg.norm(mapped - continuity[:, 2:4], axis=1) / np.maximum(
                continuity[:, 4], 0.001
            )
            if (
                not np.isfinite(ratio).all()
                or np.median(ratio) > 1.0
            ):
                self.pending = None
                return self._reject("player_continuity")
        if (
            reacquired
            and self.homography is not None
            and not self._confirm_reacquire(anchors, frame_index, frame_shape)
        ):
            return self._reject("awaiting_reacquire")
        if not reacquired or self.homography is None:
            self.pending = None
        self.homography = homography.astype(np.float64)
        self.last_valid_frame = frame_index
        self.last_keypoint_frame = frame_index
        self.model_update_acceptances += 1
        self.last_stats.update(reason="accepted", reacquired=bool(reacquired))
        return True

    def update_camera_motion(
        self,
        warp: np.ndarray,
        source_frame: int | None,
        frame_index: int,
    ) -> bool:
        """Carry the court map through a validated previous-to-current affine warp."""
        frame_index = int(frame_index)
        if (
            self.homography is None
            or source_frame is None
            or int(source_frame) != self.last_valid_frame
            or frame_index <= self.last_camera_motion_frame
        ):
            return False
        warp = np.asarray(warp, dtype=np.float64)
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            return False
        image_warp = np.eye(3, dtype=np.float64)
        image_warp[:2] = warp
        if abs(float(np.linalg.det(image_warp[:2, :2]))) < 1e-6:
            return False
        try:
            homography = self.homography @ np.linalg.inv(image_warp)
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(homography).all() or abs(float(homography[2, 2])) < 1e-9:
            return False
        homography /= homography[2, 2]
        if self._project_court_to_image(homography, self.anchor_points) is None:
            return False
        self.homography = homography
        self.last_valid_frame = frame_index
        self.last_camera_motion_frame = frame_index
        if self.pending is not None:
            pending_anchors, _, confirmations = self.pending
            transformed = cv2.transform(
                np.asarray(pending_anchors, dtype=np.float32).reshape(1, -1, 2),
                warp.astype(np.float32),
            ).reshape(-1, 2)
            self.pending = (transformed, frame_index, confirmations)
        return True

    def is_recent(self, frame_index: int) -> bool:
        """Return whether the current court map has recent visual support."""
        age = int(frame_index) - self.last_valid_frame
        return (
            self.homography is not None
            and 0 <= age <= COURT_MAP_MAX_AGE_FRAMES
        )

    def transform(
        self,
        point: np.ndarray | None,
        frame_index: int,
        allow_stale: bool = False,
        clip: bool = True,
    ) -> np.ndarray | None:
        """Map an image point to court meters, optionally accepting stale maps."""
        if point is None or self.homography is None:
            return None
        max_age = COURT_MAP_MAX_AGE_FRAMES if allow_stale else 0
        age = int(frame_index) - self.last_valid_frame
        if age < 0 or age > max_age or not self.is_recent(frame_index):
            return None
        point = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        if not np.isfinite(point).all():
            return None
        mapped = cv2.perspectiveTransform(point, self.homography).reshape(2)
        lower = self.court_min - 2.0
        upper = self.court_max + 2.0
        if not np.isfinite(mapped).all() or np.any(mapped < lower) or np.any(mapped > upper):
            return None
        if clip:
            mapped = np.clip(mapped, self.court_min, self.court_max)
        return mapped.astype(np.float32)

def update_court_state(
    track: Any,
    state: Any,
    court_mapper: CourtMapper,
    frame_index: int,
    fps: float,
    pose: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Smooth a local track's court position while rejecting implausible jumps."""
    if not court_mapper.is_recent(frame_index):
        return
    observation = court_mapper.transform(get_player_ground_point(track, pose), frame_index, allow_stale=True)
    if observation is None:
        return
    if state.court_observation is None or state.court_position is None:
        position = observation.copy()
    else:
        elapsed = max((frame_index - state.court_frame) / fps, 1e-06)
        if float(np.linalg.norm(observation - state.court_observation) / elapsed) > 12.0:
            return
        position = (
            observation.copy()
            if elapsed > 1.0
            else 0.35 * observation
            + (1 - 0.35) * state.court_position
        )
    state.court_observation = observation.copy()
    state.court_position = position.astype(np.float32)
    state.court_frame = int(frame_index)


def is_player_on_court(
    track: Any,
    pose: tuple[np.ndarray, np.ndarray] | None,
    mapper: CourtMapper,
    frame_index: int,
    frame_shape: tuple[int, ...],
    min_fallback_height: float = 100,
) -> bool:
    """Return whether a local player track is eligible for on-court processing."""
    pose_data = None if pose is None else validated_pose(pose[0], pose[1], track.box)
    ankle_visible = pose_data is not None and bool(np.any(pose_data[2][15:17]))
    height = float(track.box[3] - track.box[1])
    if not ankle_visible and height < float(min_fallback_height):
        return False
    if mapper.is_recent(frame_index):
        court_point = mapper.transform(
            get_player_ground_point(track, pose),
            frame_index,
            allow_stale=True,
            clip=False,
        )
        margin = 0.75
        return court_point is not None and bool(
            np.all(court_point >= mapper.court_min - margin)
            and np.all(court_point <= mapper.court_max + margin)
        )
    x1, _, x2, y2 = map(float, track.box)
    height_px, width_px = frame_shape[:2]
    return (
        width_px * 0.11 <= (x1 + x2) * 0.5 <= width_px * 0.96
        and y2 >= height_px * 0.58
        and y2 <= height_px * 0.96
    )


__all__ = [
    "COURT_GEOMETRY",
    "COURT_KEYPOINT_COUNT",
    "COURT_LENGTH_M",
    "COURT_MAP_MAX_AGE_FRAMES",
    "COURT_PATHS",
    "COURT_WIDTH_M",
    "CourtMapper",
    "court_paths",
    "default_nba_court_vertices",
    "is_player_on_court",
    "update_court_state",
]
