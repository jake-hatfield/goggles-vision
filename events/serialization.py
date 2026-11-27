"""Serialize finalized tracking records as the public event-list schema."""

import json
import math
import os
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, NamedTuple

from events.context import nearest_wrist_distance, pose_wrists
from events.shot_actor import ObservedBall, shot_release_actor
from reid.consensus import LOCAL_TRACK_SEGMENT_MAX_GAP
from tracking.ball import normalize_fps
from tracking.team import normalize_team


class TrackObservation(NamedTuple):
    frame: int
    global_id: int | None
    local_id: int
    jersey_number: str | None
    team: str | None
    court_xy: list[float] | None
    ball_distance: float | None
    wrist_distance: float | None
    height: float
    ball_offset: tuple[float, float] | None
    jersey_contested: bool = False


class TrackObservations(NamedTuple):
    by_local: dict[int, list[TrackObservation]]
    by_global: dict[int, list[TrackObservation]]
    by_frame: dict[int, list[TrackObservation]]
    ball_frames: dict[int, ObservedBall]


_EVENT_FIELDS = (
    "start", "end", "start_frame", "end_frame", "completed_frame", "actor_frame",
)
_observation_frame = attrgetter("frame")


def _event_location(event: dict[str, Any], actor: TrackObservation | None) -> list[float] | None:
    location = event.get("location")
    if location is None and actor is not None:
        location = actor.court_xy
    if location is None or len(location) < 2:
        return None
    return [round(float(value), 3) for value in location[:2]]


def _observations_between(observations: list[TrackObservation], first: int, last: int) -> list[TrackObservation]:
    start_index = bisect_left(observations, first, key=_observation_frame)
    end_index = bisect_right(observations, last, key=_observation_frame)
    return observations[start_index:end_index]


def _ball_distance_to_player(box: list[float] | None, center: list[float] | None) -> float | None:
    if not box or not center:
        return None
    x1, y1, x2, y2 = box
    if y2 <= y1:
        return None
    x, y = center
    return math.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)) / (y2 - y1)


def _shot_actor_contradicted(actor: TrackObservation, tracks: TrackObservations, fps: float) -> bool:
    # Ignore normal release separation. Check for a player who stays well
    # outside the observed ball's path even in the lead-up to a late shot trigger.
    local_rows = tracks.by_local[actor.local_id]
    after_release = actor.frame + max(3, round(.17 * fps))
    window = _observations_between(local_rows, actor.frame - round(1.5 * fps), after_release)
    visible = [row for row in window if row.global_id == actor.global_id and row.ball_distance is not None]
    if (
        actor.ball_distance is not None and actor.ball_distance > 3
        and len(visible) >= max(5, round(.15 * fps))
        and sum(row.frame <= actor.frame for row in visible) >= 2
        and all(row.ball_distance > 1.25 for row in visible)
    ):
        return True

    # Repeated close contact by an opponent can rule out a nearby credited
    # player when the credited player's own wrists never approach the ball.
    if actor.team is None or any(row.wrist_distance is not None and row.wrist_distance <= .30 for row in visible):
        return False
    opposing_contact = Counter()
    for row in visible:
        if row.frame < actor.frame - round(.12 * fps) or row.wrist_distance is None:
            continue
        for other in tracks.by_frame[row.frame]:
            if (
                other.team is not None and other.team != actor.team
                and other.wrist_distance is not None and other.wrist_distance <= .18
                and row.wrist_distance - other.wrist_distance >= .20
            ):
                opposing_contact[other.local_id] += 1
    return max(opposing_contact.values(), default=0) >= 3


def _recorded_actor(
    event: dict[str, Any],
    frame_record: dict[str, Any],
    track_observations: TrackObservations,
    fps: float,
) -> TrackObservation | None:
    """Resolve the finalized actor of the action, using a short window for old exports."""
    local_track_id = event.get("local_track_id")
    global_id = event.get("player_id")
    if local_track_id is None and global_id is None:
        return None
    anchor = event.get("actor_frame")
    reference = anchor or event.get("start_frame") or event.get("end_frame") or frame_record.get("frame")
    if reference is None:
        return None
    reference = int(reference)
    if local_track_id is not None:
        candidates = track_observations.by_local.get(int(local_track_id), [])
    else:
        candidates = track_observations.by_global.get(int(global_id), [])

    # New events identify the exact action frame. For old passes and turnovers,
    # search back toward release rather than forward into the receiver's catch.
    before = 0 if anchor is not None else max(2, round(.35 * fps))
    after = 0 if anchor is not None or event.get("event") in ("pass", "turnover") else max(1, round(.08 * fps))
    near = _observations_between(candidates, reference - before, reference + after)
    if not near:
        return None
    if local_track_id is None and len({row.local_id for row in near}) != 1:
        return None
    actor = min(near, key=lambda row: (abs(row.frame - reference), row.frame))
    if actor.global_id is None:
        return None
    if global_id is not None and actor.global_id != int(global_id):
        previous = track_observations.by_global.get(int(global_id), [])
        if anchor is None or _observations_between(previous, reference, reference):
            return None

    if event.get("event") == "shot" and _shot_actor_contradicted(actor, track_observations, fps):
        return None
    return actor


def _event_actor(
    event: dict[str, Any], frame_record: dict[str, Any], track_observations: TrackObservations,
) -> TrackObservation | None:
    fps = normalize_fps(frame_record.get("fps"))
    start_frame = event.get("actor_frame") or event.get("start_frame")
    actor = _recorded_actor(event, frame_record, track_observations, fps)
    release = None
    if event.get("event") == "shot" and start_frame is not None:
        contact = shot_release_actor(
            int(start_frame), fps, track_observations.by_frame, track_observations.ball_frames,
        )
        # Nearby hands can belong to a defender. Replace a recorded shooter only
        # after the independent checks above have ruled out that attribution.
        if contact is not None and (actor is None or contact.local_id == actor.local_id):
            actor = release = contact
    if actor is None and event.get("event") == "shot" and start_frame is not None:
        release = shot_release_actor(
            int(start_frame), fps, track_observations.by_frame, track_observations.ball_frames, use_boxes=True,
        )
        actor = release
    if actor is None:
        return None
    if actor.global_id is None:
        return actor._replace(jersey_number=None)

    # Keep votes within this continuous stretch of the same player's track.
    local_rows = track_observations.by_local[actor.local_id]
    radius = max(3, round(.3 * fps))
    nearby = _observations_between(local_rows, actor.frame - radius, actor.frame + radius)
    center = bisect_left(nearby, actor.frame, key=_observation_frame)
    first, last = center, center
    while first > 0 and _same_track_segment(nearby[first - 1], nearby[first]):
        first -= 1
    while last + 1 < len(nearby) and _same_track_segment(nearby[last], nearby[last + 1]):
        last += 1
    sequence = nearby[first:last + 1]
    teams = {row.team for row in sequence if row.team is not None}
    event_team = normalize_team(event.get("team"))
    if len(teams) > 1 or (event_team is not None and teams and event_team not in teams):
        if release is not None:
            return actor._replace(jersey_number=None)
        return None
    team = next(iter(teams), None)
    numbers = [row.jersey_number for row in sequence if row.jersey_number is not None]
    number = numbers[0] if team and len(numbers) >= 3 and len(set(numbers)) == 1 else None
    if number is not None:
        competing_numbers = Counter(
            row.jersey_number for row in nearby[:center]
            if row.jersey_number is not None and row.jersey_number != number
        )
        if max(competing_numbers.values(), default=0) >= 2 or sum(
            row.jersey_contested for row in sequence if row.jersey_number == number
        ) >= 2:
            number = None
    return actor._replace(jersey_number=number, team=team)


def _same_track_segment(left: TrackObservation, right: TrackObservation) -> bool:
    return left.global_id == right.global_id and right.frame - left.frame <= LOCAL_TRACK_SEGMENT_MAX_GAP


def _event_list_item(
    event: dict[str, Any],
    frame_record: dict[str, Any],
    track_observations: TrackObservations,
    event_id: int,
) -> dict[str, Any]:
    actor = _event_actor(event, frame_record, track_observations)
    item = {
        "id": str(event_id),
        "type": event["event"],
    }
    if actor is not None:
        if actor.global_id is not None:
            item["player_id"] = actor.global_id
        item["local_track_id"] = actor.local_id
        if actor.team is not None:
            item["actor_team"] = actor.team
        if actor.jersey_number is not None:
            item["jersey_number"] = actor.jersey_number
    team = normalize_team(event.get("team"))
    if team is None and actor is not None:
        team = actor.team
    if team is not None:
        item["team"] = team

    for field in _EVENT_FIELDS:
        value = event.get(field)
        if value is not None:
            item[field] = value
    if actor is not None:
        item["actor_frame"] = actor.frame
    location = _event_location(event, actor)
    if location is not None:
        item["location"] = location

    result = event.get("result")
    if item["type"] == "shot" and result == "miss":
        result = "missed"
    if result is not None:
        item["result"] = result

    if item["type"] == "shot":
        attrs = {}
        if event.get("shot_id") is not None:
            attrs["shot_id"] = event["shot_id"]
        if event.get("result") == "made" and event.get("points") is not None:
            attrs["points"] = event["points"]
        elif event.get("shot_value") is not None:
            attrs["shot_value"] = event["shot_value"]
        if event.get("result") == "made" and event.get("assister") is not None:
            attrs["assister"] = event["assister"]
    else:
        common_fields = {"event", "team", "location", "result", "player_id", "local_track_id", *_EVENT_FIELDS}
        attrs = {
            key: value for key, value in event.items()
            if key not in common_fields and value is not None
        }
    if attrs:
        item["attrs"] = attrs
    return item


def write_event_list(tracks_path: Path, event_list_path: Path) -> int:
    """Write finalized events as a compact v2 event list."""
    track_observations = TrackObservations(defaultdict(list), defaultdict(list), {}, {})
    events = []
    with open(tracks_path, encoding="utf-8") as track_file:
        for line in track_file:
            record = json.loads(line)
            frame_index = record.get("frame")
            ball = record.get("ball") or {}
            ball_center = ball.get("center") if ball.get("observed", True) and not ball.get("interpolated") else None
            if ball_center and frame_index is not None:
                rim = record.get("rim") or {}
                track_observations.ball_frames[int(frame_index)] = ObservedBall(
                    *ball_center, ball.get("id"), rim.get("center"),
                )
            observations = []
            for player in record.get("players", ()):
                player_id = None if player.get("id") is None else int(player["id"])
                jersey_number = player.get("jersey_number")
                local_track_id = player.get("local_track_id")
                if local_track_id is not None and frame_index is not None:
                    number = None if jersey_number is None else str(jersey_number).strip() or None
                    if player.get("reid_eligible") is False:
                        player_id = None
                        number = None
                    elif not player.get("jersey_number_locked", number is not None):
                        number = None
                    team = normalize_team(player.get("team_semantic")) or normalize_team(
                        player.get("team")
                    )
                    box = player.get("box")
                    distance = _ball_distance_to_player(box, ball_center)
                    wrists = pose_wrists(player.get("pose")) if distance is not None and distance <= 1.5 else None
                    height = max(box[3] - box[1], 1.0) if box else 1.0
                    wrist_distance = (
                        nearest_wrist_distance(ball_center, wrists) / height if wrists else None
                    )
                    ball_offset = (
                        ((ball_center[0] - (box[0] + box[2]) / 2) / height, (ball_center[1] - box[1]) / height)
                        if distance is not None and distance <= .65 else None
                    )
                    observations.append(TrackObservation(
                        int(frame_index), player_id, int(local_track_id), number, team, player.get("court_xy"),
                        distance, wrist_distance, height, ball_offset,
                    ))
            jersey_owners = defaultdict(set)
            for row in observations:
                if row.global_id is not None and row.team is not None and row.jersey_number is not None:
                    jersey_owners[row.team, row.jersey_number].add(row.global_id)
            for index, row in enumerate(observations):
                if len(jersey_owners.get((row.team, row.jersey_number), ())) > 1:
                    row = observations[index] = row._replace(jersey_contested=True)
                track_observations.by_local[row.local_id].append(row)
                if row.global_id is not None:
                    track_observations.by_global[row.global_id].append(row)
            if frame_index is not None:
                track_observations.by_frame[int(frame_index)] = observations
            events.extend(
                (event, record)
                for event in record.get("events") or ()
                if event.get("event")
            )

    events.sort(
        key=lambda item: (
            item[0].get("start_frame") or item[0].get("end_frame") or 0,
            item[0].get("end_frame") or 0,
        )
    )
    event_list = [
        _event_list_item(
            event,
            record,
            track_observations,
            event_id,
        )
        for event_id, (event, record) in enumerate(events, start=1)
    ]
    temporary_path = event_list_path.with_name(event_list_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as destination:
            json.dump(
                {"schema_version": "v2", "events": event_list},
                destination,
                ensure_ascii=False,
                indent=2,
            )
            destination.write("\n")
        os.replace(temporary_path, event_list_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(event_list)


__all__ = ["write_event_list"]
