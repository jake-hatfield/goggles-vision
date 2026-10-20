"""Serialize finalized tracking records as the public event-list schema."""

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from reid.consensus import LOCAL_TRACK_SEGMENT_MAX_GAP
from tracking.team import normalize_team

TrackObservation = tuple[int, Any, str | None, str | None]
TrackObservations = dict[str, list[TrackObservation]]


def _event_location(event: dict[str, Any], frame_record: dict[str, Any]) -> list[float] | None:
    location = event.get("location")
    if location is None:
        player_ids = (
            event.get("player_id"),
            event.get("receiver_id"),
            event.get("gained_by"),
            event.get("turnover_player_id"),
        )
        for player_id in player_ids:
            if player_id is None:
                continue
            player = next(
                (
                    item
                    for item in frame_record.get("players", ())
                    if str(item.get("id")) == str(player_id)
                ),
                None,
            )
            if player is not None and player.get("court_xy") is not None:
                location = player["court_xy"]
                break
    if location is None or len(location) < 2:
        return None
    return [round(float(value), 3) for value in location[:2]]


def _event_track_metadata(
    event: dict[str, Any],
    frame_record: dict[str, Any],
    track_observations: TrackObservations,
) -> tuple[str | None, str | None]:
    local_track_id = event.get("local_track_id")
    if local_track_id is None:
        local_track_id = (event.get("scorer_tag") or {}).get("local_track_id")
    if local_track_id is None:
        return None, None

    fallback_frame = frame_record.get("frame")
    start_frame = event.get("start_frame")
    end_frame = event.get("end_frame")
    if start_frame is None:
        start_frame = end_frame if end_frame is not None else fallback_frame
    if end_frame is None:
        end_frame = fallback_frame if fallback_frame is not None else start_frame
    if start_frame is None or end_frame is None:
        return None, None

    first_frame, last_frame = sorted((int(start_frame), int(end_frame)))
    observations = [
        observation
        for observation in track_observations.get(str(local_track_id), ())
        if first_frame <= observation[0] <= last_frame
    ]
    jersey_values = [
        observation[2] for observation in observations if observation[2] is not None
    ]
    team_values = [
        observation[3] for observation in observations if observation[3] is not None
    ]
    jersey_number = _majority_observation_value(jersey_values)
    team = _majority_observation_value(team_values)

    if jersey_number is None or team is None:
        episode = _event_track_episode(
            track_observations.get(str(local_track_id), ()), first_frame, last_frame
        )
        if jersey_number is None:
            jersey_number = _unambiguous_observation_value(
                observation[2] for observation in episode
            )
        if team is None:
            team = _unambiguous_observation_value(
                observation[3] for observation in episode
            )
    return jersey_number, team


def _event_track_episode(
    observations: Iterable[TrackObservation],
    first_frame: int,
    last_frame: int,
) -> list[TrackObservation]:
    episodes = []
    current = []
    for observation in sorted(observations, key=lambda item: item[0]):
        if current:
            previous = current[-1]
            identity_changed = (
                previous[1] is not None
                and observation[1] is not None
                and previous[1] != observation[1]
            )
            if (
                observation[0] - previous[0] > LOCAL_TRACK_SEGMENT_MAX_GAP
                or identity_changed
            ):
                episodes.append(current)
                current = []
        current.append(observation)
    if current:
        episodes.append(current)

    overlapping = [
        episode
        for episode in episodes
        if episode[0][0] <= last_frame and episode[-1][0] >= first_frame
    ]
    return overlapping[0] if len(overlapping) == 1 else []


def _majority_observation_value(values: Iterable[str | None]) -> str | None:
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None
    value, support = counts.most_common(1)[0]
    if support < 2 or support * 2 <= sum(counts.values()):
        return None
    return value


def _unambiguous_observation_value(values: Iterable[str | None]) -> str | None:
    distinct_values = {value for value in values if value is not None}
    return next(iter(distinct_values)) if len(distinct_values) == 1 else None


def _event_list_item(
    event: dict[str, Any],
    frame_record: dict[str, Any],
    track_observations: TrackObservations,
    event_id: int,
) -> dict[str, Any]:
    track_jersey_number, track_team = _event_track_metadata(
        event, frame_record, track_observations
    )
    item = {
        "id": str(event_id),
        "type": "shot",
    }
    jersey_number = track_jersey_number
    if jersey_number is not None:
        item["jersey_number"] = jersey_number
    team = normalize_team(event.get("team"))
    if team is None:
        team = track_team
    if team is not None:
        item["team"] = team

    for field in ("start", "end", "start_frame", "end_frame"):
        value = event.get(field)
        if value is not None:
            item[field] = value
    location = _event_location(event, frame_record)
    if location is not None:
        item["location"] = location

    result = event.get("result")
    if result == "miss":
        result = "missed"
    if result is not None:
        item["result"] = result

    attrs = {}
    if event.get("result") == "made" and event.get("points") is not None:
        attrs["points"] = event["points"]
    elif event.get("shot_value") is not None:
        attrs["shot_value"] = event["shot_value"]
    if event.get("result") == "made" and event.get("assister") is not None:
        attrs["assister"] = event["assister"]
    if attrs:
        item["attrs"] = attrs
    return item


def write_event_list(tracks_path: Path, event_list_path: Path) -> int:
    """Write finalized shot events as a compact v2 event list."""
    track_observations: TrackObservations = defaultdict(list)
    events = []
    with open(tracks_path, encoding="utf-8") as track_file:
        for line in track_file:
            record = json.loads(line)
            frame_index = record.get("frame")
            for player in record.get("players", ()):
                player_id = player.get("id")
                jersey_number = player.get("jersey_number")
                local_track_id = player.get("local_track_id")
                if local_track_id is not None and frame_index is not None:
                    number = None if jersey_number is None else str(jersey_number).strip() or None
                    team = normalize_team(player.get("team_semantic")) or normalize_team(
                        player.get("team")
                    )
                    track_observations[str(local_track_id)].append(
                        (int(frame_index), player_id, number, team)
                    )
            events.extend(
                (event, record)
                for event in record.get("events") or ()
                if event.get("event") == "shot"
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
