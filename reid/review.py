"""Apply reviewed identity assignments to observed local-track intervals."""

import json
from pathlib import Path

from tracking.team import normalize_team


def load_identity_review(path: Path) -> dict:
    """Read a clip-specific review, validating its frame and identity fields."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("frame_count")) is not int or data["frame_count"] < 1:
        raise ValueError("Identity review requires a positive frame_count")
    edits = data.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("Identity review requires at least one edit")
    for edit in edits:
        if not isinstance(edit, dict) or not {"id", "team", "jersey_number", "segments"} <= edit.keys():
            raise ValueError("Each identity edit requires id, team, jersey_number, and segments")
        identity, team, jersey = edit["id"], edit["team"], edit["jersey_number"]
        if identity is not None and (type(identity) is not int or identity < 1):
            raise ValueError("Identity id must be a positive integer or null")
        if team is not None and team not in ("dark", "light"):
            raise ValueError("Identity team must be dark, light, or null")
        if jersey is not None and (not isinstance(jersey, str) or not jersey.isascii() or not jersey.isdigit() or len(jersey) > 2):
            raise ValueError("Jersey number must be a one- or two-digit string or null")
        segments = edit["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError("Each identity edit requires at least one segment")
        for segment in segments:
            if (
                not isinstance(segment, list) or len(segment) != 3
                or any(type(value) is not int for value in segment)
                or segment[0] < 1 or not 1 <= segment[1] <= segment[2] <= data["frame_count"]
            ):
                raise ValueError("Segments must be [local_track_id, first_frame, last_frame] within the clip")
    return data


def apply_identity_review(records: list[dict], review: dict, *, automatic=False) -> dict[int, dict[int, int | None]]:
    """Set reviewed fields and return the global IDs to remap at each frame."""
    if len(records) != review["frame_count"]:
        raise ValueError("Identity review frame_count does not match the analyzed clip")
    players_by_frame = {r["frame"]: {p["local_track_id"]: p for p in r["players"]} for r in records}
    edits_by_frame = {}
    for edit in review["edits"]:
        for local_id, first, last in edit["segments"]:
            found = False
            for frame in range(first, last + 1):
                player = players_by_frame[frame].get(local_id)
                if player is None or not player.get("reid_eligible"):
                    continue
                found = True
                edits = edits_by_frame.setdefault(frame, {})
                if local_id in edits:
                    raise ValueError(f"Identity review edits track {local_id} twice at frame {frame}")
                edits[local_id] = edit
            if not found:
                raise ValueError(f"Identity review found no eligible observations for track {local_id} in {first}–{last}")

    for frame, edits in edits_by_frame.items():
        ids = [edits[local]["id"] if local in edits else player.get("id")
               for local, player in players_by_frame[frame].items()]
        assigned = [value for value in ids if value is not None]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"Identity review would assign the same global ID twice at frame {frame}")

    remaps = {}
    for frame, edits in edits_by_frame.items():
        remaps[frame] = {}
        for local, edit in edits.items():
            player = players_by_frame[frame][local]
            identity, team, jersey = edit["id"], edit["team"], edit["jersey_number"]
            previous = player.get("id")
            if previous is not None and previous != identity:
                remaps[frame][previous] = identity
            player.update(
                id=identity, team=None if team is None else int(team == "light"),
                team_semantic=normalize_team(team), jersey_number=jersey,
                jersey_number_candidate=jersey, jersey_number_locked=jersey is not None,
                identity_label=f"R{local}" if identity is None else f"G{identity}",
            )
            player["identity_stitched" if automatic else "identity_reviewed"] = True
    return remaps


def remap_reviewed_references(value: dict | list, remaps: dict, frame: int) -> None:
    """Update recorded IDs where the review identifies the old ID at the reference frame."""
    if isinstance(value, list):
        for item in value:
            remap_reviewed_references(item, remaps, frame)
    elif isinstance(value, dict):
        frame = value.get("frame") or frame
        start = value.get("start_frame") or frame
        end = value.get("end_frame") or frame
        for key, item in value.items():
            if key in ("player_id", "receiver_id", "gained_by", "holder_id", "turnover_player_id") and item is not None:
                at = end if key in ("receiver_id", "gained_by") else start
                value[key] = remaps.get(at, {}).get(item, item)
            elif key != "players" and isinstance(item, (dict, list)):
                remap_reviewed_references(item, remaps, end if key == "scorer_tag" else frame)
        if "local_track_id" in value and "overlap" in value and "id" in value:
            old_id = value["id"]
            value["id"] = remaps.get(frame, {}).get(old_id, old_id)
