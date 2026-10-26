"""Match local tracklets against the global identity gallery."""

from typing import Any

import numpy as np

from reid.jersey import get_locked_jersey_number, get_trusted_jersey_key


def match_new_tracks(
    resolver: Any,
    rows: list[dict[str, Any]],
    frame_index: int,
    active_local_ids: set[int] | None = None,
) -> tuple[set[int], set[int]]:
    """Match new local tracklets to available global identities by evidence."""
    if not rows:
        return set(), set()
    active_local_ids = {int(local_id) for local_id in (active_local_ids or ())}
    gallery = []
    active_identity_indices = set()
    for global_id in resolver.feature_gallery_ids:
        identity = resolver.feature_gallery.get(global_id)
        if identity is None:
            continue
        if identity.active_local_id is not None and identity.active_local_id in active_local_ids:
            active_identity_indices.add(len(gallery))
        gallery.append(identity)
    if not gallery:
        return set(), set()
    team_gate = bool(resolver.parameters.get('team_gate', True))
    score_matrix = np.full((len(rows), len(gallery)), -1.0e6, dtype=np.float32)
    default_threshold = float(resolver.parameters.get("similarity_threshold", 0.80))
    threshold_matrix = np.full((len(rows), len(gallery)), default_threshold, dtype=np.float32)
    identity_teams = []
    for identity in gallery:
        identity_teams.append(resolver.get_confident_team(identity.team_counts))
    reid_similarity_floor = float(resolver.parameters.get("reid_similarity_floor", 0.76))
    uncertain_team_threshold = float(resolver.parameters.get("uncertain_team_similarity_threshold", 0.84))
    for row_index, row in enumerate(rows):
        local_id = int(row["track"].track_id)
        local_state = resolver.local_tracks[local_id]
        track_jersey = get_trusted_jersey_key(local_state)
        # A team lock can survive a local identity reset while its vote counts do not.
        track_team = (
            track_jersey[0]
            if track_jersey is not None
            else resolver.team_classifier.predict_identity_track(local_id)
        )
        if track_team is None:
            track_team = resolver.get_confident_team(local_state.team_counts)
        track_birth_frame = getattr(local_state, "reid_start_frame", None)
        if track_birth_frame is None:
            track_birth_frame = resolver.local_track_birth_frames.get(local_id, int(local_state.created_frame))
        previous_segments = getattr(resolver, "local_identity_segments", {}).get(local_id, ())
        canonical_global_id = resolver.canonicalize_global_id
        current_global_id = canonical_global_id(resolver.local_to_global.get(local_id))
        current_identity = resolver.identities.get(current_global_id)
        current_identity_numbers = set()
        if current_identity is not None:
            current_identity_numbers.update(
                str(number)
                for number, count in current_identity.jersey_number_counts.items()
                if count > 0
            )
            current_identity_numbers.update(
                str(number)
                for (_, number), count in current_identity.jersey_counts.items()
                if count > 0
            )
        previous_global_ids = {
            canonical_global_id(int(global_id))
            for _, global_id in previous_segments
            if global_id is not None
            and canonical_global_id(int(global_id)) != canonical_global_id(current_global_id)
        }
        pending_split = bool(local_state.identity_split_pending)
        track_number = get_locked_jersey_number(local_state)
        for identity_index, identity in enumerate(gallery):
            # McByte can keep a missed local track active in its buffer.
            # Only reject a merge when both tracklets were actually
            # observed in overlapping frames, not while one is merely
            # waiting in that buffer.
            if int(track_birth_frame) <= int(identity.last_seen_frame):
                continue
            same_local_identity = canonical_global_id(identity.global_id) in previous_global_ids
            if same_local_identity and pending_split:
                continue
            identity_team = identity_teams[identity_index]
            threshold = float(threshold_matrix[row_index, identity_index])
            late_jersey_key = row.get("late_jersey_key")
            identity_jerseys = {
                key for key, count in identity.jersey_counts.items() if count > 0
            }
            identity_numbers = {
                number
                for number, count in identity.jersey_number_counts.items()
                if count > 0
            }
            identity_numbers.update(number for _, number in identity_jerseys)
            exact_jersey_match = track_jersey is not None and track_jersey in identity_jerseys
            identity_numbers = {str(number) for number in identity_numbers}
            reliable_jersey_match = (
                track_number is not None
                and track_number in identity_numbers
                and track_team is not None
                and track_team == identity_team
            )
            if len(identity_jerseys) > 1 or len(identity_numbers) > 1:
                continue
            if track_number is not None and identity_numbers and track_number not in identity_numbers:
                continue
            if track_jersey is not None and identity_jerseys and not exact_jersey_match:
                continue
            target_global_id = canonical_global_id(identity.global_id)
            if (
                current_global_id is not None
                and current_global_id != target_global_id
                and not pending_split
                and (
                    (
                        current_identity_numbers
                        and identity_numbers
                        and current_identity_numbers != identity_numbers
                    )
                    or (
                        bool(current_identity_numbers) != bool(identity_numbers)
                        and not (exact_jersey_match or reliable_jersey_match)
                    )
                )
            ):
                continue
            if exact_jersey_match:
                identity_team = track_jersey[0]
            elif same_local_identity:
                threshold = max(threshold, 0.84)
                threshold_matrix[row_index, identity_index] = threshold
            if late_jersey_key is not None:
                if track_jersey != late_jersey_key:
                    continue
                if identity_jerseys and not exact_jersey_match:
                    continue
            if team_gate and track_team is not None and identity_team is not None:
                if track_team != identity_team:
                    continue
            elif team_gate and not exact_jersey_match and (track_team is not None or identity_team is not None):
                threshold = max(threshold, uncertain_team_threshold)
                threshold_matrix[row_index, identity_index] = threshold
            # Cache the embedding score while still checking track evidence each frame.
            cached = local_state.match_similarity_cache.get(identity.global_id)
            query_version = int(local_state.feature_count)
            version_key = (query_version, identity.feature_version)
            if cached is not None and cached[0] == version_key:
                similarity = cached[1]
            else:
                query_feature = row.get("query_feature", row["feature"])
                similarity = resolver.compute_similarity(query_feature, identity)
                local_state.match_similarity_cache[identity.global_id] = (version_key, similarity)
            if similarity is not None:
                if similarity < reid_similarity_floor:
                    continue
                score_matrix[row_index, identity_index] = similarity
    viable = score_matrix >= threshold_matrix
    viable_local_ids = {
        int(rows[row_index]["track"].track_id)
        for row_index in range(len(rows))
        if np.any(viable[row_index])
    }
    assignment_matrix = np.where(viable, score_matrix, -1.0e6)
    if active_identity_indices:
        assignment_matrix[:, list(active_identity_indices)] = -1.0e6
    if not np.any(assignment_matrix >= threshold_matrix):
        return set(), viable_local_ids
    best_identity_per_track = np.argmax(assignment_matrix, axis=1)
    best_track_per_identity = np.argmax(assignment_matrix, axis=0)
    matched_local_ids = set()
    for row_index, identity_index in enumerate(best_identity_per_track):
        best_score = float(assignment_matrix[row_index, identity_index])
        threshold = float(threshold_matrix[row_index, identity_index])
        if best_score < threshold or best_track_per_identity[identity_index] != row_index:
            continue
        row = rows[row_index]
        local_id = int(row["track"].track_id)
        identity = gallery[identity_index]
        previous_global_id = resolver.canonicalize_global_id(
            resolver.local_to_global.get(local_id)
        )
        local_state = resolver.local_tracks[local_id]
        track_jersey = get_trusted_jersey_key(local_state)
        track_number = get_locked_jersey_number(local_state)
        identity_jerseys = {
            key for key, count in identity.jersey_counts.items() if count > 0
        }
        identity_numbers = {
            str(number)
            for number, count in identity.jersey_number_counts.items()
            if count > 0
        }
        identity_numbers.update(str(number) for _, number in identity_jerseys)
        exact_jersey_match = track_jersey is not None and track_jersey in identity_jerseys
        track_team = (
            track_jersey[0]
            if track_jersey is not None
            else resolver.team_classifier.predict_identity_track(local_id)
        )
        if track_team is None:
            track_team = resolver.get_confident_team(local_state.team_counts)
        identity_team = resolver.get_confident_team(identity.team_counts)
        reliable_jersey_match = (
            track_number is not None
            and track_number in identity_numbers
            and track_team is not None
            and track_team == identity_team
        )
        if previous_global_id is not None and previous_global_id != identity.global_id:
            previous_identity = resolver.identities.get(previous_global_id)
            if not local_state.identity_split_pending:
                if not resolver.alias_global_identity(
                    previous_global_id,
                    identity.global_id,
                    allow_unlabeled_merge=exact_jersey_match or reliable_jersey_match,
                ):
                    continue
                if previous_identity is not None and previous_identity.active_local_id == local_id:
                    resolver.feature_gallery.pop(previous_global_id, None)
                    if previous_global_id in resolver.feature_gallery_ids:
                        resolver.feature_gallery_ids.remove(previous_global_id)
        previous_local_id = identity.active_local_id
        if previous_local_id is not None and previous_local_id != local_id:
            resolver.local_to_global.pop(int(previous_local_id), None)
            resolver.local_tracks.pop(int(previous_local_id), None)
        identity.active_local_id = local_id
        assignment_frame = (
            local_state.created_frame
            if local_state.identity_split_pending
            else frame_index
        )
        resolver.assign_local_identity(local_id, identity.global_id, assignment_frame)
        local_state.identity_split_pending = False
        matched_local_ids.add(local_id)
        for feature in row.get("track_features", [row["feature"]]):
            resolver.append_feature(identity, feature)
        resolver.local_tracks[local_id].identity_feature_count = len(resolver.local_tracks[local_id].features or [])
        for team_id, count in (resolver.local_tracks[local_id].team_counts or {}).items():
            identity.team_counts[int(team_id)] = identity.team_counts.get(int(team_id), 0) + int(count)
        resolver.match_count += 1
    return matched_local_ids, viable_local_ids - matched_local_ids
