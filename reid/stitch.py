"""Reattach observed tracklets to well-supported team and jersey identities."""

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from reid.consensus import LOCAL_TRACK_SEGMENT_MAX_GAP
from reid.crops import jersey_torso_crop
from reid.wardrobe import sample_wardrobe, wardrobe_handoffs
from tracking.team import JerseyTeamClassifier


def _observation_teams(players):
    samples = {ref: (p["jersey_l"], p["jersey_s"]) for ref, p in players.items()
               if p.get("jersey_l") is not None and p.get("jersey_s") is not None}
    profiles = np.asarray(list(samples.values()), dtype=np.float32).reshape(-1, 2)
    if len(profiles) > 1024:
        profiles = profiles[np.linspace(0, len(profiles) - 1, 1024, dtype=int)]
    centers = JerseyTeamClassifier._fit_identity_color_centers(profiles)
    if centers is None:
        return {}
    separation = np.linalg.norm(centers[0] - centers[1])
    teams = {}
    for ref, features in samples.items():
        distances = np.linalg.norm(centers - features, axis=1)
        winner = int(np.argmin(distances))
        if distances[winner] <= .65 * separation and abs(distances[0] - distances[1]) >= .25 * separation:
            teams[ref] = winner
    return teams


def _tracklets(players, teams):
    by_track = defaultdict(list)
    for ref in players:
        by_track[ref[1]].append(ref)
    for refs in by_track.values():
        start = 0
        team = None
        for index, ref in enumerate(refs):
            current = teams.get(ref)
            gap = index and ref[0] - refs[index - 1][0] > LOCAL_TRACK_SEGMENT_MAX_GAP
            changed = (
                team is not None and current is not None and current != team
                and sum(teams.get(item) == current for item in refs[index:index + 4]) >= 3
            )
            if index and (gap or changed):
                yield team, refs[start:index]
                start, team = index, None
            if team is None:
                team = current
        yield team, refs[start:]


def _sample_tracklet_appearance(source_video, players, reader, predictions, stride):
    """Read extra jersey and wardrobe cues, including withheld identity rows."""
    by_frame = defaultdict(list)
    for ref in players:
        by_frame[ref[0]].append(ref)
    capture = cv2.VideoCapture(str(source_video))
    crops, batch_refs = [], []
    wardrobe = {}

    def flush():
        if crops:
            reads = reader.predict(crops)
            for ref, prediction in zip(batch_refs, reads):
                if prediction.get("confidence", 0) > predictions.get(ref, {}).get("confidence", 0):
                    predictions[ref] = prediction
            crops.clear()
            batch_refs.clear()

    try:
        for frame in range(1, max(by_frame) + 1):
            if not capture.grab():
                break
            if (frame - 1) % stride or frame not in by_frame:
                continue
            ok, image = capture.retrieve()
            if not ok:
                continue
            for ref in by_frame[frame]:
                player = players[ref]
                pose = player.get("pose")
                if pose is not None:
                    pose = np.asarray(pose, dtype=np.float32)
                    pose = (pose[:, :2], pose[:, 2])
                appearance = sample_wardrobe(image, player["box"], pose)
                if appearance is not None:
                    wardrobe[ref] = appearance
                if predictions.get(ref, {}).get("confidence", 0) >= .90:
                    continue
                crop = jersey_torso_crop(image, player["box"], pose, reader.parameters)
                if crop is not None:
                    crops.append({"full": crop["full"].copy()})
                    batch_refs.append(ref)
                if len(crops) >= 32:
                    flush()
        flush()
    finally:
        capture.release()
    return wardrobe


def build_stitch_review(records, predictions, wardrobe=None, *, stride=1, fps=30):
    """Use jersey reads and clothing appearance to reconnect team-consistent tracklets."""
    players = {(row["frame"], p["local_track_id"]): p for row in records for p in row["players"]
               if p.get("reid_eligible") and p.get("local_track_id") is not None}
    teams = _observation_teams(players)
    direct = {ref: read["number"] for ref, read in predictions.items()
              if ref in teams and read.get("number") and read.get("confidence", 0) >= .90}
    anchors = defaultdict(Counter)
    for ref, number in direct.items():
        p = players[ref]
        if p.get("id") is not None and p.get("team") == teams[ref] and p.get("jersey_number") == number:
            anchors[teams[ref], number][p["id"]] += 1
    roster = {}
    for key, votes in anchors.items():
        best = votes.most_common(2)
        if best[0][1] >= 3 and (len(best) == 1 or best[0][1] >= 2 * best[1][1]):
            roster[key] = best[0][0]
    # A global identity must have exactly one supported roster key.
    id_counts = Counter(roster.values())
    roster = {key: identity for key, identity in roster.items() if id_counts[identity] == 1}
    canonical_teams = {identity: key[0] for key, identity in roster.items()}
    tracklets = list(_tracklets(players, teams))
    proposals = {}
    for team, refs in tracklets:
        if team is None:
            continue
        votes = Counter(direct[ref] for ref in refs if teams.get(ref) == team and ref in direct)
        if not votes:
            continue
        ranked = votes.most_common(2)
        number, support = ranked[0]
        identity = roster.get((team, number))
        if identity is None or (len(ranked) > 1 and support <= ranked[1][1]):
            continue
        known = sum(players[ref].get("id") == identity and players[ref].get("jersey_number") == number for ref in refs)
        if support < 3 and not (support >= 2 and known >= 4):
            # Brief reappearances can be supported by a single near-certain read.
            if len(refs) > 4 or max(predictions[ref]["confidence"] for ref in refs if direct.get(ref) == number) < .97:
                continue
        for ref in refs:
            if teams.get(ref, team) == team:
                proposals[ref] = (identity, team, number)

    roster_by_id = {identity: key for key, identity in roster.items()}
    for team, refs in tracklets:
        known = Counter(players[ref].get("id") for ref in refs if players[ref].get("id") in roster_by_id)
        if not known:
            continue
        identity, support = known.most_common(1)[0]
        expected_team, number = roster_by_id[identity]
        if expected_team != team or support < 6 or support * 2 <= len(refs):
            continue
        confirmed = [ref for ref in refs if players[ref].get("id") == identity and direct.get(ref) == number]
        if not confirmed:
            continue
        # Isolated contradictory OCR reads should not erase short, continuous track gaps.
        opposing = Counter(direct[ref] for ref in refs if ref in direct and direct[ref] != number)
        if max(opposing.values(), default=0) >= 2:
            continue
        last_seen = None
        for ref in refs:
            old_id = players[ref].get("id")
            if old_id == identity:
                last_seen = ref[0]
            elif old_id is not None:
                last_seen = None
            elif last_seen is not None and ref[0] - last_seen <= 15 and teams.get(ref, team) == team:
                proposals.setdefault(ref, (identity, team, number))

    # Only clear stale cross-team IDs when multiple adjacent observations agree.
    for team, refs in tracklets:
        if team is None or sum(teams.get(ref) == team for ref in refs) < 5:
            continue
        for ref in refs:
            identity = players[ref].get("id")
            if ref not in proposals and identity in canonical_teams and canonical_teams[identity] != team:
                proposals[ref] = (None, team, None)

    if wardrobe:
        handoffs = wardrobe_handoffs(players, teams, tracklets, direct, roster, wardrobe, stride, fps)
        claimed = Counter((ref[0], key[0]) for ref, key in handoffs.items())
        for ref, key in handoffs.items():
            # Preserve a conflicting jersey-backed stitch. A wardrobe relink may
            # replace only the inherited identity or agree with stronger evidence.
            existing = proposals.get(ref)
            if claimed[ref[0], key[0]] == 1 and (
                existing is None or existing == key or existing[0] == players[ref].get("id")
            ):
                proposals[ref] = key

    edits = defaultdict(list)
    for row in records:
        frame = row["frame"]
        assigned = Counter(p["id"] for p in row["players"] if p.get("id") is not None)
        for p in row["players"]:
            ref = frame, p["local_track_id"]
            if ref not in proposals:
                continue
            identity, team, number = proposals[ref]
            old = p.get("id")
            if identity is not None and identity != old and assigned[identity]:
                continue
            if (old, p.get("team"), p.get("jersey_number")) == (identity, team, number):
                continue
            if old is not None:
                assigned[old] -= 1
            if identity is not None:
                assigned[identity] += 1
            edits[identity, team, number].append([ref[1], frame, frame])
    if not edits:
        return None
    return {"frame_count": len(records), "edits": [
        {"id": identity, "team": "light" if team else "dark", "jersey_number": number, "segments": segments}
        for (identity, team, number), segments in edits.items()
    ]}


def stitch_identities(records: list[dict], source_video: Path, sampler, resolver) -> dict | None:
    """Run the automatic stitching pass on the decoded clip and its cached OCR."""
    if len(records) < 12 or not sampler.predictions or getattr(sampler.reader, "error", None):
        return None
    players = {(row["frame"], p["local_track_id"]): p for row in records for p in row["players"]
               if p.get("reid_eligible") and p.get("local_track_id") is not None}
    if not players:
        return None
    predictions = dict(sampler.predictions)
    stride = max(1, round(resolver.fps / 10))
    wardrobe = _sample_tracklet_appearance(source_video, players, sampler.reader, predictions, stride)
    review = build_stitch_review(records, predictions, wardrobe, stride=stride, fps=resolver.fps)
    if review is not None:
        print(f"Automatic identity stitching: {sum(len(edit['segments']) for edit in review['edits'])} corrected observations")
    return review
