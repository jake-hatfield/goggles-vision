"""Use pose-aligned shorts and shoes to resolve same-team tracker handoffs."""

from collections import Counter, defaultdict
from itertools import groupby

import cv2
import numpy as np

from reid.consensus import LOCAL_TRACK_SEGMENT_MAX_GAP


# Downweight the chroma of the shorts and the brightness extremes of the shoes.
_WEIGHTS = np.array((1, .4, .4, 1, .4, .4, .5, .4, .4, .4, .7, .7, .7, 1, 1, .7, 1, 1))
_WINDOW = 7  # Samples are collected at approximately 10 Hz.


def sample_wardrobe(image, box, pose):
    """Summarize visible fabric and footwear in normalized Lab coordinates."""
    height = box[3] - box[1]
    if pose is None or height < 65:
        return None
    coordinates, confidence = pose
    radius = max(2, round(height * .035))
    regions = ([], [])
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        centers = (
            (.72 * coordinates[hip] + .28 * coordinates[knee], min(confidence[hip], confidence[knee])),
            (coordinates[ankle] + (0, height * .023), confidence[ankle]),
        )
        for patches, (center, score) in zip(regions, centers):
            if not (score >= .25 and np.isfinite(center).all()):
                continue
            x, y = np.rint(center).astype(int)
            # Avoid patches where clipping would leave only a sliver of the garment.
            if radius <= x < image.shape[1] - radius and radius <= y < image.shape[0] - radius:
                patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
                patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3))
    if not all(regions):
        return None
    return np.concatenate([
        np.percentile(np.concatenate(patches), (20, 50, 80), axis=0).ravel() / 255
        for patches in regions
    ])


def _distance(left, right):
    return np.linalg.norm((left - right) * _WEIGHTS)


def _profile(refs, samples):
    return np.median([samples[ref] for ref in refs], axis=0)


def _clear_match(profile, key, gallery, match_limit):
    distances = [_distance(profile, color) for other, color in gallery.items() if other != key]
    distance = _distance(profile, gallery[key])
    return distance <= match_limit and min(distances, default=np.inf) >= distance + .025


def _supported_tail(refs, sampled, samples, direct, old_key, key, gallery, stride):
    """Extend the correction only while successive views support the new player."""
    stop = len(refs)
    new_number = key[2]
    contradictions = [ref for ref in refs if ref in direct and direct[ref] != new_number]
    votes = Counter(direct[ref] for ref in contradictions)
    conflict = next((ref for ref in contradictions if votes[direct[ref]] >= 2), None)
    if conflict is not None:
        stop = refs.index(conflict)
        sampled = [ref for ref in sampled if ref[0] < conflict[0]]
    last_frame = refs[0][0]
    for index, ref in enumerate(sampled):
        if ref[0] - last_frame > 3 * stride:
            sampled = sampled[:index]
            break
        last_frame = ref[0]
    if len(sampled) < 4:
        return []
    last_supported = None
    for start in range(0, len(sampled), _WINDOW):
        window = sampled[start:start + _WINDOW]
        if len(window) < 4:
            window = sampled[-_WINDOW:]
        profile = _profile(window, samples)
        if _distance(profile, gallery[old_key]) < .13 or not _clear_match(profile, key, gallery, .105):
            stop = min(stop, refs.index(sampled[start]))
            break
        last_supported = window[-1][0]
    if last_supported is None:
        return []
    return [ref for ref in refs[:stop] if ref[0] <= last_supported + stride]


def wardrobe_handoffs(players, teams, tracklets, direct, roster, samples, stride, fps):
    """Return high-confidence relinks when a single local track changes occupants."""
    def identity(ref):
        player = players[ref]
        return player.get("id"), player.get("team"), player.get("jersey_number")

    anchors = defaultdict(list)
    for ref, number in direct.items():
        key = identity(ref)
        if key[0] is not None and key[1:] == (teams[ref], number):
            anchors[key].append(ref)
    profiles = {}
    for key, refs in anchors.items():
        sampled = [ref for ref in refs if ref in samples]
        if len(sampled) >= 3:
            profiles[key] = _profile(sampled, samples)
    # Candidate identities need a unique jersey association. The inherited local ID
    # only needs evidence from its own prefix, even when that ID fragmented earlier.
    roster_keys = {key for key in profiles if roster.get(key[1:]) == key[0]}
    keys_by_identity = defaultdict(list)
    for key in profiles:
        keys_by_identity[key[:2]].append(key)
    proposals = {}
    recent_handoffs = {}
    for team, tracklet in tracklets:
        for inherited, group in groupby(tracklet, key=lambda ref: identity(ref)[:2]):
            keys = keys_by_identity[inherited]
            if inherited[1] != team or len(keys) != 1:
                continue
            old_key = keys[0]
            refs = list(group)
            sampled = [ref for ref in refs if ref in samples and teams.get(ref) == team]
            if len(sampled) < 4:
                continue
            gallery = {key: color for key, color in profiles.items() if key[1] == team}
            state = tracklet[0][1], old_key
            previous = recent_handoffs.get(state)
            if previous and refs[0][0] - previous[0] <= round(.75 * fps):
                _, key, recent_profile = previous
                # Across a short gap, compare with the already verified view of
                # this player as well as the other roster players' profiles.
                recent_gallery = {**gallery, key: recent_profile}
                tail = _supported_tail(refs, sampled, samples, direct, old_key, key, recent_gallery, stride)
                if tail:
                    proposals.update({ref: key for ref in tail})
                    recent_handoffs[state] = tail[-1][0], key, recent_profile
                    continue

            prefix = set(anchors[old_key]).intersection(refs)
            if len(prefix) < 3:
                continue
            for index in range(4, len(sampled) - _WINDOW + 1):
                start = sampled[index]
                prev = sampled[index - 1]
                jump = samples[start][0] - samples[prev][0]
                if abs(jump) < .06 or start[0] - prev[0] > LOCAL_TRACK_SEGMENT_MAX_GAP:
                    continue
                if prev[0] - sampled[index - 4][0] > 6 * stride:
                    continue
                if sum(ref[0] < start[0] for ref in prefix) < 3:
                    continue
                old = _profile(sampled[index - 4:index], samples)
                new = _profile(sampled[index:index + _WINDOW], samples)
                # Require an abrupt shorts change, sustained in the next views, and
                # independent corroboration from the shoes.
                change = new[0] - old[0]
                if abs(change) < .08 or jump * change <= 0:
                    continue
                if np.linalg.norm((old - new)[9:] * _WEIGHTS[9:]) < .05:
                    continue
                if _distance(old, profiles[old_key]) > .085:
                    continue
                key = min(gallery, key=lambda item: _distance(new, gallery[item]))
                if key not in roster_keys or key[2] == old_key[2] or not _clear_match(new, key, gallery, .085):
                    continue
                if sum(ref[0] < start[0] for ref in anchors[key]) < 3:
                    continue
                tail = _supported_tail(refs[refs.index(start):], sampled[index:], samples, direct,
                                       old_key, key, gallery, stride)
                if len(tail) < 4 * stride:
                    continue
                proposals.update({ref: key for ref in tail})
                recent = [ref for ref in sampled[index:] if ref[0] <= tail[-1][0]]
                recent_handoffs[state] = tail[-1][0], key, _profile(recent, samples)
                break
    return proposals
