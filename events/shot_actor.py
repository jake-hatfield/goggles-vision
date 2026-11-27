"""Find the last supported hand-to-ball contact before a recorded shot."""

from collections import defaultdict
from math import hypot
from typing import NamedTuple


class ObservedBall(NamedTuple):
    x: float
    y: float
    track_id: int | None
    rim: list[float] | None


def shot_release_actor(start_frame, fps, observations, ball_frames, *, use_boxes=False):
    """Return the local player observation at release when the flight supports it."""
    first = max(1, start_frame - round(1.45 * fps))
    last = start_frame + max(1, round(.14 * fps))
    support_span = max(3, round(.19 * fps))
    flight_span = max(5, round(.42 * fps))
    anchor_ball = next(
        (ball_frames[frame] for frame in range(start_frame, last + 1) if frame in ball_frames),
        None,
    )
    rim_samples = [
        ball for frame in range(start_frame, start_frame + round(.85 * fps) + 1)
        if (ball := ball_frames.get(frame)) is not None and ball.rim
        and (anchor_ball is None or ball.track_id == anchor_ball.track_id)
    ]
    closest = min(rim_samples, key=lambda ball: hypot(ball.x - ball.rim[0], ball.y - ball.rim[1]), default=None)
    target_rim = closest.rim if closest is not None else None
    contacts = defaultdict(list)
    releases = defaultdict(list)
    for frame in range(first, last + 1):
        ball = ball_frames.get(frame)
        if ball is None:
            continue
        if anchor_ball and anchor_ball.track_id is not None and ball.track_id != anchor_ball.track_id:
            continue
        players = []
        for player in observations.get(frame, ()):
            contact = _upper_body_contact(player) if use_boxes else player.wrist_distance
            if contact is not None:
                players.append((contact, player))
        if not players:
            continue
        players.sort(key=lambda sample: sample[0])
        distance, player = players[0]
        if (distance > (.37 if use_boxes else .24)
            or (len(players) > 1 and players[1][0] - distance < (.09 if use_boxes else .07))):
            continue
        recent = contacts[player.local_id]
        recent.append(player)
        recent[:] = [item for item in recent if frame - item.frame <= support_span]
        if len(recent) < (3 if use_boxes else 2):
            continue
        if use_boxes and not any(row.ball_offset[1] >= .15 for row in recent):
            continue
        if not _leaves_hands(player, ball, frame, flight_span, observations, ball_frames, target_rim):
            continue
        releases[player.local_id].append(player)

    candidates = sorted((rows[-1] for rows in releases.values()), key=lambda row: -row.frame)
    if not candidates:
        return None
    best = candidates[0]
    # Two distinct players with credible contact in the same release window
    # provide an ambiguous handoff, block, or occlusion.
    if len(candidates) > 1 and best.frame - candidates[1].frame <= round(.30 * fps):
        return None
    return best


def _upper_body_contact(player):
    if player.ball_offset is None:
        return None
    horizontal, height = player.ball_offset
    if -.37 <= height <= .49:
        return abs(horizontal) + .5 * max(0, -height - .18)
    return None


def _leaves_hands(player, ball, frame, flight_span, observations, ball_frames, target_rim):
    minimum_rise = player.height * .20
    rising = separating = approaching = 0
    previous = ball
    previous_frame = frame
    for n in range(frame + 1, frame + flight_span + 1):
        later = ball_frames.get(n)
        if later is None:
            continue
        if later.track_id != ball.track_id or n - previous_frame > 4:
            break
        if later.y - previous.y > player.height * .16:
            break
        previous, previous_frame = later, n
        if ball.y - later.y < minimum_rise:
            continue
        rising += 1
        follow = next((row for row in observations.get(n, ()) if row.local_id == player.local_id), None)
        if follow is not None and (
            (follow.wrist_distance is not None and follow.wrist_distance >= .33)
            or (follow.ball_distance is not None and follow.ball_distance >= .22)
        ):
            separating += 1
        if target_rim and (
            hypot(ball.x - target_rim[0], ball.y - target_rim[1])
            - hypot(later.x - target_rim[0], later.y - target_rim[1])
        ) >= player.height * .27:
            approaching += 1
        if rising >= 3 and separating >= 2:
            return True
        if rising >= 4 and approaching >= 3:
            return True
    return False
