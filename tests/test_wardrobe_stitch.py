import unittest

import numpy as np

from reid.review import apply_identity_review
from reid.stitch import build_stitch_review


def _appearance(shorts, shoes, shoe_b):
    return np.array([shorts, .5, .5] * 3 + [shoes, .5, shoe_b] * 3)


def _clip(*, competing_match=False, competing_claim=False, occupied=False, conflicting_ocr=False,
          missing_number=False):
    records, predictions, appearance = [], {}, {}
    previous = _appearance(.28, .25, .50)
    incoming = _appearance(.14, .38, .56)
    for frame in range(1, 64):
        players = []

        def observe(local_id, identity, number, team, profile, read=False):
            players.append({
                "local_track_id": local_id, "id": identity, "jersey_number": number, "team": team,
                "jersey_l": .65 if team else .15, "jersey_s": .2, "reid_eligible": True,
            })
            ref = frame, local_id
            if (frame - 1) % 3 == 0:
                appearance[ref] = profile
                if read:
                    predictions[ref] = {"number": number, "confidence": .99}

        if frame <= 31 or frame >= 41:
            number = None if missing_number and frame >= 13 else "4"
            observe(31, 7, number, 0, previous if frame < 13 else incoming,
                    read=frame < 13 or (conflicting_ocr and 13 <= frame <= 19))
            if competing_claim:
                observe(32, 9, "14", 0, previous if frame < 13 else incoming, read=frame < 13)
        if frame < 13 or occupied:
            observe(2, 11, "10", 0, incoming, read=True)
        if competing_match and frame < 13:
            observe(3, 6, "5", 0, incoming, read=True)
        observe(5, 8, "18", 1, incoming, read=True)
        records.append({"frame": frame, "players": players})
    return records, predictions, appearance


class WardrobeStitchTest(unittest.TestCase):
    def _local_ids(self, **options):
        records, predictions, appearance = _clip(**options)
        review = build_stitch_review(records, predictions, appearance, stride=3, fps=30)
        if review:
            apply_identity_review(records, review, automatic=True)
        return {row["frame"]: player["id"] for row in records for player in row["players"]
                if player["local_track_id"] == 31}

    def test_same_team_handoff_and_short_gap(self):
        self.assertEqual(self._local_ids(missing_number=True), {
            frame: 7 if frame < 13 else 11 for frame in range(1, 64) if frame <= 31 or frame >= 41
        })

    def test_ambiguous_rival_appearance_stays_unresolved(self):
        self.assertEqual(set(self._local_ids(competing_match=True).values()), {7})

    def test_visible_candidate_keeps_its_id(self):
        self.assertEqual(set(self._local_ids(occupied=True).values()), {7})

    def test_two_tracks_cannot_claim_the_same_id(self):
        self.assertEqual(set(self._local_ids(competing_claim=True).values()), {7})

    def test_repeated_number_read_takes_precedence(self):
        self.assertEqual(set(self._local_ids(conflicting_ocr=True).values()), {7})
