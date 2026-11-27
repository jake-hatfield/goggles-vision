import json
import tempfile
import unittest
from pathlib import Path

from events.attribution import event_team_from_shooter_context
from events.serialization import write_event_list
from reid.records import translate_event_identity_references
from reid.review import remap_reviewed_references


def _track(local_id, global_id, number, team="dark", *, locked=True):
    return {
        "local_track_id": local_id, "id": global_id, "jersey_number": number,
        "team_semantic": team, "jersey_number_locked": locked,
    }


def _export(event, player_frames, ball_frames=None):
    with tempfile.TemporaryDirectory() as folder:
        tracks = Path(folder) / "tracks.jsonl"
        output = Path(folder) / "events.json"
        with tracks.open("w") as stream:
            for frame, players in enumerate(player_frames, 1):
                row = {"frame": frame, "fps": 30, "players": players,
                       "events": [event] if frame == len(player_frames) else []}
                if ball_frames is not None:
                    row["ball"] = ball_frames[frame - 1]
                stream.write(json.dumps(row) + "\n")
        write_event_list(tracks, output)
        return json.loads(output.read_text())["events"][0]


class EventAttributionTest(unittest.TestCase):
    def test_shot_uses_release_identity_even_if_track_switches_in_flight(self):
        event = {"event": "shot", "player_id": 21, "local_track_id": 2, "actor_frame": 4,
                 "start_frame": 4, "end_frame": 18, "result": "made", "points": 2, "team": "dark"}
        rows = [[_track(2, 21, "4")]] * 6 + [[_track(2, 41, "10")]] * 12
        out = _export(event, rows)
        self.assertEqual((out["player_id"], out["jersey_number"], out["actor_frame"]), (21, "4", 4))
        self.assertEqual((out["result"], out["attrs"]["points"]), ("made", 2))

    def test_shot_withholds_actor_far_from_observed_ball(self):
        event = {"event": "shot", "player_id": 21, "local_track_id": 2, "actor_frame": 10,
                 "start_frame": 10, "end_frame": 14, "result": "made", "points": 2, "team": "dark"}
        rows = [[{**_track(2, 21, "4"), "box": [600, 300, 650, 400]}]] * 14
        far = {"center": [100, 310], "observed": True}
        out = _export(event, rows, [far] * 14)
        self.assertNotIn("player_id", out)
        self.assertEqual((out["result"], out["attrs"]["points"]), ("made", 2))

        near = {"center": [608, 310], "observed": True}
        self.assertEqual(_export(event, rows, [near] * 3 + [far] * 11)["player_id"], 21)

    def test_shot_withholds_actor_when_ball_is_by_opposing_wrists(self):
        event = {"event": "shot", "player_id": 21, "local_track_id": 2, "actor_frame": 10,
                 "start_frame": 10, "end_frame": 14, "result": "made", "points": 2, "team": "dark"}
        shooter = {**_track(2, 21, "4"), "box": [90, 90, 130, 190],
                   "pose": [None] * 9 + [[120, 140, .99]]}
        rival = {**_track(3, 41, "20", "light"), "box": [90, 85, 120, 185],
                 "pose": [None] * 9 + [[100, 101, .99]]}
        ball = {"center": [100, 100], "observed": True}
        self.assertNotIn("player_id", _export(event, [[shooter, rival]] * 14, [ball] * 14))
        self.assertEqual(_export(event, [[shooter]] * 14, [ball] * 14)["player_id"], 21)

    def test_shot_recovers_a_different_player_from_hand_contact_and_flight(self):
        event = {"event": "shot", "player_id": 21, "local_track_id": 2, "actor_frame": 7,
                 "start_frame": 7, "end_frame": 14, "result": "made", "points": 2, "team": "dark"}
        old = {**_track(2, 21, "4"), "box": [600, 130, 650, 230]}
        shooter = {**_track(3, 41, "20", "light"), "box": [90, 130, 150, 230],
                   "pose": [None] * 9 + [[120, 155, .99]]}
        centers = [155] * 6 + [148, 132, 117, 103, 90, 79, 69, 65]
        ball = [{"center": [120, y], "id": 6, "observed": True} for y in centers]
        out = _export(event, [[old, shooter]] * 14, ball)
        self.assertEqual((out["local_track_id"], out["player_id"], out["actor_team"]), (3, 41, "light"))
        self.assertNotIn("jersey_number", out)
        self.assertEqual((out["team"], out["result"], out["attrs"]["points"]), ("dark", "made", 2))

    def test_box_evidence_can_recover_a_shooter_without_an_identity(self):
        event = {"event": "shot", "start_frame": 7, "end_frame": 14, "result": "miss"}
        shooter = {**_track(3, None, None), "box": [90, 130, 150, 230], "reid_eligible": False}
        centers = [170] * 5 + [165, 150, 134, 119, 104, 89, 79, 70, 64]
        ball = [{"center": [120, y], "observed": True} for y in centers]
        out = _export(event, [[shooter]] * 14, ball)
        self.assertEqual(out["local_track_id"], 3)
        self.assertNotIn("player_id", out)
        self.assertNotIn("jersey_number", out)

    def test_withholds_number_after_recent_change_to_another_number(self):
        event = {"event": "shot", "player_id": 41, "local_track_id": 2, "actor_frame": 8,
                 "start_frame": 8, "end_frame": 14, "team": "dark"}
        rows = [[_track(2, 21, "1")]] * 6 + [[_track(2, 41, "10")]] * 8
        out = _export(event, rows)
        self.assertEqual(out["player_id"], 41)
        self.assertNotIn("jersey_number", out)

    def test_withholds_number_while_two_teammates_have_the_same_label(self):
        event = {"event": "shot", "player_id": 21, "local_track_id": 2, "actor_frame": 8,
                 "start_frame": 8, "end_frame": 14, "team": "dark"}
        rows = [[_track(2, 21, "10"), _track(3, 41, "10")]] * 14
        out = _export(event, rows)
        self.assertEqual(out["player_id"], 21)
        self.assertNotIn("jersey_number", out)
        rivals = [[_track(2, 21, "10"), _track(3, 41, "10", "light")]] * 14
        self.assertEqual(_export(event, rivals)["jersey_number"], "10")

    def test_legacy_pass_looks_up_passer_before_the_catch(self):
        event = {"event": "pass", "player_id": 21, "receiver_id": 41,
                 "start_frame": 9, "end_frame": 14, "team": "dark"}
        rows = [[_track(2, 21, "4")]] * 7 + [[_track(3, 41, "10")]] * 7
        out = _export(event, rows)
        self.assertEqual((out["player_id"], out["local_track_id"], out["jersey_number"]), (21, 2, "4"))
        self.assertEqual(out["attrs"]["receiver_id"], 41)

    def test_does_not_attach_jersey_from_wrong_team(self):
        event = {"event": "shot", "player_id": 21, "local_track_id": 2, "actor_frame": 4,
                 "start_frame": 4, "end_frame": 8, "team": "light", "result": "miss"}
        out = _export(event, [[_track(2, 21, "4")]] * 8)
        self.assertEqual((out["team"], out["result"]), ("light", "missed"))
        self.assertNotIn("jersey_number", out)
        self.assertNotIn("player_id", out)

    def test_withholds_number_with_a_single_locked_observation(self):
        event = {"event": "dribble", "player_id": 21, "start_frame": 4, "end_frame": 8, "team": "dark"}
        rows = [[_track(2, 21, "4")]] + [[_track(2, 21, "4", locked=False)]] * 7
        out = _export(event, rows)
        self.assertEqual(out["player_id"], 21)
        self.assertNotIn("jersey_number", out)

    def test_uses_finalized_track_id_after_stitching(self):
        event = {"event": "dribble", "player_id": 21, "local_track_id": 2, "actor_frame": 4,
                 "start_frame": 4, "end_frame": 8, "team": "dark"}
        out = _export(event, [[_track(2, 41, "10")]] * 8)
        self.assertEqual((out["player_id"], out["jersey_number"]), (41, "10"))
        ambiguous = _export(event, [[_track(2, 41, "10"), _track(3, 21, "4")]] * 8)
        self.assertNotIn("jersey_number", ambiguous)
        self.assertNotIn("player_id", ambiguous)

    def test_actor_and_receiver_references_use_different_frames(self):
        class Resolver:
            def get_global_id_at(self, local_id, frame):
                return local_id * 100 + frame

        event = {"event": "pass", "player_id": 2, "local_track_id": 2, "actor_frame": 7,
                 "receiver_id": 3, "start_frame": 9, "end_frame": 14}
        out = translate_event_identity_references(event, Resolver(), parent_frame=14)
        self.assertEqual((out["player_id"], out["receiver_id"]), (207, 314))
        remap_reviewed_references(out, {7: {207: 1}, 14: {314: 3}}, 14)
        self.assertEqual((out["player_id"], out["receiver_id"]), (1, 3))

    def test_wrist_evidence_updates_both_shooter_references(self):
        event = {"event": "shot", "player_id": 2, "local_track_id": 2, "start_frame": 4}
        players = [
            {"id": 2, "team_semantic": "dark", "pose": [None] * 9 + [[100, 100, .99]]},
            {"id": 3, "team_semantic": "light", "pose": [None] * 9 + [[40, 40, .99]]},
        ]
        rows = [{"frame": frame, "players": players, "ball": {"center": [40, 40], "observed": True}}
                for frame in range(2, 5)]
        self.assertEqual(event_team_from_shooter_context(event, rows), "light")
        self.assertEqual((event["player_id"], event["local_track_id"]), (3, 3))
