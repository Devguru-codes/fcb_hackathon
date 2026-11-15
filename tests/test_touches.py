import unittest

import numpy as np

from tempolens.match import A, H, MatchTable
from tempolens.touches import dfl_event_frame, snap_to_contact


class DflEventFrameTests(unittest.TestCase):
    def test_times_map_into_the_halves_in_table_order(self):
        halves = [(100.0, 50), (200.0, 50)]          # (first ball time, frames) per half
        self.assertEqual(dfl_event_frame(100.0, halves), 0)
        self.assertEqual(dfl_event_frame(101.96, halves), 49)
        self.assertIsNone(dfl_event_frame(102.0, halves))      # after half 1, before half 2
        self.assertEqual(dfl_event_frame(200.0, halves), 50)
        self.assertIsNone(dfl_event_frame(99.0, halves))


class SnapToContactTests(unittest.TestCase):
    def test_removes_the_clock_offset_and_drops_events_without_contact(self):
        n = 2000
        players = np.full((n, 2, 2), 30.0, dtype=np.float32)
        players[:, 1] = 70.0
        ball = np.full((n, 2), 50.0, dtype=np.float32)
        touches = [300, 700, 1100, 1500]                  # the true contacts, home then away
        for i, f in enumerate(touches):
            players[f - 3:f + 4, i % 2] = 50.0                # that team's player reaches the ball
        table = MatchTable(name="t", ball=ball, players=players,
                           slot_team=np.array([H, A], dtype=np.int8), period=np.ones(n, np.int8))
        logged = np.array(touches + [1800]) + 30           # event clock 30 frames late, one fake
        teams = np.array([H, A, H, A, H], dtype=np.int8)
        frames, kept = snap_to_contact(table, logged, teams)
        self.assertEqual(len(frames), 4)                   # the event with no contact is dropped
        np.testing.assert_array_equal(kept, teams[:4])
        self.assertTrue(np.all(np.abs(frames - np.array(touches)) <= 3))


class DflEventFrameTests(unittest.TestCase):
    def test_event_times_map_to_table_frames_across_halves_and_time_zones(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens.touches import dfl_touch_frames

        def ball(section, first, n, start):
            frames = "\n".join('<Frame N="{}" T="2023-05-27T{}:{:02d}.{:03d}+00:00" X="0" Y="0"/>'.format(
                first + i, start, (i * 40) // 1000, (i * 40) % 1000) for i in range(n))
            return '<FrameSet GameSection="{}" TeamId="BALL">\n{}\n</FrameSet>'.format(section, frames)

        with tempfile.TemporaryDirectory() as folder:
            dfl = Path(folder) / "dfl_idsse"
            dfl.mkdir()
            (Path(folder) / "metrica_sample_data").mkdir()   # marks an open-data folder
            (dfl / "DFL_02_01_matchinformation_DFL-COM-0_DFL-MAT-TEST.xml").write_text(
                '<General HomeTeamId="HOME" GuestTeamId="AWAY"/>', encoding="utf-8")
            (dfl / "DFL_04_03_positions_raw_observed_DFL-COM-0_DFL-MAT-TEST.xml").write_text(
                "<Positions>\n{}\n{}\n</Positions>".format(ball("firstHalf", 10000, 100, "13:30"),
                                                          ball("secondHalf", 50000, 50, "14:30")),
                encoding="utf-8")
            events = ['<Event EventTime="2023-05-27T15:30:00.400+02:00"><Play Team="HOME"/></Event>',
                      '<Event EventTime="2023-05-27T16:30:01.000+02:00"><Play Team="AWAY"/></Event>',
                      '<Event EventTime="2023-05-27T17:00:00.000+02:00"><Play Team="HOME"/></Event>']
            (dfl / "DFL_03_02_events_raw_DFL-COM-0_DFL-MAT-TEST.xml").write_text(
                "<Events>{}</Events>".format("".join(events)), encoding="utf-8")
            with patch.dict(os.environ, {"MTAH_OPEN_DATA_DIR": folder}):
                frames, teams = dfl_touch_frames("TEST")
        # 13:30:00.4 UTC is frame 10 of the first half; 14:30:01 UTC is frame 25 of the second,
        # which starts at table index 100; the third event falls in no half and is dropped
        self.assertEqual(frames.tolist(), [10, 125])
        self.assertEqual(teams.tolist(), [H, A])


if __name__ == "__main__":
    unittest.main()
