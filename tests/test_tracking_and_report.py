import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tempolens.dashboard import build
from tempolens.report import write_report
from tempolens.tracking import extract_features


class TrackingAndReportTests(unittest.TestCase):
    def test_extract_features_respects_tracking_frame_alignment(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata = folder / "atd.xml"
            raw = folder / "atd.txt"
            metadata.write_text(
                "<root><Team id='home'><Name>Barcelona</Name></Team>"
                "<Team id='away'><Name>Athletic</Name></Team>"
                "<Player teamId='home'/><Player teamId='away'/></root>", encoding="utf-8")
            raw.write_text(
                "1:0.50,0.50,0;0.70,0.50,0:0.51,0.50,0\n"
                "2:0.50,0.50,0;0.70,0.50,0:NaN,NaN,NaN\n"
                "3:0.50,0.50,0;0.70,0.50,0:0.69,0.50,0\n", encoding="utf-8")
            with patch("tempolens.tracking.require", return_value=[str(metadata), str(raw)]), \
                 patch("tempolens.tracking.source_frames", return_value=3):
                features, home, away = extract_features(data_dir=str(folder))
            self.assertEqual((home, away), ("Barcelona", "Athletic"))
            self.assertEqual(features.ball_seen.tolist(), [True, False, True])
            self.assertLess(features.nearest_home_m[0], features.nearest_away_m[0])
            self.assertGreater(features.nearest_home_m[2], features.nearest_away_m[2])
            self.assertEqual(features.ball_age.tolist(), [0, 1, 0])

    def test_report_and_dashboard_do_not_embed_source_assets(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            features = type("Features", (), {
                "ball_seen": np.array([True, True, False, True]),
                "ball_x": np.array([1.0, 2.0, np.nan, 5.0]),
                "ball_y": np.array([3.0, 4.0, np.nan, 6.0]),
            })()
            write_report(folder, np.array([0, 0, 2, 1]), features)
            report = folder / "transition_report.json"
            self.assertEqual(len(json.loads(report.read_text(encoding="utf-8"))["spells"]), 2)
            page = folder / "atlas.html"
            build(report, page)
            content = page.read_text(encoding="utf-8")
            self.assertIn("TempoLens", content)
            self.assertNotIn(".mp4", content)


if __name__ == "__main__":
    unittest.main()
