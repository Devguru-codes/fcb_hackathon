import tempfile
import unittest
from pathlib import Path

import numpy as np

from tempolens.review import disagreements, render


class ReviewTests(unittest.TestCase):
    def test_short_disagreements_are_dropped_and_spans_merge(self):
        a = np.zeros(400, dtype=np.int8)
        b = a.copy()
        b[100:110] = 1          # 10 frames: too short
        b[200:230] = 2          # 30 frames: kept
        b[260:290] = 1          # 30 frames: kept, its context overlaps the previous one
        a[390:] = b[390:] = 3   # ungraded tail never counts
        cores, spans = disagreements(a, b, min_frames=25, context=50)
        self.assertEqual(cores, [(200, 230), (260, 290)])
        self.assertEqual(spans, [(150, 340)])

    def test_render_writes_exactly_the_span_frames(self):
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "in.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
            for i in range(60):
                writer.write(np.full((180, 320, 3), i * 4, dtype=np.uint8))
            writer.release()
            a = np.zeros(60, dtype=np.int8)
            b = a.copy()
            b[20:40] = 1
            truth = np.full(60, 3, dtype=np.int8)
            cores, spans = disagreements(a, b, min_frames=10, context=5)
            written = render(video, Path(folder) / "reel.mp4", a, b, truth, cores, spans, ("v4", "v5"))
            self.assertEqual(written, 30)
            self.assertEqual(int(cv2.VideoCapture(str(Path(folder) / "reel.mp4"))
                                 .get(cv2.CAP_PROP_FRAME_COUNT)), 30)


if __name__ == "__main__":
    unittest.main()
