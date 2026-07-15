import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from stream import TrainingVideoRecorder, detections_from_result


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        if isinstance(self.value, list):
            return FakeTensor(self.value[index])
        raise TypeError(index)

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class FakeBox:
    xyxy = FakeTensor([[10.0, 20.0, 1280.0, 720.0]])
    cls = FakeTensor([0])
    conf = FakeTensor([0.9])
    id = FakeTensor([3])


class FakeResult:
    boxes = [FakeBox()]
    orig_shape = (720, 1280)


class FakeModel:
    names = {0: "target"}


class StreamTests(unittest.TestCase):
    def test_ground_anchor_is_clamped_inside_image(self) -> None:
        detection = detections_from_result(FakeResult(), FakeModel())[0]
        self.assertEqual(detection["ground_anchor_uv"], [645.0, 719.0])

    def test_training_video_is_rate_limited_and_logs_frame_mapping(self) -> None:
        class FakeWriter:
            def __init__(self):
                self.frames = []

            def isOpened(self):
                return True

            def write(self, frame):
                self.frames.append(frame.copy())

            def release(self):
                pass

        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as directory, patch(
            "stream.cv2.VideoWriter", return_value=writer
        ):
            recorder = TrainingVideoRecorder(
                directory,
                width=16,
                height=12,
                fps=10,
                segment_seconds=60,
                jpeg_quality=85,
                min_free_disk_gb=0,
                max_total_gb=0,
                session={"test": True},
            )
            frame = np.zeros((12, 16, 3), dtype=np.uint8)
            recorder.record(1, 1_000_000_000, frame)
            recorder.record(2, 1_050_000_000, frame)
            recorder.record(3, 1_100_000_000, frame)
            recorder.close()

            rows = (Path(directory) / "video_frames.ndjson").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(len(writer.frames), 2)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
