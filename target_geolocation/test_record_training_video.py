from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from record_training_video import gstreamer_command, unique_video_path


class FullResolutionRecorderTests(unittest.TestCase):
    def test_pipeline_uses_imx219_maximum_mode(self) -> None:
        output = Path("/tmp/test.mkv")
        command = gstreamer_command(
            output=output,
            sensor_id=0,
            width=3280,
            height=2464,
            fps=21,
            bitrate_kbps=50000,
            rotation="counterclockwise90",
            exposure_min_us=13,
            exposure_max_us=2000,
        )
        rendered = " ".join(str(value) for value in command)
        self.assertIn("width=(int)3280", rendered)
        self.assertIn("height=(int)2464", rendered)
        self.assertIn("framerate=(fraction)21/1", rendered)
        self.assertIn("exposuretimerange=13000 2000000", command)
        self.assertIn("nvvidconv", command)
        self.assertIn("flip-method=1", command)
        self.assertIn("width=(int)2464", rendered)
        self.assertIn("height=(int)3280", rendered)
        self.assertIn("x264enc", command)
        self.assertIn("bitrate=50000", command)
        self.assertIn("speed-preset=ultrafast", command)
        self.assertIn("h264parse", command)
        self.assertNotIn("nvjpegenc", command)
        self.assertIn("matroskamux", command)

    def test_video_path_does_not_overwrite_same_second(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "record_training_video.datetime"
        ) as fake_datetime:
            fake_datetime.now.return_value.strftime.return_value = "20260720T120000Z"
            first = unique_video_path(Path(directory))
            first.touch()
            second = unique_video_path(Path(directory))
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.endswith("_1.mkv"))


if __name__ == "__main__":
    unittest.main()
