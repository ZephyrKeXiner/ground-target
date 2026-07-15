import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from target_geolocation.core import CameraCalibration
from target_geolocation.replay import replay_events


class ReplayTests(unittest.TestCase):
    def test_recorded_events_can_be_replayed(self) -> None:
        events = [
            {
                "event": "telemetry",
                "receive_monotonic_ns": 100_000_000_000,
                "message_type": "ATTITUDE",
                "message": {
                    "time_boot_ms": 10_000,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
            },
            {
                "event": "telemetry",
                "receive_monotonic_ns": 100_000_000_000,
                "message_type": "GLOBAL_POSITION_INT",
                "message": {
                    "time_boot_ms": 10_000,
                    "lat": 10_000_000,
                    "lon": 1_030_000_000,
                    "alt": 100_000,
                    "relative_alt": 100_000,
                    "vx": 0,
                    "vy": 0,
                    "vz": 0,
                },
            },
            {
                "event": "bbox_frame",
                "receive_monotonic_ns": 100_010_000_000,
                "frame": {
                    "camera_id": "test",
                    "frame_id": 4,
                    "capture_monotonic_ns": 100_000_000_000,
                    "image": {"width": 320, "height": 240},
                    "detections": [
                        {
                            "detection_id": 0,
                            "bbox_xyxy": [150, 110, 170, 130],
                            "ground_anchor_uv": [160, 120],
                        }
                    ],
                },
            },
        ]
        calibration = CameraCalibration(
            camera_id="test",
            image_width=320,
            image_height=240,
            camera_matrix=np.array(
                [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.zeros(5),
            rotation_body_from_camera=np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            lever_arm_body_m=np.zeros(3),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.ndjson"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            results = list(
                replay_events(
                    events_path=path,
                    calibration=calibration,
                    geolocation_config={
                        "agl_source": "fixed",
                        "fixed_agl_m": 100,
                    },
                )
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["valid"])
        self.assertTrue(results[0]["replayed"])
        self.assertEqual(results[0]["frame_id"], 4)


if __name__ == "__main__":
    unittest.main()
