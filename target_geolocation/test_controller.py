import time
import unittest

import numpy as np

from target_geolocation.controller import TelemetryBuffer, process_frame
from target_geolocation.core import CameraCalibration


class FakeMessage:
    def __init__(self, message_type: str, **fields):
        self._message_type = message_type
        for name, value in fields.items():
            setattr(self, name, value)

    def get_type(self) -> str:
        return self._message_type


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = TelemetryBuffer()
        receive_time = time.monotonic()
        self.telemetry.ingest(
            FakeMessage(
                "ATTITUDE",
                time_boot_ms=10_000,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            ),
            receive_time,
        )
        self.telemetry.ingest(
            FakeMessage(
                "GLOBAL_POSITION_INT",
                time_boot_ms=10_000,
                lat=10_000_000,
                lon=1_030_000_000,
                alt=100_000,
                relative_alt=100_000,
                vx=1_000,
                vy=0,
                vz=0,
            ),
            receive_time,
        )
        self.calibration = CameraCalibration(
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

    def test_frame_json_is_fused_with_pose(self) -> None:
        frame = {
            "camera_id": "test",
            "frame_id": 9,
            "capture_time_boot_ms": 10_000,
            "image": {"width": 320, "height": 240},
            "detections": [
                {
                    "detection_id": 2,
                    "track_id": 7,
                    "bbox_xyxy": [150, 110, 170, 130],
                    "ground_anchor_uv": [160, 120],
                }
            ],
        }
        result = process_frame(
            frame=frame,
            receive_monotonic_s=time.monotonic(),
            telemetry=self.telemetry,
            calibration=self.calibration,
            geolocation_config={"agl_source": "fixed", "fixed_agl_m": 100},
        )[0]

        self.assertTrue(result["valid"])
        self.assertEqual(result["frame_id"], 9)
        self.assertEqual(result["track_id"], 7)
        self.assertAlmostEqual(result["target"]["latitude_deg"], 1.0)
        self.assertAlmostEqual(result["target"]["longitude_deg"], 103.0)
        self.assertEqual(result["quality"]["agl_source"], "fixed")
        self.assertIn("roll_1deg", result["debug"]["sensitivity"])
        self.assertAlmostEqual(
            result["debug"]["sensitivity"]["timestamp_10ms"]["magnitude_m"],
            0.1,
            places=3,
        )
        self.assertIn("当前时间戳不是传感器曝光时间", result["quality"]["warnings"][0])

if __name__ == "__main__":
    unittest.main()
