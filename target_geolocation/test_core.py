import math
import unittest

import numpy as np

from target_geolocation.core import (
    CameraCalibration,
    VehiclePose,
    project_pixel_to_ground,
)


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
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
        self.pose = VehiclePose(
            latitude_deg=1.0,
            longitude_deg=103.0,
            altitude_msl_m=100.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
        )

    def project(self, u: float, v: float):
        return project_pixel_to_ground(
            u=u,
            v=v,
            pose=self.pose,
            vehicle_agl_m=100.0,
            calibration=self.calibration,
        )

    def test_nadir_center_is_below_vehicle(self) -> None:
        result = self.project(160.0, 120.0)
        self.assertAlmostEqual(result.north_m, 0.0, places=6)
        self.assertAlmostEqual(result.east_m, 0.0, places=6)
        self.assertAlmostEqual(result.slant_range_m, 100.0, places=6)

    def test_image_right_is_east_when_heading_north(self) -> None:
        result = self.project(200.0, 120.0)
        self.assertAlmostEqual(result.north_m, 0.0, places=6)
        self.assertGreater(result.east_m, 0.0)

    def test_image_top_is_north_when_heading_north(self) -> None:
        result = self.project(160.0, 80.0)
        self.assertGreater(result.north_m, 0.0)
        self.assertAlmostEqual(result.east_m, 0.0, places=6)

    def test_yaw_rotates_image_top_toward_east(self) -> None:
        east_pose = VehiclePose(
            latitude_deg=self.pose.latitude_deg,
            longitude_deg=self.pose.longitude_deg,
            altitude_msl_m=self.pose.altitude_msl_m,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=math.pi / 2.0,
        )
        result = project_pixel_to_ground(
            u=160.0,
            v=80.0,
            pose=east_pose,
            vehicle_agl_m=100.0,
            calibration=self.calibration,
        )
        self.assertAlmostEqual(result.north_m, 0.0, places=6)
        self.assertGreater(result.east_m, 0.0)


if __name__ == "__main__":
    unittest.main()

