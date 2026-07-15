from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


EARTH_RADIUS_M = 6_378_137.0


class GeolocationError(ValueError):
    """Raised when a pixel cannot be projected onto the configured ground plane."""


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_body_from_camera: np.ndarray
    lever_arm_body_m: np.ndarray

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraCalibration":
        if not data.get("calibrated", False):
            raise GeolocationError(
                "camera.calibrated is false; fill in the measured camera calibration first"
            )

        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(data.get("distortion", []), dtype=np.float64)
        rotation = np.asarray(
            data["rotation_body_from_camera"], dtype=np.float64
        )
        lever_arm = np.asarray(
            data.get("lever_arm_body_m", [0.0, 0.0, 0.0]), dtype=np.float64
        )

        if camera_matrix.shape != (3, 3):
            raise GeolocationError("camera_matrix must be a 3x3 matrix")
        if rotation.shape != (3, 3):
            raise GeolocationError("rotation_body_from_camera must be a 3x3 matrix")
        if lever_arm.shape != (3,):
            raise GeolocationError("lever_arm_body_m must contain 3 numbers")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise GeolocationError("rotation_body_from_camera is not orthonormal")
        if np.linalg.det(rotation) < 0.99:
            raise GeolocationError("rotation_body_from_camera must be a proper rotation")

        return cls(
            camera_id=str(data.get("camera_id", "camera")),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            camera_matrix=camera_matrix,
            distortion=distortion,
            rotation_body_from_camera=rotation,
            lever_arm_body_m=lever_arm,
        )


@dataclass(frozen=True)
class VehiclePose:
    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_rad_s: float = 0.0
    pitch_rate_rad_s: float = 0.0
    yaw_rate_rad_s: float = 0.0


@dataclass(frozen=True)
class GroundProjection:
    latitude_deg: float
    longitude_deg: float
    north_m: float
    east_m: float
    slant_range_m: float
    ray_down_component: float
    ray_camera: tuple[float, float, float]
    ray_body: tuple[float, float, float]
    ray_ned: tuple[float, float, float]
    camera_offset_ned_m: tuple[float, float, float]
    camera_agl_m: float


def rotation_ned_from_body(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Return the FRD-body to NED rotation matrix used by ArduPilot attitude."""
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def bbox_anchor(
    detection: Mapping[str, Any], default_anchor: str = "bottom_center"
) -> tuple[float, float]:
    supplied = detection.get("ground_anchor_uv")
    if supplied is not None:
        if len(supplied) != 2:
            raise GeolocationError("ground_anchor_uv must contain [u, v]")
        return float(supplied[0]), float(supplied[1])

    bbox = detection.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        raise GeolocationError("detection must contain bbox_xyxy=[x1,y1,x2,y2]")

    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        raise GeolocationError("bbox_xyxy has non-positive width or height")

    u = (x1 + x2) * 0.5
    if default_anchor == "center":
        return u, (y1 + y2) * 0.5
    if default_anchor == "bottom_center":
        return u, y2
    raise GeolocationError(f"unsupported bbox anchor: {default_anchor}")


def project_pixel_to_ground(
    *,
    u: float,
    v: float,
    pose: VehiclePose,
    vehicle_agl_m: float,
    calibration: CameraCalibration,
    min_down_component: float = 0.1,
) -> GroundProjection:
    """Project one image pixel onto a locally flat ground plane.

    Camera axes follow OpenCV (x right, y down, z forward). Vehicle axes are
    ArduPilot FRD and the earth-local frame is NED.
    """
    if vehicle_agl_m <= 0:
        raise GeolocationError("vehicle AGL must be positive")
    if not (0 <= u < calibration.image_width and 0 <= v < calibration.image_height):
        raise GeolocationError("pixel is outside the calibrated image")

    pixel = np.array([[[u, v]]], dtype=np.float64)
    normalized = cv2.undistortPoints(
        pixel,
        calibration.camera_matrix,
        calibration.distortion,
    )[0, 0]
    ray_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)

    rotation = rotation_ned_from_body(
        pose.roll_rad,
        pose.pitch_rad,
        pose.yaw_rad,
    )
    camera_offset_ned = rotation @ calibration.lever_arm_body_m
    ray_body = calibration.rotation_body_from_camera @ ray_camera
    ray_ned = rotation @ ray_body

    down = float(ray_ned[2])
    if down < min_down_component:
        raise GeolocationError(
            f"camera ray is too close to/above the horizon (down={down:.3f})"
        )

    camera_agl_m = vehicle_agl_m - float(camera_offset_ned[2])
    if camera_agl_m <= 0:
        raise GeolocationError("calculated camera AGL is not positive")

    distance_along_ray = camera_agl_m / down
    north_m = float(camera_offset_ned[0] + distance_along_ray * ray_ned[0])
    east_m = float(camera_offset_ned[1] + distance_along_ray * ray_ned[1])

    latitude_rad = math.radians(pose.latitude_deg)
    latitude = pose.latitude_deg + math.degrees(north_m / EARTH_RADIUS_M)
    longitude = pose.longitude_deg + math.degrees(
        east_m / (EARTH_RADIUS_M * math.cos(latitude_rad))
    )

    return GroundProjection(
        latitude_deg=latitude,
        longitude_deg=longitude,
        north_m=north_m,
        east_m=east_m,
        slant_range_m=float(distance_along_ray),
        ray_down_component=down,
        ray_camera=tuple(float(value) for value in ray_camera),
        ray_body=tuple(float(value) for value in ray_body),
        ray_ned=tuple(float(value) for value in ray_ned),
        camera_offset_ned_m=tuple(float(value) for value in camera_offset_ned),
        camera_agl_m=camera_agl_m,
    )


def ensure_image_matches(
    image: Mapping[str, Any], calibration: CameraCalibration
) -> None:
    width = int(image.get("width", -1))
    height = int(image.get("height", -1))
    if (width, height) != (calibration.image_width, calibration.image_height):
        raise GeolocationError(
            "frame resolution "
            f"{width}x{height} does not match calibration "
            f"{calibration.image_width}x{calibration.image_height}"
        )


def as_float_sequence(value: Sequence[Any], length: int, name: str) -> list[float]:
    if len(value) != length:
        raise GeolocationError(f"{name} must contain {length} values")
    return [float(item) for item in value]
