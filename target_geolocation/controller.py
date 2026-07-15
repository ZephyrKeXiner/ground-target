from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Optional

try:
    from pymavlink import mavutil
except ImportError:  # permits pure JSON/projection tests without pymavlink installed
    mavutil = None

from target_geolocation.core import (
    EARTH_RADIUS_M,
    CameraCalibration,
    GeolocationError,
    VehiclePose,
    bbox_anchor,
    ensure_image_matches,
    project_pixel_to_ground,
)
from target_geolocation.systemd_notify import SystemdNotifier


@dataclass(frozen=True)
class TimedAttitude:
    boot_s: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_rad_s: float
    pitch_rate_rad_s: float
    yaw_rate_rad_s: float


@dataclass(frozen=True)
class TimedPosition:
    boot_s: float
    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float
    relative_alt_m: float
    velocity_north_mps: float
    velocity_east_mps: float
    velocity_down_mps: float
    velocity_valid: bool


@dataclass(frozen=True)
class TimedScalar:
    boot_s: float
    value: float


class ClockAligner:
    """Estimate Jetson monotonic time = FC boot time + offset.

    The minimum recent receive offset approximates the clock offset while
    rejecting positive network/scheduling delay. TIMESYNC is still preferred
    for final accuracy.
    """

    def __init__(self, max_samples: int = 300) -> None:
        self._offset_candidates: deque[float] = deque(maxlen=max_samples)

    def update(self, boot_s: float, receive_monotonic_s: float) -> None:
        self._offset_candidates.append(receive_monotonic_s - boot_s)

    @property
    def ready(self) -> bool:
        return bool(self._offset_candidates)

    @property
    def offset_s(self) -> float:
        if not self._offset_candidates:
            raise RuntimeError("flight-controller clock has not been observed")
        return min(self._offset_candidates)

    def monotonic_to_boot(self, monotonic_s: float) -> float:
        return monotonic_s - self.offset_s

    def diagnostics(self) -> dict[str, Any]:
        values = sorted(self._offset_candidates)
        if not values:
            return {"sample_count": 0}

        minimum = values[0]

        def percentile(fraction: float) -> float:
            index = min(len(values) - 1, round((len(values) - 1) * fraction))
            return values[index]

        return {
            "sample_count": len(values),
            "offset_s": minimum,
            "median_excess_delay_ms": (percentile(0.5) - minimum) * 1000.0,
            "p95_excess_delay_ms": (percentile(0.95) - minimum) * 1000.0,
        }


class TelemetryBuffer:
    def __init__(self, max_samples: int = 2000) -> None:
        self._lock = threading.Lock()
        self._attitudes: deque[TimedAttitude] = deque(maxlen=max_samples)
        self._positions: deque[TimedPosition] = deque(maxlen=max_samples)
        self._terrain_height: deque[TimedScalar] = deque(maxlen=max_samples)
        self._gps_h_acc: deque[TimedScalar] = deque(maxlen=max_samples)
        self.clock = ClockAligner()

    def ingest(self, message: Any, receive_monotonic_s: float) -> None:
        message_type = message.get_type()
        time_boot_ms = getattr(message, "time_boot_ms", None)

        with self._lock:
            if time_boot_ms is not None:
                boot_s = float(time_boot_ms) / 1000.0
                self.clock.update(boot_s, receive_monotonic_s)
            elif self.clock.ready:
                boot_s = self.clock.monotonic_to_boot(receive_monotonic_s)
            else:
                return

            if message_type == "ATTITUDE":
                self._attitudes.append(
                    TimedAttitude(
                        boot_s,
                        float(message.roll),
                        float(message.pitch),
                        float(message.yaw),
                        float(getattr(message, "rollspeed", 0.0)),
                        float(getattr(message, "pitchspeed", 0.0)),
                        float(getattr(message, "yawspeed", 0.0)),
                    )
                )
            elif message_type == "GLOBAL_POSITION_INT":
                raw_vx = getattr(message, "vx", None)
                raw_vy = getattr(message, "vy", None)
                raw_vz = getattr(message, "vz", None)
                velocity_valid = all(
                    value is not None and int(value) != 32767
                    for value in (raw_vx, raw_vy, raw_vz)
                )
                self._positions.append(
                    TimedPosition(
                        boot_s,
                        float(message.lat) / 1e7,
                        float(message.lon) / 1e7,
                        float(message.alt) / 1000.0,
                        float(message.relative_alt) / 1000.0,
                        float(raw_vx) / 100.0 if velocity_valid else 0.0,
                        float(raw_vy) / 100.0 if velocity_valid else 0.0,
                        float(raw_vz) / 100.0 if velocity_valid else 0.0,
                        velocity_valid,
                    )
                )
            elif message_type == "TERRAIN_REPORT":
                height_m = float(message.current_height)
                if height_m > 0:
                    self._terrain_height.append(TimedScalar(boot_s, height_m))
            elif message_type == "GPS_RAW_INT":
                h_acc_mm = getattr(message, "h_acc", None)
                if h_acc_mm not in (None, 0, 0xFFFFFFFF):
                    self._gps_h_acc.append(
                        TimedScalar(boot_s, float(h_acc_mm) / 1000.0)
                    )

    def capture_boot_time(
        self, frame: Mapping[str, Any], receive_monotonic_s: float
    ) -> tuple[float, str]:
        if "capture_time_boot_ms" in frame:
            return float(frame["capture_time_boot_ms"]) / 1000.0, "fc_boot"

        key = "capture_monotonic_ns"
        if key not in frame and "capture_time_ns" in frame:
            key = "capture_time_ns"
        if key in frame:
            monotonic_s = float(frame[key]) / 1e9
            return self.clock.monotonic_to_boot(monotonic_s), "jetson_monotonic"

        return self.clock.monotonic_to_boot(receive_monotonic_s), "receive_time_fallback"

    def pose_at(self, boot_s: float) -> tuple[VehiclePose, TimedPosition, float]:
        with self._lock:
            attitude, attitude_age = _interpolate_attitude(self._attitudes, boot_s)
            position, position_age = _interpolate_position(self._positions, boot_s)

        if attitude is None:
            raise GeolocationError("no ATTITUDE samples around capture time")
        if position is None:
            raise GeolocationError("no GLOBAL_POSITION_INT samples around capture time")

        return (
            VehiclePose(
                latitude_deg=position.latitude_deg,
                longitude_deg=position.longitude_deg,
                altitude_msl_m=position.altitude_msl_m,
                roll_rad=attitude.roll_rad,
                pitch_rad=attitude.pitch_rad,
                yaw_rad=attitude.yaw_rad,
                roll_rate_rad_s=attitude.roll_rate_rad_s,
                pitch_rate_rad_s=attitude.pitch_rate_rad_s,
                yaw_rate_rad_s=attitude.yaw_rate_rad_s,
            ),
            position,
            max(attitude_age, position_age),
        )

    def agl_at(
        self,
        boot_s: float,
        position: TimedPosition,
        geolocation_config: Mapping[str, Any],
    ) -> tuple[float, str, float]:
        source = str(geolocation_config.get("agl_source", "auto"))
        max_sensor_age_s = float(
            geolocation_config.get("max_altitude_sensor_age_ms", 500)
        ) / 1000.0

        with self._lock:
            terrain, terrain_age = _nearest_scalar(self._terrain_height, boot_s)

        if source in ("auto", "terrain") and terrain is not None:
            if terrain_age <= max_sensor_age_s:
                return terrain.value, "terrain", terrain_age
            if source == "terrain":
                raise GeolocationError("terrain-height sample is stale")

        if source in ("auto", "ground_alt_msl"):
            ground_alt = geolocation_config.get("ground_alt_msl_m")
            if ground_alt is not None:
                agl = position.altitude_msl_m - float(ground_alt)
                if agl > 0:
                    return agl, "ground_alt_msl", 0.0
            if source == "ground_alt_msl":
                raise GeolocationError("ground_alt_msl_m is missing or above vehicle")

        if source in ("auto", "fixed"):
            fixed_agl = geolocation_config.get("fixed_agl_m")
            if fixed_agl is not None and float(fixed_agl) > 0:
                return float(fixed_agl), "fixed", 0.0
            if source == "fixed":
                raise GeolocationError("fixed_agl_m must be positive")

        if source in ("auto", "relative_alt") and geolocation_config.get(
            "allow_relative_alt_fallback", False
        ):
            if position.relative_alt_m > 0:
                return position.relative_alt_m, "relative_alt_fallback", 0.0

        raise GeolocationError(
            "no valid AGL source; configure terrain, ground_alt_msl_m, fixed_agl_m, "
            "or explicitly allow relative_alt fallback"
        )

    def gps_h_acc_at(self, boot_s: float) -> Optional[float]:
        with self._lock:
            sample, _ = _nearest_scalar(self._gps_h_acc, boot_s)
        return sample.value if sample is not None else None

    def clock_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return self.clock.diagnostics()


def _bracket(samples: Iterable[Any], boot_s: float) -> tuple[Any, Any, float, float]:
    values = list(samples)
    if not values:
        return None, None, math.inf, 0.0

    times = [item.boot_s for item in values]
    index = bisect_left(times, boot_s)
    if index == 0:
        return values[0], values[0], abs(values[0].boot_s - boot_s), 0.0
    if index == len(values):
        return values[-1], values[-1], abs(values[-1].boot_s - boot_s), 0.0

    left, right = values[index - 1], values[index]
    span = right.boot_s - left.boot_s
    ratio = 0.0 if span <= 0 else (boot_s - left.boot_s) / span
    age = min(boot_s - left.boot_s, right.boot_s - boot_s)
    return left, right, age, ratio


def _lerp(left: float, right: float, ratio: float) -> float:
    return left + (right - left) * ratio


def _lerp_angle(left: float, right: float, ratio: float) -> float:
    delta = (right - left + math.pi) % (2 * math.pi) - math.pi
    return left + delta * ratio


def _interpolate_attitude(
    samples: Iterable[TimedAttitude], boot_s: float
) -> tuple[Optional[TimedAttitude], float]:
    left, right, age, ratio = _bracket(samples, boot_s)
    if left is None:
        return None, age
    return (
        TimedAttitude(
            boot_s,
            _lerp(left.roll_rad, right.roll_rad, ratio),
            _lerp(left.pitch_rad, right.pitch_rad, ratio),
            _lerp_angle(left.yaw_rad, right.yaw_rad, ratio),
            _lerp(left.roll_rate_rad_s, right.roll_rate_rad_s, ratio),
            _lerp(left.pitch_rate_rad_s, right.pitch_rate_rad_s, ratio),
            _lerp(left.yaw_rate_rad_s, right.yaw_rate_rad_s, ratio),
        ),
        age,
    )


def _interpolate_position(
    samples: Iterable[TimedPosition], boot_s: float
) -> tuple[Optional[TimedPosition], float]:
    left, right, age, ratio = _bracket(samples, boot_s)
    if left is None:
        return None, age
    return (
        TimedPosition(
            boot_s,
            _lerp(left.latitude_deg, right.latitude_deg, ratio),
            _lerp(left.longitude_deg, right.longitude_deg, ratio),
            _lerp(left.altitude_msl_m, right.altitude_msl_m, ratio),
            _lerp(left.relative_alt_m, right.relative_alt_m, ratio),
            _lerp(left.velocity_north_mps, right.velocity_north_mps, ratio),
            _lerp(left.velocity_east_mps, right.velocity_east_mps, ratio),
            _lerp(left.velocity_down_mps, right.velocity_down_mps, ratio),
            left.velocity_valid and right.velocity_valid,
        ),
        age,
    )


def _nearest_scalar(
    samples: Iterable[TimedScalar], boot_s: float
) -> tuple[Optional[TimedScalar], float]:
    left, right, _, _ = _bracket(samples, boot_s)
    if left is None:
        return None, math.inf
    selected = min((left, right), key=lambda item: abs(item.boot_s - boot_s))
    return selected, abs(selected.boot_s - boot_s)


RECORDED_TELEMETRY_TYPES = {
    "ATTITUDE",
    "GLOBAL_POSITION_INT",
    "TERRAIN_REPORT",
    "GPS_RAW_INT",
}


class EventLogger:
    """Thread-safe NDJSON recorder used for deterministic offline replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8")
        self._pending_lines = 0
        self._last_flush_s = time.monotonic()

    def write(self, event: Mapping[str, Any]) -> None:
        record = dict(event)
        record.setdefault("logged_monotonic_ns", time.monotonic_ns())
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._pending_lines += 1
            now = time.monotonic()
            if self._pending_lines >= 50 or now - self._last_flush_s >= 1.0:
                self._file.flush()
                self._pending_lines = 0
                self._last_flush_s = now

    def close(self) -> None:
        with self._lock:
            self._file.flush()
            self._file.close()


def _telemetry_fields(message: Any, message_type: str) -> dict[str, Any]:
    fields_by_type = {
        "ATTITUDE": (
            "time_boot_ms",
            "roll",
            "pitch",
            "yaw",
            "rollspeed",
            "pitchspeed",
            "yawspeed",
        ),
        "GLOBAL_POSITION_INT": (
            "time_boot_ms",
            "lat",
            "lon",
            "alt",
            "relative_alt",
            "vx",
            "vy",
            "vz",
            "hdg",
        ),
        "TERRAIN_REPORT": (
            "lat",
            "lon",
            "spacing",
            "terrain_height",
            "current_height",
            "pending",
            "loaded",
        ),
        "GPS_RAW_INT": (
            "time_usec",
            "fix_type",
            "lat",
            "lon",
            "alt",
            "eph",
            "epv",
            "vel",
            "cog",
            "satellites_visible",
            "h_acc",
            "v_acc",
            "vel_acc",
            "hdg_acc",
        ),
    }
    return {
        name: getattr(message, name)
        for name in fields_by_type.get(message_type, ())
        if hasattr(message, name)
    }


class MavlinkReceiver(threading.Thread):
    def __init__(
        self,
        connection_string: str,
        source_system: int,
        telemetry: TelemetryBuffer,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.connection_string = connection_string
        self.source_system = source_system
        self.telemetry = telemetry
        self.event_logger = event_logger
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error: Optional[BaseException] = None
        self.connection: Any = None

    def run(self) -> None:
        try:
            if mavutil is None:
                raise RuntimeError("pymavlink is required to receive flight telemetry")
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self.source_system,
                autoreconnect=True,
            )
            heartbeat = self.connection.wait_heartbeat(timeout=15)
            if heartbeat is None:
                raise TimeoutError("no ArduPilot heartbeat within 15 seconds")
            self.ready_event.set()

            while not self.stop_event.is_set():
                message = self.connection.recv_match(blocking=True, timeout=0.5)
                if message is None or message.get_type() == "BAD_DATA":
                    continue
                receive_monotonic_s = time.monotonic()
                self.telemetry.ingest(message, receive_monotonic_s)
                message_type = message.get_type()
                if (
                    self.event_logger is not None
                    and message_type in RECORDED_TELEMETRY_TYPES
                ):
                    self.event_logger.write(
                        {
                            "event": "telemetry",
                            "receive_monotonic_ns": round(
                                receive_monotonic_s * 1e9
                            ),
                            "message_type": message_type,
                            "message": _telemetry_fields(message, message_type),
                        }
                    )
        except BaseException as exc:  # report thread failures to the main loop
            self.error = exc
            self.ready_event.set()

    def stop(self) -> None:
        self.stop_event.set()


def _parse_listen(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError("listen address must be HOST:PORT")
    return host or "0.0.0.0", int(port)


def _result_base(frame: Mapping[str, Any], detection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "target_geolocation",
        "frame_id": frame.get("frame_id"),
        "camera_id": frame.get("camera_id"),
        "detection_id": detection.get("detection_id"),
        "track_id": detection.get("track_id"),
        "class_id": detection.get("class_id"),
        "class_name": detection.get("class_name"),
        "confidence": detection.get("confidence"),
    }


def _shift_entry(north_m: float, east_m: float) -> dict[str, Any]:
    return {
        "delta_north_m": north_m,
        "delta_east_m": east_m,
        "magnitude_m": math.hypot(north_m, east_m),
    }


def _projection_shift(reference: Any, shifted: Any, denominator: float = 1.0) -> dict[str, Any]:
    latitude_rad = math.radians(reference.latitude_deg)
    return _shift_entry(
        math.radians(shifted.latitude_deg - reference.latitude_deg)
        * EARTH_RADIUS_M
        / denominator,
        math.radians(shifted.longitude_deg - reference.longitude_deg)
        * EARTH_RADIUS_M
        * math.cos(latitude_rad)
        / denominator,
    )


def _central_projection_shift(reference: Any, positive: Any, negative: Any) -> dict[str, Any]:
    latitude_rad = math.radians(reference.latitude_deg)
    return _shift_entry(
        math.radians(positive.latitude_deg - negative.latitude_deg)
        * EARTH_RADIUS_M
        * 0.5,
        math.radians(positive.longitude_deg - negative.longitude_deg)
        * EARTH_RADIUS_M
        * math.cos(latitude_rad)
        * 0.5,
    )


def _move_pose(pose: VehiclePose, position: TimedPosition, seconds: float) -> VehiclePose:
    north_m = position.velocity_north_mps * seconds
    east_m = position.velocity_east_mps * seconds
    latitude_rad = math.radians(pose.latitude_deg)
    return replace(
        pose,
        latitude_deg=pose.latitude_deg + math.degrees(north_m / EARTH_RADIUS_M),
        longitude_deg=pose.longitude_deg
        + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(latitude_rad))),
        roll_rad=pose.roll_rad + pose.roll_rate_rad_s * seconds,
        pitch_rad=pose.pitch_rad + pose.pitch_rate_rad_s * seconds,
        yaw_rad=pose.yaw_rad + pose.yaw_rate_rad_s * seconds,
    )


def projection_sensitivity(
    *,
    u: float,
    v: float,
    pose: VehiclePose,
    position: TimedPosition,
    vehicle_agl_m: float,
    calibration: CameraCalibration,
    min_down_component: float,
    reference: Any,
) -> dict[str, Any]:
    """Return local target shifts caused by small positive input changes."""

    def project(test_u: float, test_v: float, test_pose: VehiclePose, agl: float):
        return project_pixel_to_ground(
            u=test_u,
            v=test_v,
            pose=test_pose,
            vehicle_agl_m=agl,
            calibration=calibration,
            min_down_component=min_down_component,
        )

    result: dict[str, Any] = {}
    angle_step = math.radians(1.0)
    for name, field in (("roll_1deg", "roll_rad"), ("pitch_1deg", "pitch_rad"), ("yaw_1deg", "yaw_rad")):
        positive = project(u, v, replace(pose, **{field: getattr(pose, field) + angle_step}), vehicle_agl_m)
        negative = project(u, v, replace(pose, **{field: getattr(pose, field) - angle_step}), vehicle_agl_m)
        result[name] = _central_projection_shift(reference, positive, negative)

    if 0.0 < u - 1.0 and u + 1.0 < calibration.image_width:
        result["anchor_u_1px"] = _central_projection_shift(
            reference,
            project(u + 1.0, v, pose, vehicle_agl_m),
            project(u - 1.0, v, pose, vehicle_agl_m),
        )
    elif u + 1.0 < calibration.image_width:
        result["anchor_u_1px"] = _projection_shift(
            reference, project(u + 1.0, v, pose, vehicle_agl_m)
        )

    if 0.0 < v - 1.0 and v + 1.0 < calibration.image_height:
        result["anchor_v_1px"] = _central_projection_shift(
            reference,
            project(u, v + 1.0, pose, vehicle_agl_m),
            project(u, v - 1.0, pose, vehicle_agl_m),
        )
    elif v + 1.0 < calibration.image_height:
        result["anchor_v_1px"] = _projection_shift(
            reference, project(u, v + 1.0, pose, vehicle_agl_m)
        )

    if vehicle_agl_m > 1.0:
        result["agl_1m"] = _central_projection_shift(
            reference,
            project(u, v, pose, vehicle_agl_m + 1.0),
            project(u, v, pose, vehicle_agl_m - 1.0),
        )
    else:
        result["agl_1m"] = _projection_shift(
            reference, project(u, v, pose, vehicle_agl_m + 1.0)
        )

    if position.velocity_valid:
        seconds = 0.010
        positive_agl = vehicle_agl_m - position.velocity_down_mps * seconds
        negative_agl = vehicle_agl_m + position.velocity_down_mps * seconds
        if positive_agl > 0 and negative_agl > 0:
            result["timestamp_10ms"] = _central_projection_shift(
                reference,
                project(u, v, _move_pose(pose, position, seconds), positive_agl),
                project(u, v, _move_pose(pose, position, -seconds), negative_agl),
            )

    return result


def diagnostic_warnings(
    *,
    timestamp_source: str,
    timestamp_semantics: str,
    pose_age_s: float,
    agl_source: str,
    ray_down_component: float,
    gps_h_acc_m: Optional[float],
    velocity_valid: bool,
) -> list[str]:
    warnings: list[str] = []
    if timestamp_source == "receive_time_fallback":
        warnings.append("没有采集时间戳，正在使用UDP接收时间")
    if timestamp_semantics in ("opencv_read_return", "unknown"):
        warnings.append("当前时间戳不是传感器曝光时间，运动时会产生沿航迹偏差")
    if pose_age_s > 0.05:
        warnings.append(f"姿态/位置样本距离图像时刻{pose_age_s * 1000.0:.1f}ms")
    if agl_source in ("fixed", "relative_alt_fallback"):
        warnings.append(f"高度来源为{agl_source}，地面起伏或起飞点误差会直接缩放投影")
    if ray_down_component < 0.5:
        warnings.append("相机射线较斜，角度和高度误差会被明显放大")
    if gps_h_acc_m is not None and gps_h_acc_m > 2.0:
        warnings.append(f"飞控GPS水平精度估计为{gps_h_acc_m:.2f}m")
    if not velocity_valid:
        warnings.append("没有有效速度，无法估算时间戳误差造成的位移")
    return warnings


def process_frame(
    *,
    frame: Mapping[str, Any],
    receive_monotonic_s: float,
    telemetry: TelemetryBuffer,
    calibration: CameraCalibration,
    geolocation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ensure_image_matches(frame.get("image", {}), calibration)
    boot_s, timestamp_source = telemetry.capture_boot_time(frame, receive_monotonic_s)
    pose, position, pose_age_s = telemetry.pose_at(boot_s)
    agl_m, agl_source, agl_age_s = telemetry.agl_at(
        boot_s, position, geolocation_config
    )

    max_pose_age_s = float(geolocation_config.get("max_pose_age_ms", 150)) / 1000.0
    if pose_age_s > max_pose_age_s:
        raise GeolocationError(
            f"pose sample is stale by {pose_age_s * 1000:.1f} ms"
        )

    default_anchor = str(geolocation_config.get("bbox_anchor", "bottom_center"))
    min_down = float(geolocation_config.get("min_ray_down_component", 0.1))
    gps_h_acc_m = telemetry.gps_h_acc_at(boot_s)
    timestamp_semantics = str(frame.get("capture_time_semantics", "unknown"))
    clock_diagnostics = telemetry.clock_diagnostics()
    diagnostics_enabled = bool(
        geolocation_config.get("debug_sensitivity", True)
    )

    results: list[dict[str, Any]] = []
    for detection in frame.get("detections", []):
        result = _result_base(frame, detection)
        try:
            u, v = bbox_anchor(detection, default_anchor)
            projection = project_pixel_to_ground(
                u=u,
                v=v,
                pose=pose,
                vehicle_agl_m=agl_m,
                calibration=calibration,
                min_down_component=min_down,
            )
            warnings = diagnostic_warnings(
                timestamp_source=timestamp_source,
                timestamp_semantics=timestamp_semantics,
                pose_age_s=pose_age_s,
                agl_source=agl_source,
                ray_down_component=projection.ray_down_component,
                gps_h_acc_m=gps_h_acc_m,
                velocity_valid=position.velocity_valid,
            )
            sensitivity: dict[str, Any] = {}
            sensitivity_error: Optional[str] = None
            if diagnostics_enabled:
                try:
                    sensitivity = projection_sensitivity(
                        u=u,
                        v=v,
                        pose=pose,
                        position=position,
                        vehicle_agl_m=agl_m,
                        calibration=calibration,
                        min_down_component=min_down,
                        reference=projection,
                    )
                except (GeolocationError, ValueError) as exc:
                    sensitivity_error = str(exc)
            result.update(
                {
                    "valid": True,
                    "capture_time_boot_ms": round(boot_s * 1000.0, 3),
                    "anchor_uv": [u, v],
                    "target": {
                        "latitude_deg": projection.latitude_deg,
                        "longitude_deg": projection.longitude_deg,
                    },
                    "offset_ned_m": [
                        projection.north_m,
                        projection.east_m,
                        agl_m,
                    ],
                    "slant_range_m": projection.slant_range_m,
                    "quality": {
                        "timestamp_source": timestamp_source,
                        "timestamp_semantics": timestamp_semantics,
                        "pose_age_ms": pose_age_s * 1000.0,
                        "agl_source": agl_source,
                        "agl_age_ms": agl_age_s * 1000.0,
                        "vehicle_agl_m": agl_m,
                        "ray_down_component": projection.ray_down_component,
                        "gps_h_acc_m": gps_h_acc_m,
                        "uncertainty_m": None,
                        "warnings": warnings,
                    },
                    "debug": {
                        "bbox_xyxy": detection.get("bbox_xyxy"),
                        "ground_anchor_uv_supplied": detection.get(
                            "ground_anchor_uv"
                        ),
                        "vehicle": {
                            "latitude_deg": pose.latitude_deg,
                            "longitude_deg": pose.longitude_deg,
                            "altitude_msl_m": pose.altitude_msl_m,
                            "roll_deg": math.degrees(pose.roll_rad),
                            "pitch_deg": math.degrees(pose.pitch_rad),
                            "yaw_deg": math.degrees(pose.yaw_rad),
                            "roll_rate_deg_s": math.degrees(
                                pose.roll_rate_rad_s
                            ),
                            "pitch_rate_deg_s": math.degrees(
                                pose.pitch_rate_rad_s
                            ),
                            "yaw_rate_deg_s": math.degrees(
                                pose.yaw_rate_rad_s
                            ),
                            "velocity_ned_mps": [
                                position.velocity_north_mps,
                                position.velocity_east_mps,
                                position.velocity_down_mps,
                            ],
                            "velocity_valid": position.velocity_valid,
                        },
                        "camera": {
                            "ray_camera": list(projection.ray_camera),
                            "ray_body": list(projection.ray_body),
                            "ray_ned": list(projection.ray_ned),
                            "camera_offset_ned_m": list(
                                projection.camera_offset_ned_m
                            ),
                            "camera_agl_m": projection.camera_agl_m,
                        },
                        "timing": {
                            "clock_alignment": clock_diagnostics,
                            "exposure_time_ns": frame.get("exposure_time_ns"),
                            "sensor_timestamp_ns": frame.get(
                                "sensor_timestamp_ns"
                            ),
                            "frame_readout_time_ns": frame.get(
                                "frame_readout_time_ns"
                            ),
                        },
                        "sensitivity": sensitivity,
                        "sensitivity_error": sensitivity_error,
                    },
                }
            )
        except (GeolocationError, ValueError, TypeError) as exc:
            result.update({"valid": False, "reason": str(exc)})
        results.append(result)

    return results


def main() -> int:
    notifier = SystemdNotifier()
    parser = argparse.ArgumentParser(
        description="Fuse bbox JSON with MAVLink telemetry and estimate target GPS"
    )
    parser.add_argument(
        "--config",
        default="target_geolocation/config.json",
        help="camera/geolocation JSON configuration",
    )
    parser.add_argument("--mavlink", default="udpin:0.0.0.0:14550")
    parser.add_argument("--source-system", type=int, default=245)
    parser.add_argument("--listen", type=_parse_listen, default=("127.0.0.1", 15100))
    parser.add_argument("--output", help="optional NDJSON output log")
    parser.add_argument(
        "--events",
        help="optional replayable NDJSON log containing telemetry and bbox inputs",
    )
    args = parser.parse_args()

    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        calibration = CameraCalibration.from_mapping(config["camera"])
        geolocation_config = config.get("geolocation", {})
    except (OSError, KeyError, json.JSONDecodeError, GeolocationError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    event_logger = EventLogger(args.events) if args.events else None
    if event_logger is not None:
        event_logger.write(
            {
                "event": "session_start",
                "wall_time_unix_ns": time.time_ns(),
                "mavlink": args.mavlink,
                "listen": f"{args.listen[0]}:{args.listen[1]}",
                "config": config,
            }
        )

    telemetry = TelemetryBuffer()
    receiver = MavlinkReceiver(
        args.mavlink,
        args.source_system,
        telemetry,
        event_logger=event_logger,
    )
    receiver.start()
    receiver.ready_event.wait(timeout=16)
    if receiver.error is not None:
        print(f"MAVLink error: {receiver.error}", file=sys.stderr)
        if event_logger is not None:
            event_logger.close()
        return 3
    if not receiver.ready_event.is_set():
        print("MAVLink receiver did not become ready", file=sys.stderr)
        receiver.stop()
        receiver.join(timeout=1.0)
        if event_logger is not None:
            event_logger.close()
        return 3

    listen_host, listen_port = args.listen
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_host, listen_port))
    sock.settimeout(0.5)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("a", encoding="utf-8")
    else:
        output_file = None

    print(
        f"ready: MAVLink={args.mavlink}, bbox UDP={listen_host}:{listen_port}, "
        f"camera={calibration.camera_id}",
        file=sys.stderr,
    )
    notifier.ready(
        f"MAVLink ready; listening for bbox on {listen_host}:{listen_port}"
    )

    try:
        while True:
            notifier.watchdog("controller telemetry and UDP loop healthy")
            if receiver.error is not None:
                raise receiver.error
            try:
                payload, sender = sock.recvfrom(65535)
            except socket.timeout:
                continue

            receive_monotonic_s = time.monotonic()
            try:
                frame = json.loads(payload.decode("utf-8"))
                if event_logger is not None:
                    event_logger.write(
                        {
                            "event": "bbox_frame",
                            "receive_monotonic_ns": round(
                                receive_monotonic_s * 1e9
                            ),
                            "sender": f"{sender[0]}:{sender[1]}",
                            "frame": frame,
                        }
                    )
                results = process_frame(
                    frame=frame,
                    receive_monotonic_s=receive_monotonic_s,
                    telemetry=telemetry,
                    calibration=calibration,
                    geolocation_config=geolocation_config,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, GeolocationError) as exc:
                results = [
                    {
                        "type": "target_geolocation",
                        "valid": False,
                        "reason": str(exc),
                        "sender": f"{sender[0]}:{sender[1]}",
                    }
                ]

            for result in results:
                line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                print(line, flush=True)
                if output_file is not None:
                    output_file.write(line + "\n")
                    output_file.flush()
                if event_logger is not None:
                    event_logger.write({"event": "result", "result": result})
    except KeyboardInterrupt:
        return 0
    finally:
        notifier.stopping("controller stopping")
        receiver.stop()
        receiver.join(timeout=1.0)
        sock.close()
        if output_file is not None:
            output_file.close()
        if event_logger is not None:
            event_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
