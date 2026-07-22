from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from target_geolocation.systemd_notify import SystemdNotifier


ROTATION_FLIP_METHODS = {
    "none": 0,
    "counterclockwise90": 1,
    "rotate180": 2,
    "clockwise90": 3,
}
ROTATION_BODY_FROM_VIDEO_CAMERA = {
    "none": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "counterclockwise90": [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "rotate180": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    "clockwise90": [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
}


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def unique_video_path(
    directory: Path, width: int = 3280, height: int = 2464, fps: int = 21
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = directory / f"imx219_{timestamp}_{width}x{height}_{fps}fps"
    candidate = base.with_suffix(".mkv")
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{base.name}_{suffix}.mkv"
        suffix += 1
    return candidate


def gstreamer_command(
    *,
    output: Path,
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    bitrate_kbps: int,
    rotation: str,
    exposure_min_us: int,
    exposure_max_us: int,
) -> list[str]:
    caps = (
        "video/x-raw(memory:NVMM),"
        f"width=(int){width},height=(int){height},"
        f"format=(string)NV12,framerate=(fraction){fps}/1"
    )
    output_width, output_height = (
        (height, width)
        if rotation in {"counterclockwise90", "clockwise90"}
        else (width, height)
    )
    source_properties = [
        "nvarguscamerasrc",
        f"sensor-id={sensor_id}",
        "do-timestamp=true",
    ]
    if exposure_max_us > 0:
        source_properties.append(
            f"exposuretimerange={exposure_min_us * 1000} {exposure_max_us * 1000}"
        )
    return [
        "gst-launch-1.0",
        "-e",
        *source_properties,
        "!",
        caps,
        "!",
        "nvvidconv",
        f"flip-method={ROTATION_FLIP_METHODS[rotation]}",
        "!",
        (
            "video/x-raw,"
            f"width=(int){output_width},height=(int){output_height},"
            "format=(string)I420"
        ),
        "!",
        "queue",
        "max-size-buffers=4",
        "leaky=downstream",
        "!",
        "x264enc",
        f"bitrate={bitrate_kbps}",
        "speed-preset=ultrafast",
        "tune=zerolatency",
        f"key-int-max={fps * 2}",
        "threads=0",
        "!",
        "h264parse",
        "config-interval=-1",
        "!",
        "matroskamux",
        "!",
        "filesink",
        f"location={output}",
        "sync=false",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record full-resolution IMX219 H.264 MKV from service start"
    )
    parser.add_argument(
        "--output-dir",
        default=env("GROUND_TARGET_VIDEO_OUTPUT_DIR", "/home/argus/ai/training_videos"),
    )
    parser.add_argument(
        "--sensor-id", type=int, default=int(env("GROUND_TARGET_VIDEO_SENSOR_ID", "0"))
    )
    parser.add_argument(
        "--width", type=int, default=int(env("GROUND_TARGET_VIDEO_WIDTH", "3280"))
    )
    parser.add_argument(
        "--height", type=int, default=int(env("GROUND_TARGET_VIDEO_HEIGHT", "2464"))
    )
    parser.add_argument(
        "--fps", type=int, default=int(env("GROUND_TARGET_VIDEO_FPS", "21"))
    )
    parser.add_argument(
        "--bitrate-kbps",
        type=int,
        default=int(env("GROUND_TARGET_VIDEO_BITRATE_KBPS", "50000")),
        help="H.264 target bitrate in kbit/s",
    )
    parser.add_argument(
        "--rotation",
        choices=tuple(ROTATION_FLIP_METHODS),
        default=env("GROUND_TARGET_VIDEO_ROTATION", "counterclockwise90"),
        help="rotation applied to the saved video",
    )
    parser.add_argument(
        "--exposure-min-us",
        type=int,
        default=int(env("GROUND_TARGET_VIDEO_EXPOSURE_MIN_US", "13")),
    )
    parser.add_argument(
        "--exposure-max-us",
        type=int,
        default=int(env("GROUND_TARGET_VIDEO_EXPOSURE_MAX_US", "2000")),
        help="maximum automatic exposure in microseconds; 0 uses the sensor default",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=float(env("GROUND_TARGET_VIDEO_MIN_FREE_DISK_GB", "5")),
    )
    parser.add_argument(
        "--stall-timeout-seconds",
        type=float,
        default=float(env("GROUND_TARGET_VIDEO_STALL_TIMEOUT_SECONDS", "30")),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float(env("GROUND_TARGET_VIDEO_DURATION_SECONDS", "0")),
        help="0 records until the service is stopped",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height and fps must be positive")
    if args.bitrate_kbps <= 0:
        parser.error("bitrate must be positive")
    if args.exposure_min_us <= 0:
        parser.error("minimum exposure must be positive")
    if 0 < args.exposure_max_us < args.exposure_min_us:
        parser.error("maximum exposure must be 0 or at least the minimum exposure")
    if args.min_free_disk_gb < 0 or args.stall_timeout_seconds < 0:
        parser.error("disk threshold and stall timeout cannot be negative")
    if args.duration_seconds < 0:
        parser.error("duration cannot be negative")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    minimum_free_bytes = round(args.min_free_disk_gb * 1024**3)
    if shutil.disk_usage(output_directory).free < minimum_free_bytes:
        print(
            f"Not recording: free disk is below {args.min_free_disk_gb:.1f} GiB",
            file=sys.stderr,
        )
        return 0

    output_width, output_height = (
        (args.height, args.width)
        if args.rotation in {"counterclockwise90", "clockwise90"}
        else (args.width, args.height)
    )
    video_path = unique_video_path(
        output_directory, width=output_width, height=output_height, fps=args.fps
    )
    metadata_path = video_path.with_suffix(".json")
    command = gstreamer_command(
        output=video_path,
        sensor_id=args.sensor_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate_kbps=args.bitrate_kbps,
        rotation=args.rotation,
        exposure_min_us=args.exposure_min_us,
        exposure_max_us=args.exposure_max_us,
    )
    metadata = {
        "version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "video_file": video_path.name,
        "camera": {
            "sensor": "IMX219",
            "sensor_id": args.sensor_id,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "output_width": output_width,
            "output_height": output_height,
            "format": "H.264 Matroska",
            "encoder": "x264enc",
            "encoder_preset": "ultrafast",
            "bitrate_kbps": args.bitrate_kbps,
            "exposure_min_us": args.exposure_min_us,
            "exposure_max_us": args.exposure_max_us,
        },
        "mount": {
            "assumption": "lens down; image top points to aircraft left",
            "yaw_from_aircraft_nose_deg": -90.0,
            "rotation_body_from_raw_camera": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "video_transform": {
            "rotation": args.rotation,
            "description": (
                "saved video top points to aircraft nose"
                if args.rotation == "counterclockwise90"
                else "see rotation_body_from_video_camera"
            ),
            "rotation_body_from_video_camera": ROTATION_BODY_FROM_VIDEO_CAMERA[
                args.rotation
            ],
        },
        "capture_time_semantics": "GStreamer pipeline clock; no per-frame Argus metadata",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exposure_status = (
        f"exposure <= {args.exposure_max_us} us"
        if args.exposure_max_us > 0
        else "sensor default exposure range"
    )
    print(
        f"Recording IMX219 {args.width}x{args.height}@{args.fps} as "
        f"{output_width}x{output_height} {args.rotation}, H.264 "
        f"{args.bitrate_kbps} kbit/s, {exposure_status} to {video_path}"
    )
    process = subprocess.Popen(command)
    notifier = SystemdNotifier()
    stop_requested = False
    stop_reason = ""

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested, stop_reason
        if not stop_requested:
            stop_requested = True
            stop_reason = f"signal {signum}"
            if process.poll() is None:
                process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_s = time.monotonic()
    last_size = -1
    last_growth_s = started_s
    ready = False
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                if stop_requested:
                    notifier.stopping(f"recording stopped: {stop_reason}")
                    return 0
                print(
                    "Recorder pipeline exited unexpectedly "
                    f"with status {return_code}; requesting service restart",
                    file=sys.stderr,
                )
                return return_code if return_code != 0 else 1

            now_s = time.monotonic()
            size = video_path.stat().st_size if video_path.exists() else 0
            if size > last_size:
                last_size = size
                last_growth_s = now_s
                if size > 0 and not ready:
                    notifier.ready(
                        f"recording {args.width}x{args.height}@{args.fps}: {video_path.name}"
                    )
                    ready = True

            if (
                not stop_requested
                and args.stall_timeout_seconds > 0
                and now_s - last_growth_s >= args.stall_timeout_seconds
            ):
                stop_requested = True
                stop_reason = "video file stopped growing"
                print(stop_reason, file=sys.stderr)
                process.send_signal(signal.SIGINT)
                process.wait(timeout=10)
                return 1

            if (
                not stop_requested
                and args.duration_seconds > 0
                and now_s - started_s >= args.duration_seconds
            ):
                stop_requested = True
                stop_reason = "requested duration completed"
                process.send_signal(signal.SIGINT)
                continue

            if (
                not stop_requested
                and shutil.disk_usage(output_directory).free < minimum_free_bytes
            ):
                stop_requested = True
                stop_reason = "low disk space"
                print(
                    f"Stopping recording below {args.min_free_disk_gb:.1f} GiB free",
                    file=sys.stderr,
                )
                process.send_signal(signal.SIGINT)
                continue

            notifier.watchdog(
                f"recording {video_path.name}; {size / 1024**2:.1f} MiB"
            )
            time.sleep(1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return 1
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
