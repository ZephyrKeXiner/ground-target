from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from target_geolocation.core import CameraCalibration, GeolocationError


CONFIG_ERROR_EXIT = 78
PROJECT_DIR = Path(__file__).resolve().parent.parent


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def validate() -> tuple[Path, Path]:
    python_path = project_path(env("GROUND_TARGET_PYTHON", ".venv/bin/python"))
    config_path = project_path(
        env("GROUND_TARGET_CONFIG", "target_geolocation/config.json")
    )
    model_path = project_path(env("GROUND_TARGET_MODEL", "model/exp-3.engine"))

    if not python_path.is_file():
        raise GeolocationError(f"Python environment is missing: {python_path}")
    if not config_path.is_file():
        raise GeolocationError(
            f"calibration config is missing: {config_path}; copy and calibrate config.example.json"
        )
    if not model_path.is_file():
        raise GeolocationError(f"YOLO model is missing: {model_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        calibration = CameraCalibration.from_mapping(config["camera"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise GeolocationError(f"invalid calibration config: {exc}") from exc

    output_width = int(env("GROUND_TARGET_OUTPUT_WIDTH", "1280"))
    output_height = int(env("GROUND_TARGET_OUTPUT_HEIGHT", "720"))
    if (calibration.image_width, calibration.image_height) != (
        output_width,
        output_height,
    ):
        raise GeolocationError(
            "camera calibration resolution "
            f"{calibration.image_width}x{calibration.image_height} does not match "
            f"service output {output_width}x{output_height}"
        )

    return config_path, model_path


def prepare() -> int:
    validate()
    runs_directory = project_path(env("GROUND_TARGET_RUNS_DIR", "runs"))
    runtime_directory = Path(env("GROUND_TARGET_RUNTIME_DIR", "/run/ground-target"))
    runs_directory.mkdir(parents=True, exist_ok=True)
    runtime_directory.mkdir(parents=True, exist_ok=True)

    session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = runs_directory / session_name
    suffix = 1
    while run_directory.exists():
        run_directory = runs_directory / f"{session_name}-{suffix}"
        suffix += 1
    (run_directory / "camera").mkdir(parents=True)

    pointer = runtime_directory / "run-dir"
    temporary_pointer = runtime_directory / f"run-dir.{os.getpid()}.tmp"
    temporary_pointer.write_text(str(run_directory) + "\n", encoding="utf-8")
    os.replace(temporary_pointer, pointer)

    latest = runs_directory / "latest"
    temporary_link = runs_directory / f".latest.{os.getpid()}.tmp"
    temporary_link.symlink_to(run_directory.name, target_is_directory=True)
    os.replace(temporary_link, latest)
    print(f"prepared ground-target run directory: {run_directory}")
    return 0


def run_directory() -> Path:
    pointer = Path(env("GROUND_TARGET_RUNTIME_DIR", "/run/ground-target")) / "run-dir"
    try:
        path = Path(pointer.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise RuntimeError(
            f"run directory pointer is unavailable: {pointer}; prepare service did not run"
        ) from exc
    if not path.is_dir():
        raise RuntimeError(f"run directory does not exist: {path}")
    return path


def exec_controller() -> None:
    config_path, _ = validate()
    run = run_directory()
    python_path = project_path(env("GROUND_TARGET_PYTHON", ".venv/bin/python"))
    arguments = [
        str(python_path),
        "-m",
        "target_geolocation.controller",
        "--config",
        str(config_path),
        "--mavlink",
        env("GROUND_TARGET_MAVLINK", "udpin:0.0.0.0:14550"),
        "--source-system",
        env("GROUND_TARGET_SOURCE_SYSTEM", "245"),
        "--listen",
        env("GROUND_TARGET_BBOX_LISTEN", "127.0.0.1:15100"),
        "--output",
        str(run / "results.ndjson"),
        "--events",
        str(run / "events.ndjson"),
    ]
    os.chdir(PROJECT_DIR)
    os.execv(str(python_path), arguments)


def exec_yolo() -> None:
    _, model_path = validate()
    run = run_directory()
    python_path = project_path(env("GROUND_TARGET_PYTHON", ".venv/bin/python"))
    bbox_host, separator, bbox_port = env(
        "GROUND_TARGET_BBOX_DESTINATION", "127.0.0.1:15100"
    ).rpartition(":")
    if not separator:
        raise ValueError("GROUND_TARGET_BBOX_DESTINATION must be HOST:PORT")

    arguments = [
        str(python_path),
        str(PROJECT_DIR / "stream.py"),
        "--model",
        str(model_path),
        "--tracker",
        env("GROUND_TARGET_TRACKER", "bytetrack.yaml"),
        "--imgsz",
        env("GROUND_TARGET_IMG_SIZE", "640"),
        "--conf",
        env("GROUND_TARGET_CONFIDENCE", "0.60"),
        "--sensor-id",
        env("GROUND_TARGET_SENSOR_ID", "0"),
        "--capture-width",
        env("GROUND_TARGET_CAPTURE_WIDTH", "1280"),
        "--capture-height",
        env("GROUND_TARGET_CAPTURE_HEIGHT", "720"),
        "--output-width",
        env("GROUND_TARGET_OUTPUT_WIDTH", "1280"),
        "--output-height",
        env("GROUND_TARGET_OUTPUT_HEIGHT", "720"),
        "--framerate",
        env("GROUND_TARGET_FRAMERATE", "30"),
        "--flip-method",
        env("GROUND_TARGET_FLIP_METHOD", "0"),
        "--camera-id",
        env("GROUND_TARGET_CAMERA_ID", "down_cam"),
        "--bbox-host",
        bbox_host,
        "--bbox-port",
        bbox_port,
        "--no-display",
        "--record-dir",
        str(run / "camera"),
        "--record-images",
        env("GROUND_TARGET_RECORD_IMAGES", "detections"),
        "--record-image-fps",
        env("GROUND_TARGET_RECORD_IMAGE_FPS", "2"),
        "--jpeg-quality",
        env("GROUND_TARGET_JPEG_QUALITY", "90"),
        "--min-free-disk-gb",
        env("GROUND_TARGET_MIN_FREE_DISK_GB", "5"),
    ]
    if env_bool("GROUND_TARGET_RECORD_ANNOTATED"):
        arguments.append("--record-annotated")
    if env_bool("GROUND_TARGET_RECORD_VIDEO"):
        arguments.extend(
            [
                "--training-video-dir",
                str(run / "training_video"),
                "--video-fps",
                env("GROUND_TARGET_VIDEO_FPS", "10"),
                "--video-segment-seconds",
                env("GROUND_TARGET_VIDEO_SEGMENT_SECONDS", "60"),
                "--video-quality",
                env("GROUND_TARGET_VIDEO_QUALITY", "85"),
                "--video-max-total-gb",
                env("GROUND_TARGET_VIDEO_MAX_TOTAL_GB", "20"),
            ]
        )
    if env_bool("GROUND_TARGET_PRINT_JSON"):
        arguments.append("--print-json")

    os.chdir(PROJECT_DIR)
    os.execv(str(python_path), arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description="systemd entrypoint for ground-target")
    parser.add_argument("action", choices=("validate", "prepare", "controller", "yolo"))
    args = parser.parse_args()
    try:
        if args.action == "validate":
            config, model = validate()
            print(f"configuration valid: config={config}, model={model}")
            return 0
        if args.action == "prepare":
            return prepare()
        if args.action == "controller":
            exec_controller()
        if args.action == "yolo":
            exec_yolo()
    except (GeolocationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ground-target service configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
