from __future__ import annotations

import argparse
import cv2
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO

from target_geolocation.systemd_notify import SystemdNotifier


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def class_name(model: YOLO, class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detections_from_result(result, model: YOLO) -> list[dict]:
    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections

    image_height, image_width = result.orig_shape
    for detection_id, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        track_id = None if box.id is None else int(box.id[0].item())

        anchor_u = min(max((x1 + x2) * 0.5, 0.0), image_width - 1.0)
        anchor_v = min(max(y2, 0.0), image_height - 1.0)
        detections.append(
            {
                "detection_id": detection_id,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name(model, class_id),
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2],
                # 对地面目标默认使用bbox底边中心作为接地点。
                "ground_anchor_uv": [anchor_u, anchor_v],
            }
        )

    return detections


def build_frame_message(
    *,
    frame_id: int,
    capture_monotonic_ns: int,
    frame,
    result,
    model: YOLO,
    camera_id: str,
) -> dict:
    height, width = frame.shape[:2]
    return {
        "version": 1,
        "camera_id": camera_id,
        "frame_id": frame_id,
        "capture_monotonic_ns": capture_monotonic_ns,
        "capture_time_semantics": "opencv_read_return",
        "image": {"width": width, "height": height},
        "detections": detections_from_result(result, model),
    }


class FrameRecorder:
    """Record lightweight bbox logs and selected images for offline review."""

    def __init__(
        self,
        directory: str,
        *,
        image_mode: str,
        image_fps: float,
        annotated: bool,
        jpeg_quality: int,
        min_free_disk_gb: float,
        session: dict,
    ) -> None:
        self.directory = Path(directory)
        self.frames_directory = self.directory / "frames"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_directory.mkdir(parents=True, exist_ok=True)
        self.image_mode = image_mode
        self.minimum_image_interval_ns = (
            round(1e9 / image_fps) if image_fps > 0 else 0
        )
        self.annotated = annotated
        self.jpeg_quality = jpeg_quality
        self.min_free_disk_bytes = round(min_free_disk_gb * 1024**3)
        self.low_disk_warned = False
        self.last_image_ns: int | None = None
        self.log_file = (self.directory / "bbox.ndjson").open(
            "a", encoding="utf-8"
        )
        self.pending_lines = 0
        self.last_flush_ns = time.monotonic_ns()
        (self.directory / "session.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _should_save_image(self, message: dict) -> bool:
        if self.image_mode == "none":
            return False
        if self.image_mode == "detections" and not message["detections"]:
            return False
        if (
            self.min_free_disk_bytes > 0
            and shutil.disk_usage(self.directory).free < self.min_free_disk_bytes
        ):
            if not self.low_disk_warned:
                print(
                    "Image recording paused because free disk space is below "
                    f"{self.min_free_disk_bytes / 1024**3:.1f} GiB"
                )
                self.low_disk_warned = True
            return False
        capture_ns = int(message["capture_monotonic_ns"])
        if (
            self.last_image_ns is not None
            and self.minimum_image_interval_ns > 0
            and capture_ns - self.last_image_ns < self.minimum_image_interval_ns
        ):
            return False
        self.last_image_ns = capture_ns
        return True

    def record(self, message: dict, frame, result) -> None:
        if self._should_save_image(message):
            frame_id = int(message["frame_id"])
            original_relative = Path("frames") / f"{frame_id:08d}.jpg"
            original_path = self.directory / original_relative
            options = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            if cv2.imwrite(str(original_path), frame, options):
                message.setdefault("recording", {})["image_file"] = str(
                    original_relative
                )
            else:
                print(f"Failed to record {original_path}")

            if self.annotated:
                annotated_relative = (
                    Path("frames") / f"{frame_id:08d}_annotated.jpg"
                )
                annotated_path = self.directory / annotated_relative
                if cv2.imwrite(str(annotated_path), result.plot(), options):
                    message.setdefault("recording", {})[
                        "annotated_image_file"
                    ] = str(annotated_relative)

        self.log_file.write(
            json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        self.pending_lines += 1
        now_ns = time.monotonic_ns()
        if self.pending_lines >= 30 or now_ns - self.last_flush_ns >= 1_000_000_000:
            self.log_file.flush()
            self.pending_lines = 0
            self.last_flush_ns = now_ns

    def close(self) -> None:
        self.log_file.flush()
        self.log_file.close()


class TrainingVideoRecorder:
    """Write raw, unannotated frames into short MJPEG AVI training segments."""

    def __init__(
        self,
        directory: str,
        *,
        width: int,
        height: int,
        fps: int,
        segment_seconds: float,
        jpeg_quality: int,
        min_free_disk_gb: float,
        max_total_gb: float,
        session: dict,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.minimum_frame_interval_ns = round(1e9 / fps)
        self.last_recorded_capture_ns: int | None = None
        self.segment_duration_ns = round(segment_seconds * 1e9)
        self.jpeg_quality = jpeg_quality
        self.min_free_disk_bytes = round(min_free_disk_gb * 1024**3)
        self.max_total_bytes = round(max_total_gb * 1024**3)
        self.writer = None
        self.segment_start_ns: int | None = None
        self.segment_frame_index = 0
        self.segment_index = self._next_segment_index()
        self.segment_relative: str | None = None
        self.disabled = False
        self.low_disk = False
        self.storage_available = True
        self.last_storage_check_ns = 0
        self.pending_lines = 0
        self.last_flush_ns = time.monotonic_ns()
        self.frames_log = (self.directory / "video_frames.ndjson").open(
            "a", encoding="utf-8"
        )
        (self.directory / "video_session.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _next_segment_index(self) -> int:
        indexes = []
        for path in self.directory.glob("segment_*.avi"):
            try:
                indexes.append(int(path.stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(indexes, default=-1) + 1

    def _close_segment(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.segment_start_ns = None
        self.segment_frame_index = 0
        self.segment_relative = None

    def _open_segment(self, capture_ns: int) -> bool:
        filename = f"segment_{self.segment_index:05d}.avi"
        path = self.directory / filename
        escaped_path = str(path).replace("\\", "\\\\").replace('"', '\\"')
        pipeline = (
            "appsrc ! "
            f"video/x-raw,format=(string)BGR,width=(int){self.width},"
            f"height=(int){self.height},framerate=(fraction){self.fps}/1 ! "
            "videoconvert ! video/x-raw,format=(string)I420 ! "
            f"nvjpegenc quality={self.jpeg_quality} ! avimux ! "
            f'filesink location="{escaped_path}" sync=false'
        )
        writer = cv2.VideoWriter(
            pipeline,
            cv2.CAP_GSTREAMER,
            0,
            float(self.fps),
            (self.width, self.height),
            True,
        )
        backend = "nvjpegenc"
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                float(self.fps),
                (self.width, self.height),
                True,
            )
            backend = "opencv_mjpg"

        if not writer.isOpened():
            writer.release()
            self.disabled = True
            print("Training video disabled: neither nvjpegenc nor MJPG writer opened")
            return False

        self.writer = writer
        self.segment_start_ns = capture_ns
        self.segment_frame_index = 0
        self.segment_relative = filename
        self.segment_index += 1
        print(f"Training video segment opened: {path} backend={backend}")
        return True

    def _storage_is_available(self) -> bool:
        now_ns = time.monotonic_ns()
        if now_ns - self.last_storage_check_ns < 5_000_000_000:
            return self.storage_available
        self.last_storage_check_ns = now_ns

        total_bytes = sum(
            path.stat().st_size for path in self.directory.glob("segment_*.avi")
        )
        if self.max_total_bytes > 0 and total_bytes >= self.max_total_bytes:
            print(
                "Training video stopped because this run reached its "
                f"{self.max_total_bytes / 1024**3:.1f} GiB limit"
            )
            self.disabled = True
            self.storage_available = False
            return False

        available = (
            self.min_free_disk_bytes <= 0
            or shutil.disk_usage(self.directory).free >= self.min_free_disk_bytes
        )
        if not available and not self.low_disk:
            print(
                "Training video paused because free disk space is below "
                f"{self.min_free_disk_bytes / 1024**3:.1f} GiB"
            )
        if available and self.low_disk:
            print("Training video resumed after disk space recovered")
        self.low_disk = not available
        self.storage_available = available
        return available

    def record(self, frame_id: int, capture_ns: int, frame) -> None:
        if self.disabled:
            return
        if (
            self.last_recorded_capture_ns is not None
            and capture_ns - self.last_recorded_capture_ns
            < self.minimum_frame_interval_ns
        ):
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            self.disabled = True
            self._close_segment()
            print(
                "Training video disabled: frame resolution changed from "
                f"{self.width}x{self.height} to {frame.shape[1]}x{frame.shape[0]}"
            )
            return
        if not self._storage_is_available():
            self._close_segment()
            return
        if (
            self.writer is None
            or self.segment_start_ns is None
            or capture_ns - self.segment_start_ns >= self.segment_duration_ns
        ):
            self._close_segment()
            if not self._open_segment(capture_ns):
                return

        self.writer.write(frame)
        self.last_recorded_capture_ns = capture_ns
        record = {
            "source_frame_id": frame_id,
            "capture_monotonic_ns": capture_ns,
            "capture_time_semantics": "opencv_read_return",
            "video_file": self.segment_relative,
            "video_frame_index": self.segment_frame_index,
        }
        self.frames_log.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.segment_frame_index += 1
        self.pending_lines += 1
        now_ns = time.monotonic_ns()
        if self.pending_lines >= 30 or now_ns - self.last_flush_ns >= 1_000_000_000:
            self.frames_log.flush()
            self.pending_lines = 0
            self.last_flush_ns = now_ns

    def close(self) -> None:
        self._close_segment()
        self.frames_log.flush()
        self.frames_log.close()


def main() -> int:
    from ultralytics import YOLO

    notifier = SystemdNotifier()

    parser = argparse.ArgumentParser(
        description="Run CSI YOLO tracking and send one bbox JSON datagram per frame"
    )
    parser.add_argument("--model", default="./model/exp-3.engine")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument("--camera-id", default="down_cam")
    parser.add_argument("--bbox-host", default="127.0.0.1")
    parser.add_argument("--bbox-port", type=int, default=15100)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--record-dir",
        help="record bbox.ndjson and selected images for offline debugging",
    )
    parser.add_argument(
        "--record-images",
        choices=("none", "detections", "all"),
        default="detections",
    )
    parser.add_argument(
        "--record-image-fps",
        type=float,
        default=2.0,
        help="maximum saved image rate; bbox JSON is still logged every frame",
    )
    parser.add_argument("--record-annotated", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=5.0,
        help="stop saving images below this free-space threshold; JSON continues",
    )
    parser.add_argument(
        "--training-video-dir",
        help="save raw, unannotated MJPEG AVI segments for model training",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="maximum recorded training-video frame rate",
    )
    parser.add_argument("--video-segment-seconds", type=float, default=60.0)
    parser.add_argument("--video-quality", type=int, default=85)
    parser.add_argument(
        "--video-max-total-gb",
        type=float,
        default=20.0,
        help="maximum training-video size for this run; 0 means unlimited",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 runs continuously; positive values are useful for testing",
    )
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.record_image_fps < 0:
        parser.error("--record-image-fps cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.min_free_disk_gb < 0:
        parser.error("--min-free-disk-gb cannot be negative")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if args.video_segment_seconds <= 0:
        parser.error("--video-segment-seconds must be positive")
    if not 1 <= args.video_quality <= 100:
        parser.error("--video-quality must be between 1 and 100")
    if args.video_max_total_gb < 0:
        parser.error("--video-max-total-gb cannot be negative")

    model = YOLO(args.model, task="detect")
    pipeline = gstreamer_pipeline(
        sensor_id=args.sensor_id,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        display_width=args.output_width,
        display_height=args.output_height,
        framerate=args.framerate,
        flip_method=args.flip_method,
    )
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("Cannot open CSI camera with GStreamer pipeline")

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.bbox_host, args.bbox_port)
    frame_id = 0
    systemd_ready = False
    recorder = None
    video_recorder = None
    if args.record_dir:
        recorder = FrameRecorder(
            args.record_dir,
            image_mode=args.record_images,
            image_fps=args.record_image_fps,
            annotated=args.record_annotated,
            jpeg_quality=args.jpeg_quality,
            min_free_disk_gb=args.min_free_disk_gb,
            session={
                "version": 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "camera_id": args.camera_id,
                "capture": {
                    "sensor_id": args.sensor_id,
                    "capture_width": args.capture_width,
                    "capture_height": args.capture_height,
                    "output_width": args.output_width,
                    "output_height": args.output_height,
                    "framerate": args.framerate,
                    "flip_method": args.flip_method,
                },
                "detection": {
                    "tracker": args.tracker,
                    "imgsz": args.imgsz,
                    "confidence": args.conf,
                },
            },
        )
    if args.training_video_dir:
        video_recorder = TrainingVideoRecorder(
            args.training_video_dir,
            width=args.output_width,
            height=args.output_height,
            fps=args.video_fps,
            segment_seconds=args.video_segment_seconds,
            jpeg_quality=args.video_quality,
            min_free_disk_gb=args.min_free_disk_gb,
            max_total_gb=args.video_max_total_gb,
            session={
                "version": 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "format": "MJPEG AVI",
                "raw_unannotated": True,
                "width": args.output_width,
                "height": args.output_height,
                "declared_fps": args.video_fps,
                "segment_seconds": args.video_segment_seconds,
                "jpeg_quality": args.video_quality,
                "max_total_gb": args.video_max_total_gb,
                "camera_id": args.camera_id,
                "sensor_id": args.sensor_id,
                "flip_method": args.flip_method,
            },
        )

    print(
        f"YOLO bbox sender ready: model={args.model}, "
        f"destination={destination[0]}:{destination[1]}"
    )

    try:
        while True:
            ok, frame = capture.read()
            # 当前相机驱动没有直接提供曝光时间，使用取帧返回时刻近似。
            capture_monotonic_ns = time.monotonic_ns()
            if not ok:
                print("Failed to read frame")
                return 1

            if video_recorder is not None:
                video_recorder.record(frame_id, capture_monotonic_ns, frame)

            result = model.track(
                source=frame,
                tracker=args.tracker,
                persist=True,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )[0]

            message = build_frame_message(
                frame_id=frame_id,
                capture_monotonic_ns=capture_monotonic_ns,
                frame=frame,
                result=result,
                model=model,
                camera_id=args.camera_id,
            )
            if recorder is not None:
                recorder.record(message, frame, result)
            payload = json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            udp.sendto(payload, destination)

            if not systemd_ready:
                notifier.ready(
                    f"camera and YOLO ready; sending bbox to {destination[0]}:{destination[1]}"
                )
                systemd_ready = True
            else:
                notifier.watchdog(
                    f"camera/YOLO healthy; processed frame {frame_id}"
                )

            if args.print_json:
                print(payload.decode("utf-8"), flush=True)

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break

            if not args.no_display:
                cv2.imshow("YOLO CSI", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        notifier.stopping("camera/YOLO stopping")
        capture.release()
        udp.close()
        if recorder is not None:
            recorder.close()
        if video_recorder is not None:
            video_recorder.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
