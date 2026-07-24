from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from config import CropConfig
from detector import TargetDetector
from registry import TargetRegistry
from tracker import TargetTracker
from utils import crop_and_enhance_target


DEFAULT_VIDEO = (
    Path(__file__).resolve().parent.parent
    / "video"
    / "test_video.mp4"
)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    result = image.copy()
    cv.rectangle(result, (0, 0), (330, 35), (0, 0, 0), -1)
    cv.putText(
        result,
        title,
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return result


def build_view(frame: np.ndarray, debug_view: np.ndarray) -> np.ndarray:
    combined = np.hstack(
        [
            add_title(frame, "Original video"),
            add_title(debug_view, "Red / blue target mask"),
        ]
    )
    scale = min(1.0, 1400 / combined.shape[1], 800 / combined.shape[0])
    if scale < 1.0:
        combined = cv.resize(
            combined,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_AREA,
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help=f"视频文件路径（默认：{DEFAULT_VIDEO.name}）",
    )
    return parser.parse_args()


def run_video(video_path: Path) -> int:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return 1

    detector = TargetDetector()
    tracker = TargetTracker()
    registry = TargetRegistry()
    crop_config = CropConfig()
    confirmed_crops: dict[str, np.ndarray] = {}
    track_target_ids: dict[int, int] = {}

    paused = False
    display: np.ndarray | None = None

    print(f"Video: {video_path}")
    print("Controls: Space/P pause, Q/Esc quit")
    cv.namedWindow("Pentagon Fitting", cv.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                analysis = detector.analyze(frame)
                newly_confirmed = tracker.update(analysis.detections)

                for track in tracker.tracks:
                    target_id = track_target_ids.get(track.track_id)
                    if (
                        target_id is not None
                        and track.confirmed
                        and track.missed_frames == 0
                    ):
                        registry.observe(
                            target_id,
                            frame,
                            track.detection,
                        )

                visible_target_ids = {
                    track_target_ids[track.track_id]
                    for track in tracker.tracks
                    if (
                        track.confirmed
                        and track.missed_frames == 0
                        and track.track_id in track_target_ids
                    )
                }
                for track in newly_confirmed:
                    (
                        target_id,
                        is_new_target,
                        match_description,
                    ) = registry.resolve(
                        frame,
                        track.detection,
                        visible_target_ids,
                    )
                    track_target_ids[track.track_id] = target_id
                    visible_target_ids.add(target_id)

                    window_name = f"Target {target_id}"
                    confirmed_crops[window_name] = crop_and_enhance_target(
                        frame,
                        track.detection.polygon,
                        crop_config,
                    )
                    if is_new_target:
                        cv.namedWindow(
                            window_name,
                            cv.WINDOW_AUTOSIZE,
                        )
                        print(f"Confirmed new target {target_id}")
                    else:
                        print(
                            f"Reidentified target {target_id} "
                            f"({match_description})"
                        )

                tracker.draw_status(
                    analysis.debug_view,
                    track_target_ids,
                )
                display = build_view(frame, analysis.debug_view)

            if display is not None:
                cv.imshow("Pentagon Fitting", display)
            for window_name, crop in confirmed_crops.items():
                cv.imshow(window_name, crop)

            key = cv.waitKey(30 if paused else 1) & 0xFF
            if key in (ord(" "), ord("p"), ord("P")):
                paused = not paused
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    run_video(Path(args.video).expanduser().resolve())


if __name__ == "__main__":
    main()
