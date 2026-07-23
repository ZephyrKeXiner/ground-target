from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from detector import *
from config import *
from tracker import *

import cv2 as cv
import numpy as np
from ultralytics import YOLO

DEFAULT_VIDEO = Path(__file__).resolve().parent / "video" / "test_video.mp4"

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
    crop_config = CropConfig()
    confirmed_crops: dict[str, np.ndarray] = {}

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_delay = max(1, round(1000 / fps)) if fps > 0 else 30
    paused = False
    display: np.ndarray | None = None

    print(f"Video: {video_path}")
    print("Controls: Space/P pause, Q/Esc quit")
    cv.namedWindow("Pentagon Fitting", cv.WINDOW_NORMAL)
    # model = YOLO("../model/exp-seg-1.pt")

    try:
        while cap.isOpened():
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                analysis = detector.analyze(frame)
                newly_confirmed = tracker.update(
                    analysis.strict_detections,
                    analysis.perspective_detections,
                )
                tracker.draw_status(analysis.debug_view)

                for track in newly_confirmed:
                    window_name = f"Target {track.track_id}"
                    confirmed_crops[window_name] = crop_and_enhance_target(
                        frame,
                        track.detection.polygon,
                        crop_config,
                    )
                    # model.predict(
                    #     frame,
                        
                    # )
                    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
                    print(f"Confirmed target {track.track_id}")

                display = build_view(frame, analysis.debug_view)

            if display is not None:
                cv.imshow("Pentagon Fitting", display)
            for window_name, crop in confirmed_crops.items():
                cv.imshow(window_name, crop)

            key = cv.waitKey(frame_delay) & 0xFF
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
