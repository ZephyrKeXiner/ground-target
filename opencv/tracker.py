from config import *
from detector import *
from utils import *

import cv2 as cv

@dataclass
class TargetTrack:
    track_id: int
    detection: Detection
    observed_frames: int = 1
    confirmed: bool = False
    missed_frames: int = 0
    
    
class TargetTracker:
    """匹配相邻帧目标，并控制连续帧确认与短暂漏检。"""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[TargetTrack] = []
        self.next_track_id = 1

    def update(
        self,
        strict_detections: list[Detection],
        perspective_detections: list[Detection],
    ) -> list[TargetTrack]:
        # 缺边/斜视候选只能续接已有轨迹，不能独立创建误报窗口。
        candidates = list(strict_detections)
        candidates.extend(
            detection
            for detection in perspective_detections
            if any(
                self._match_distance(detection, track) is not None
                for track in self.tracks
            )
        )

        current_tracks: list[TargetTrack] = []
        newly_confirmed: list[TargetTrack] = []
        used_previous: set[int] = set()

        # 先匹配可靠的严格候选；透视候选只用于填补它没有覆盖的旧轨迹。
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (item.mode != "strict", -item.area),
        )
        for detection in ordered_candidates:
            best_index: int | None = None
            best_distance = float("inf")

            for index, track in enumerate(self.tracks):
                if index in used_previous:
                    continue
                distance = self._match_distance(detection, track)
                if distance is not None and distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None:
                # 在轨迹创建点再次强制约束，避免多个候选争抢同一旧轨迹时，
                # 未匹配的透视候选意外获得新的 Target 编号。
                if detection.mode != "strict":
                    continue
                track = TargetTrack(self.next_track_id, detection)
                self.next_track_id += 1
            else:
                previous = self.tracks[best_index]
                used_previous.add(best_index)
                track = TargetTrack(
                    track_id=previous.track_id,
                    detection=detection,
                    observed_frames=min(
                        previous.observed_frames + 1,
                        self.config.confirmation_frames,
                    ),
                    confirmed=previous.confirmed,
                )

            if (
                track.observed_frames >= self.config.confirmation_frames
                and not track.confirmed
            ):
                track.confirmed = True
                newly_confirmed.append(track)
            current_tracks.append(track)

        for index, previous in enumerate(self.tracks):
            if index in used_previous:
                continue
            missed_frames = previous.missed_frames + 1
            if missed_frames <= self.config.max_missed_frames:
                current_tracks.append(
                    TargetTrack(
                        track_id=previous.track_id,
                        detection=previous.detection,
                        observed_frames=previous.observed_frames,
                        confirmed=previous.confirmed,
                        missed_frames=missed_frames,
                    )
                )

        self.tracks = current_tracks
        return newly_confirmed

    def draw_status(self, image: np.ndarray) -> None:
        for track in self.tracks:
            x, y, _, height = track.detection.bounding_box
            if track.missed_frames > 0:
                text = (
                    f"ID {track.track_id}: miss "
                    f"{track.missed_frames}/{self.config.max_missed_frames}"
                )
                color = (0, 165, 255)
            elif track.confirmed:
                text = f"ID {track.track_id}: CONFIRMED"
                color = (0, 255, 0)
            else:
                text = (
                    f"ID {track.track_id}: {track.observed_frames}/"
                    f"{self.config.confirmation_frames}"
                )
                color = (0, 255, 255)

            cv.putText(
                image,
                text,
                (x, y + height + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv.LINE_AA,
            )

    def _match_distance(
        self,
        detection: Detection,
        track: TargetTrack,
    ) -> float | None:
        x, y, width, height = detection.bounding_box
        old_x, old_y, old_width, old_height = track.detection.bounding_box

        current_area = width * height
        old_area = old_width * old_height
        if min(current_area, old_area) == 0:
            return None
        if max(current_area, old_area) / min(current_area, old_area) > self.config.max_size_ratio:
            return None

        center = (x + width / 2, y + height / 2)
        old_center = (old_x + old_width / 2, old_y + old_height / 2)
        distance = float(np.hypot(center[0] - old_center[0], center[1] - old_center[1]))
        max_size = max(width, height, old_width, old_height)
        allowed_distance = (
            self.config.match_distance_ratio
            * max_size
            * (track.missed_frames + 1)
        )
        return distance if distance <= allowed_distance else None
