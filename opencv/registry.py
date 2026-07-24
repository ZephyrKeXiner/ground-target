from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from config import RegistryConfig
from utils import Detection


@dataclass(frozen=True)
class AppearanceSignature:
    image: np.ndarray
    hash_bits: np.ndarray
    sharpness: float


@dataclass(frozen=True)
class SceneSignature:
    points: np.ndarray
    descriptors: np.ndarray
    target_geometry: np.ndarray


@dataclass(frozen=True)
class SceneMatch:
    position_error: float
    inliers: int


@dataclass
class RegisteredTarget:
    target_id: int
    target_color: str
    appearance_signatures: list[AppearanceSignature] = field(
        default_factory=list,
    )
    scene_signatures: list[SceneSignature] = field(
        default_factory=list,
    )


class TargetRegistry:
    """用场景位置为主、目标外观为辅识别重新入画的目标。"""

    def __init__(
        self,
        config: RegistryConfig | None = None,
    ) -> None:
        self.config = config or RegistryConfig()
        self.targets: dict[int, RegisteredTarget] = {}
        self.next_target_id = 1
        self._orb = cv.ORB_create(
            nfeatures=self.config.scene_orb_features,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=self.config.scene_fast_threshold,
        )
        self._matcher = cv.BFMatcher(cv.NORM_HAMMING)

    def resolve(
        self,
        frame: np.ndarray,
        detection: Detection,
        excluded_target_ids: set[int] | None = None,
    ) -> tuple[int, bool, str | None]:
        """返回全局 ID、是否为新目标，以及匹配依据。"""
        appearance = self._create_appearance_signature(
            frame,
            detection,
        )
        scene = self._create_scene_signature(frame, detection)
        excluded = excluded_target_ids or set()
        candidates = [
            target
            for target in self.targets.values()
            if (
                target.target_id not in excluded
                and target.target_color == detection.target_color
            )
        ]

        scene_target, scene_match = self._find_scene_target(
            scene,
            candidates,
        )
        if scene_target is not None and scene_match is not None:
            self._store_appearance(scene_target, appearance)
            self._store_scene(scene_target, scene)
            return (
                scene_target.target_id,
                False,
                (
                    f"scene error={scene_match.position_error:.2f}, "
                    f"inliers={scene_match.inliers}"
                ),
            )

        appearance_target, appearance_match = (
            self._find_appearance_target(
                appearance,
                candidates,
            )
        )
        if (
            appearance_target is not None
            and appearance_match is not None
        ):
            hash_distance, correlation = appearance_match
            self._store_appearance(
                appearance_target,
                appearance,
            )
            self._store_scene(appearance_target, scene)
            return (
                appearance_target.target_id,
                False,
                (
                    f"appearance correlation={correlation:.2f}, "
                    f"hash={hash_distance}"
                ),
            )

        target = RegisteredTarget(
            target_id=self.next_target_id,
            target_color=detection.target_color,
        )
        self.next_target_id += 1
        self.targets[target.target_id] = target
        self._store_appearance(target, appearance)
        self._store_scene(target, scene)
        return target.target_id, True, None

    def observe(
        self,
        target_id: int,
        frame: np.ndarray,
        detection: Detection,
    ) -> None:
        """为正在画面中的目标保留若干最清晰的外观样本。"""
        target = self.targets.get(target_id)
        if target is None:
            return

        signature = self._create_appearance_signature(
            frame,
            detection,
        )
        self._store_appearance(target, signature)

    def _find_scene_target(
        self,
        candidate: SceneSignature | None,
        targets: list[RegisteredTarget],
    ) -> tuple[RegisteredTarget | None, SceneMatch | None]:
        if candidate is None:
            return None, None

        best_target: RegisteredTarget | None = None
        best_match: SceneMatch | None = None
        for target in targets:
            for reference in target.scene_signatures:
                match = self._match_scenes(reference, candidate)
                if match is None:
                    continue
                if (
                    best_match is None
                    or match.position_error
                    < best_match.position_error
                    or (
                        match.position_error
                        == best_match.position_error
                        and match.inliers > best_match.inliers
                    )
                ):
                    best_target = target
                    best_match = match
        return best_target, best_match

    def _find_appearance_target(
        self,
        candidate: AppearanceSignature | None,
        targets: list[RegisteredTarget],
    ) -> tuple[
        RegisteredTarget | None,
        tuple[int, float] | None,
    ]:
        if candidate is None:
            return None, None

        best_target: RegisteredTarget | None = None
        best_match: tuple[int, float] | None = None
        for target in targets:
            match = self._best_appearance_match(
                candidate,
                target.appearance_signatures,
            )
            if match is None:
                continue
            hash_distance, correlation = match
            if (
                best_match is None
                or correlation > best_match[1]
                or (
                    correlation == best_match[1]
                    and hash_distance < best_match[0]
                )
            ):
                best_target = target
                best_match = match
        return best_target, best_match

    def _best_appearance_match(
        self,
        candidate: AppearanceSignature,
        references: list[AppearanceSignature],
    ) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None

        for reference in references:
            hash_distance = int(
                np.count_nonzero(
                    candidate.hash_bits
                    != reference.hash_bits
                )
            )
            correlation = self._correlation(
                candidate.image,
                reference.image,
            )
            if (
                hash_distance
                > self.config.max_hash_distance
                or correlation
                < self.config.min_correlation
            ):
                continue

            if (
                best is None
                or correlation > best[1]
                or (
                    correlation == best[1]
                    and hash_distance < best[0]
                )
            ):
                best = hash_distance, correlation

        return best

    def _store_appearance(
        self,
        target: RegisteredTarget,
        signature: AppearanceSignature | None,
    ) -> None:
        if signature is None:
            return

        target.appearance_signatures.append(signature)
        target.appearance_signatures.sort(
            key=lambda item: item.sharpness,
            reverse=True,
        )
        del target.appearance_signatures[
            self.config.max_signatures_per_target :
        ]

    def _store_scene(
        self,
        target: RegisteredTarget,
        signature: SceneSignature | None,
    ) -> None:
        if signature is None:
            return
        target.scene_signatures.append(signature)
        del target.scene_signatures[
            : -self.config.max_scene_signatures_per_target
        ]

    def _create_appearance_signature(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> AppearanceSignature | None:
        body = self._rectify_body(frame, detection)
        if body is None or body.size == 0:
            return None

        gray = cv.cvtColor(body, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        left, top, right, bottom = self.config.number_roi
        x1 = round(np.clip(left, 0.0, 1.0) * width)
        y1 = round(np.clip(top, 0.0, 1.0) * height)
        x2 = round(np.clip(right, 0.0, 1.0) * width)
        y2 = round(np.clip(bottom, 0.0, 1.0) * height)
        number_region = gray[y1:y2, x1:x2]
        if number_region.size == 0:
            return None

        size = self.config.signature_size
        normalized = cv.resize(
            number_region,
            (size, size),
            interpolation=cv.INTER_CUBIC,
        )
        normalized = cv.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4),
        ).apply(normalized)
        sharpness = float(
            cv.Laplacian(normalized, cv.CV_64F).var()
        )

        frequency = cv.dct(normalized.astype(np.float32))
        low_frequency = frequency[:8, :8]
        values_without_dc = low_frequency.reshape(-1)[1:]
        median = float(np.median(values_without_dc))
        hash_bits = (low_frequency > median).reshape(-1)
        return AppearanceSignature(
            image=normalized,
            hash_bits=hash_bits,
            sharpness=sharpness,
        )

    def _create_scene_signature(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> SceneSignature | None:
        frame_height, frame_width = frame.shape[:2]
        scale = min(
            1.0,
            self.config.scene_max_width / frame_width,
        )
        if scale < 1.0:
            scene = cv.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv.INTER_AREA,
            )
        else:
            scene = frame

        gray = cv.cvtColor(scene, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        mask = np.full((height, width), 255, dtype=np.uint8)

        # DJI 画面边缘通常有静态 HUD。忽略它，避免把屏幕坐标
        # 误当成地面坐标；中间区域仍保留足够的环境特征。
        mask[: round(0.10 * height)] = 0
        mask[round(0.88 * height) :] = 0
        mask[:, : round(0.025 * width)] = 0
        mask[:, round(0.95 * width) :] = 0

        x, y, box_width, box_height = detection.bounding_box
        scaled_box = np.array(
            [x, y, box_width, box_height],
            dtype=np.float32,
        ) * scale
        scaled_x, scaled_y, scaled_width, scaled_height = (
            scaled_box
        )
        padding = 0.5 * max(scaled_width, scaled_height)
        cv.rectangle(
            mask,
            (
                max(0, round(scaled_x - padding)),
                max(0, round(scaled_y - padding)),
            ),
            (
                min(
                    width - 1,
                    round(scaled_x + scaled_width + padding),
                ),
                min(
                    height - 1,
                    round(scaled_y + scaled_height + padding),
                ),
            ),
            0,
            -1,
        )

        keypoints, descriptors = self._orb.detectAndCompute(
            gray,
            mask,
        )
        if descriptors is None or len(keypoints) == 0:
            return None

        points = np.float32(
            [keypoint.pt for keypoint in keypoints]
        )
        center = np.float32(
            [
                scaled_x + scaled_width / 2,
                scaled_y + scaled_height / 2,
            ]
        )
        target_size = max(scaled_width, scaled_height)
        geometry = np.float32(
            [
                center,
                center + (target_size, 0),
                center + (0, target_size),
            ]
        )
        return SceneSignature(
            points=points,
            descriptors=descriptors,
            target_geometry=geometry,
        )

    def _match_scenes(
        self,
        reference: SceneSignature,
        candidate: SceneSignature,
    ) -> SceneMatch | None:
        pairs = self._matcher.knnMatch(
            reference.descriptors,
            candidate.descriptors,
            k=2,
        )
        good_matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if (
                first.distance
                < self.config.scene_ratio_test * second.distance
            )
        ]
        if len(good_matches) < self.config.scene_min_matches:
            return None

        source = np.float32(
            [
                reference.points[match.queryIdx]
                for match in good_matches
            ]
        )
        destination = np.float32(
            [
                candidate.points[match.trainIdx]
                for match in good_matches
            ]
        )
        transform, inlier_mask = cv.findHomography(
            source,
            destination,
            cv.RANSAC,
            3.0,
        )
        if transform is None or inlier_mask is None:
            return None

        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / len(good_matches)
        if (
            inliers < self.config.scene_min_inliers
            or inlier_ratio
            < self.config.scene_min_inlier_ratio
        ):
            return None

        projected = cv.perspectiveTransform(
            reference.target_geometry.reshape(1, -1, 2),
            transform,
        ).reshape(-1, 2)
        predicted_center = projected[0]
        candidate_center = candidate.target_geometry[0]
        predicted_size = max(
            np.linalg.norm(projected[1] - projected[0]),
            np.linalg.norm(projected[2] - projected[0]),
        )
        candidate_size = max(
            np.linalg.norm(
                candidate.target_geometry[1]
                - candidate.target_geometry[0]
            ),
            np.linalg.norm(
                candidate.target_geometry[2]
                - candidate.target_geometry[0]
            ),
        )
        if min(predicted_size, candidate_size) <= 0:
            return None

        scale_ratio = (
            max(predicted_size, candidate_size)
            / min(predicted_size, candidate_size)
        )
        if scale_ratio > self.config.scene_max_scale_ratio:
            return None

        position_error = float(
            np.linalg.norm(
                predicted_center - candidate_center
            )
            / max(predicted_size, candidate_size)
        )
        if (
            position_error
            > self.config.scene_max_position_error
        ):
            return None
        return SceneMatch(position_error, inliers)

    def _rectify_body(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> np.ndarray | None:
        if (
            len(detection.polygon) != 5
            or not 0 <= detection.apex_index < 5
        ):
            return None

        points = detection.polygon.reshape(
            5,
            2,
        ).astype(np.float32)
        apex_index = detection.apex_index
        apex = points[apex_index]
        shoulder_1 = points[(apex_index + 1) % 5]
        bottom_1 = points[(apex_index + 2) % 5]
        shoulder_2 = points[(apex_index - 1) % 5]
        bottom_2 = points[(apex_index - 2) % 5]

        base_center = (shoulder_1 + shoulder_2) / 2
        downward = base_center - apex
        if np.linalg.norm(downward) == 0:
            return None

        left_direction = np.array(
            [-downward[1], downward[0]],
            dtype=np.float32,
        )
        if (
            np.dot(
                shoulder_1 - base_center,
                left_direction,
            )
            > np.dot(
                shoulder_2 - base_center,
                left_direction,
            )
        ):
            left_shoulder = shoulder_1
            left_bottom = bottom_1
            right_shoulder = shoulder_2
            right_bottom = bottom_2
        else:
            left_shoulder = shoulder_2
            left_bottom = bottom_2
            right_shoulder = shoulder_1
            right_bottom = bottom_1

        source = np.float32(
            [
                left_shoulder,
                right_shoulder,
                right_bottom,
                left_bottom,
            ]
        )
        size = self.config.rectified_size
        destination = np.float32(
            [
                [0, 0],
                [size - 1, 0],
                [size - 1, size - 1],
                [0, size - 1],
            ]
        )
        transform = cv.getPerspectiveTransform(
            source,
            destination,
        )
        return cv.warpPerspective(
            frame,
            transform,
            (size, size),
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_REPLICATE,
        )

    @staticmethod
    def _correlation(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        first_float = first.astype(np.float32)
        second_float = second.astype(np.float32)
        first_float -= first_float.mean()
        second_float -= second_float.mean()
        denominator = (
            np.linalg.norm(first_float)
            * np.linalg.norm(second_float)
        )
        if denominator == 0:
            return 0.0
        return float(
            np.clip(
                np.sum(first_float * second_float)
                / denominator,
                -1.0,
                1.0,
            )
        )
