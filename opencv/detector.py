import cv2 as cv
import numpy as np

from config import DetectorConfig
from utils import (
    AdaptiveSVThreshold,
    Detection,
    FrameAnalysis,
    FrameMasks,
    largest_component_area,
)


class TargetDetector:
    """检测内部为红色或蓝色的标准房形五边形。"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.threshold = AdaptiveSVThreshold(
            saturation=self.config.initial_saturation,
            value=self.config.initial_value,
        )

    def analyze(self, frame: np.ndarray) -> FrameAnalysis:
        masks = self._create_masks(frame)
        detections = self._find_detections(masks)
        debug_view = self._draw_detection_view(
            masks.color,
            detections,
        )
        return FrameAnalysis(debug_view, detections)

    def _create_masks(self, frame: np.ndarray) -> FrameMasks:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]

        red_hue = np.zeros(hue.shape, dtype=np.uint8)
        for lower, upper in self.config.red_hue_ranges:
            red_hue = cv.bitwise_or(
                red_hue,
                cv.inRange(hue, lower, upper),
            )

        blue_lower, blue_upper = self.config.blue_hue_range
        blue_hue = cv.inRange(hue, blue_lower, blue_upper)
        hue_candidates = cv.bitwise_or(red_hue, blue_hue)
        self.threshold.update(
            hsv,
            hue_candidates,
            self.config,
        )

        valid_color = cv.inRange(
            hsv,
            np.array(
                [
                    0,
                    round(self.threshold.saturation),
                    round(self.threshold.value),
                ],
                dtype=np.uint8,
            ),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        red = cv.bitwise_and(red_hue, valid_color)
        blue = cv.bitwise_and(blue_hue, valid_color)
        blue = cv.bitwise_and(
            blue,
            cv.inRange(
                hsv[:, :, 1],
                self.config.min_blue_saturation,
                255,
            ),
        )

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        _, brightness = cv.threshold(
            gray,
            0,
            255,
            cv.THRESH_BINARY | cv.THRESH_OTSU,
        )
        return FrameMasks(brightness, red, blue)

    def _find_detections(
        self,
        masks: FrameMasks,
    ) -> list[Detection]:
        contours, _ = cv.findContours(
            masks.brightness,
            cv.RETR_LIST,
            cv.CHAIN_APPROX_SIMPLE,
        )
        detections: list[Detection] = []

        for contour in contours:
            area = cv.contourArea(contour)
            if not (
                self.config.min_shape_area
                <= area
                <= self.config.max_shape_area
            ):
                continue

            detection = self._classify_contour(
                contour,
                masks,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _classify_contour(
        self,
        contour: np.ndarray,
        masks: FrameMasks,
    ) -> Detection | None:
        polygon = self._fit_five_vertices(contour)
        if polygon is None:
            return None

        matches, apex_index = self._is_house_pentagon(polygon)
        if not matches:
            return None

        target_color = self._dominant_target_color(
            polygon,
            masks,
        )
        if target_color is None:
            return None

        return Detection(
            polygon=polygon,
            target_color=target_color,
            apex_index=apex_index,
        )

    def _dominant_target_color(
        self,
        polygon: np.ndarray,
        masks: FrameMasks,
    ) -> str | None:
        polygon_mask = np.zeros_like(masks.brightness)
        cv.fillPoly(polygon_mask, [polygon], 255)
        polygon_area = cv.countNonZero(polygon_mask)
        if polygon_area == 0:
            return None

        red_region = cv.bitwise_and(masks.red, polygon_mask)
        blue_region = cv.bitwise_and(masks.blue, polygon_mask)
        red_pixels = cv.countNonZero(red_region)
        blue_pixels = cv.countNonZero(blue_region)

        if red_pixels >= blue_pixels:
            target_color = "red"
            dominant_pixels = red_pixels
            dominant_region = red_region
        else:
            target_color = "blue"
            dominant_pixels = blue_pixels
            dominant_region = blue_region

        largest_color = largest_component_area(dominant_region)
        if (
            dominant_pixels < self.config.min_color_pixels
            or dominant_pixels / polygon_area
            < self.config.min_color_ratio
            or largest_color < self.config.min_component_pixels
            or largest_color / polygon_area
            < self.config.min_component_ratio
        ):
            return None

        return target_color

    @staticmethod
    def _fit_five_vertices(
        contour: np.ndarray,
    ) -> np.ndarray | None:
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            return None

        for epsilon_ratio in (
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
            0.04,
        ):
            polygon = cv.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True,
            )
            if len(polygon) == 5:
                return polygon
        return None

    @classmethod
    def _is_house_pentagon(
        cls,
        polygon: np.ndarray,
    ) -> tuple[bool, int]:
        if len(polygon) != 5 or not cv.isContourConvex(polygon):
            return False, -1

        points = polygon.reshape(-1, 2).astype(np.float64)
        sides = [
            np.linalg.norm(
                points[(index + 1) % 5] - points[index]
            )
            for index in range(5)
        ]
        shortest_side = min(sides)
        if (
            shortest_side == 0
            or max(sides) / shortest_side > 1.8
        ):
            return False, -1

        angles = cls._polygon_angles(polygon)
        if len(angles) != 5:
            return False, -1

        apex_index = int(np.argmin(angles))
        apex_angle = angles[apex_index]
        shoulder_angles = [
            angles[(apex_index - 1) % 5],
            angles[(apex_index + 1) % 5],
        ]
        square_angles = [
            angles[(apex_index - 2) % 5],
            angles[(apex_index + 2) % 5],
        ]
        matches = (
            45.0 <= apex_angle <= 95.0
            and all(
                125.0 <= angle <= 170.0
                for angle in shoulder_angles
            )
            and all(
                65.0 <= angle <= 120.0
                for angle in square_angles
            )
        )
        return matches, apex_index

    @staticmethod
    def _polygon_angles(
        polygon: np.ndarray,
    ) -> list[float]:
        points = polygon.reshape(-1, 2).astype(np.float64)
        angles: list[float] = []

        for index, current in enumerate(points):
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            vector_1 = previous - current
            vector_2 = following - current
            denominator = (
                np.linalg.norm(vector_1)
                * np.linalg.norm(vector_2)
            )
            if denominator == 0:
                return []

            cosine = np.dot(vector_1, vector_2) / denominator
            angle = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )
            angles.append(float(angle))

        return angles

    def _draw_detection_view(
        self,
        color_mask: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        result = cv.cvtColor(color_mask, cv.COLOR_GRAY2BGR)

        for detection in detections:
            contour_color = (
                (0, 0, 255)
                if detection.target_color == "red"
                else (255, 0, 0)
            )
            cv.drawContours(
                result,
                [detection.polygon],
                -1,
                contour_color,
                3,
            )
            x, y, _, _ = detection.bounding_box
            cv.putText(
                result,
                f"{detection.target_color} pentagon",
                (x, max(20, y - 8)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                contour_color,
                2,
                cv.LINE_AA,
            )

        cv.putText(
            result,
            f"pentagons: {len(detections)}",
            (10, 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            result,
            (
                f"dynamic S>={self.threshold.saturation:.1f} "
                f"V>={self.threshold.value:.1f}"
            ),
            (10, 92),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv.LINE_AA,
        )
        return result
