from config import *
from utils import *

import cv2 as cv
import numpy as np

class TargetDetector:
    """完成单帧分割、轮廓提取和目标几何筛选。"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.threshold = AdaptiveSVThreshold(
            saturation=self.config.initial_saturation,
            value=self.config.initial_value,
        )

    def analyze(self, frame: np.ndarray) -> FrameAnalysis:
        masks = self._create_masks(frame)
        strict, perspective = self._find_detections(masks)
        debug_view = self._draw_detection_view(masks.color, strict, perspective)
        return FrameAnalysis(debug_view, strict, perspective)

    def _create_masks(self, frame: np.ndarray) -> FrameMasks:
        # BGR -> HSV 每帧只做一次，颜色和白色结构共用结果。
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]

        red_hue = np.zeros(hue.shape, dtype=np.uint8)
        for lower, upper in self.config.red_hue_ranges:
            red_hue = cv.bitwise_or(red_hue, cv.inRange(hue, lower, upper))

        blue_lower, blue_upper = self.config.blue_hue_range
        blue_hue = cv.inRange(hue, blue_lower, blue_upper)
        hue_candidates = cv.bitwise_or(red_hue, blue_hue)
        self.threshold.update(hsv, hue_candidates, self.config)

        valid_color = cv.inRange(
            hsv,
            np.array(
                [0, round(self.threshold.saturation), round(self.threshold.value)],
                dtype=np.uint8,
            ),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        red = cv.bitwise_and(red_hue, valid_color)
        blue = cv.bitwise_and(blue_hue, valid_color)

        blue_saturation = cv.inRange(
            hsv[:, :, 1],
            self.config.min_blue_saturation,
            255,
        )
        blue = cv.bitwise_and(blue, blue_saturation)

        white = cv.inRange(
            hsv,
            np.array([0, 0, self.config.white_min_value], dtype=np.uint8),
            np.array(
                [179, self.config.white_max_saturation, 255],
                dtype=np.uint8,
            ),
        )

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        _, brightness = cv.threshold(
            gray,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
        return FrameMasks(brightness, red, blue, white)

    def _find_detections(
        self,
        masks: FrameMasks,
    ) -> tuple[list[Detection], list[Detection]]:
        contours, hierarchy = cv.findContours(
            masks.brightness,
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE,
        )
        if hierarchy is None:
            return [], []

        strict_detections: list[Detection] = []
        perspective_detections: list[Detection] = []

        for index, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if (
                self._contour_depth(index, hierarchy) % 2 == 0
                or not self.config.min_shape_area
                <= area
                <= self.config.max_shape_area
            ):
                continue

            detection = self._classify_contour(contour, masks)
            if detection is None:
                continue
            if detection.mode == "strict":
                strict_detections.append(detection)
            else:
                perspective_detections.append(detection)

        return strict_detections, perspective_detections

    def _classify_contour(
        self,
        contour: np.ndarray,
        masks: FrameMasks,
    ) -> Detection | None:
        polygon = self._fit_five_vertices(contour)
        strict_match = False
        apex_index = -1

        if polygon is not None:
            strict_match, apex_index = self._is_house_pentagon(polygon)

        if not strict_match:
            polygon = self._fit_perspective_polygon(contour)
            if polygon is None or not self._passes_perspective_geometry(
                contour,
                polygon,
            ):
                return None

        color = self._dominant_target_color(
            polygon,
            masks,
            strict_match,
        )
        if color is None:
            return None

        return Detection(
            polygon=polygon,
            target_color=color,
            mode="strict" if strict_match else "perspective",
            apex_index=apex_index if strict_match else -1,
        )

    def _dominant_target_color(
        self,
        polygon: np.ndarray,
        masks: FrameMasks,
        strict: bool,
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

        color_ratio = (
            self.config.min_color_ratio
            if strict
            else self.config.perspective_min_color_ratio
        )
        component_ratio = (
            self.config.min_component_ratio
            if strict
            else self.config.perspective_min_component_ratio
        )
        largest_color = largest_component_area(dominant_region)

        if (
            dominant_pixels < self.config.min_color_pixels
            or dominant_pixels / polygon_area < color_ratio
            or largest_color < self.config.min_component_pixels
            or largest_color / polygon_area < component_ratio
        ):
            return None

        if not strict:
            white_region = cv.bitwise_and(masks.white, polygon_mask)
            if (
                largest_component_area(white_region) / polygon_area
                < self.config.perspective_min_white_ratio
            ):
                return None

        return target_color

    def _passes_perspective_geometry(
        self,
        contour: np.ndarray,
        polygon: np.ndarray,
    ) -> bool:
        contour_area = cv.contourArea(contour)
        hull_area = cv.contourArea(cv.convexHull(contour))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0

        _, (width, height), _ = cv.minAreaRect(polygon)
        shortest_side = min(width, height)
        aspect_ratio = (
            max(width, height) / shortest_side
            if shortest_side > 0
            else float("inf")
        )
        return (
            solidity >= self.config.perspective_min_solidity
            and aspect_ratio <= self.config.perspective_max_aspect_ratio
        )

    @staticmethod
    def _contour_depth(index: int, hierarchy: np.ndarray) -> int:
        depth = 0
        parent = hierarchy[0][index][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[0][parent][3]
        return depth

    @staticmethod
    def _fit_five_vertices(contour: np.ndarray) -> np.ndarray | None:
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            return None

        for epsilon_ratio in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
            polygon = cv.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(polygon) == 5:
                return polygon
        return None
    
    # @staticmethod
    # def _fit_three_vertices(contour: np.ndarray) -> int:
    #     perimeter = cv.arcLength(contour, True)
    #     if perimeter == 0:
    #         return None
        
    #     for

    @staticmethod
    def _fit_perspective_polygon(contour: np.ndarray) -> np.ndarray | None:
        hull = cv.convexHull(contour)
        perimeter = cv.arcLength(hull, True)
        if perimeter == 0:
            return None

        best: np.ndarray | None = None
        for epsilon_ratio in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
            polygon = cv.approxPolyDP(hull, epsilon_ratio * perimeter, True)
            if not 4 <= len(polygon) <= 7:
                continue
            if best is None or abs(len(polygon) - 5) < abs(len(best) - 5):
                best = polygon
            if len(polygon) == 5:
                break
        return best

    @classmethod
    def _is_house_pentagon(cls, polygon: np.ndarray) -> tuple[bool, int]:
        """判断轮廓是否接近“正方形 + 等边三角形”的五边形。"""
        if len(polygon) != 5 or not cv.isContourConvex(polygon):
            return False, -1

        points = polygon.reshape(-1, 2).astype(np.float64)
        sides = [
            np.linalg.norm(points[(index + 1) % 5] - points[index])
            for index in range(5)
        ]
        shortest_side = min(sides)
        if shortest_side == 0 or max(sides) / shortest_side > 1.8:
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
            and all(125.0 <= angle <= 170.0 for angle in shoulder_angles)
            and all(65.0 <= angle <= 120.0 for angle in square_angles)
        )
        return matches, apex_index
    
    @classmethod
    def _is_triangle(cls, polygon:np.ndarray) ->tuple[bool, int]:
        if len(polygon) != 3:
            return False, -1
        
        points = polygon.reshape(-1, 2).astype(np.float64)
        sides = [
            np.linalg.norm(points[(index + 1) % 3] - points[index])
            for index in range(5)
        ]
        
        shortest_side = min(sides)
        if shortest_side == 0 or max(sides) / shortest_side > 1.8:
            return False, -1
        
        
        

    @staticmethod
    def _polygon_angles(polygon: np.ndarray) -> list[float]:
        points = polygon.reshape(-1, 2).astype(np.float64)
        angles: list[float] = []

        for index, current in enumerate(points):
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            vector_1 = previous - current
            vector_2 = following - current
            denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
            if denominator == 0:
                return []

            cosine = np.dot(vector_1, vector_2) / denominator
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            angles.append(float(angle))

        return angles

    def _draw_detection_view(
        self,
        color_mask: np.ndarray,
        strict: list[Detection],
        perspective: list[Detection],
    ) -> np.ndarray:
        result = cv.cvtColor(color_mask, cv.COLOR_GRAY2BGR)

        for detection in [*strict, *perspective]:
            contour_color = (
                (0, 0, 255)
                if detection.target_color == "red"
                else (255, 0, 0)
            )
            cv.drawContours(result, [detection.polygon], -1, contour_color, 3)
            points = detection.polygon.reshape(-1, 2)
            for point in points:
                cv.circle(result, tuple(point.astype(int)), 4, (0, 255, 255), -1)

            if detection.apex_index >= 0:
                apex = tuple(points[detection.apex_index].astype(int))
                cv.circle(result, apex, 6, (0, 0, 255), -1)

            x, y, _, _ = detection.bounding_box
            mode = "strict" if detection.mode == "strict" else "perspective candidate"
            cv.putText(
                result,
                f"{detection.target_color} {mode}",
                (x, max(20, y - 8)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                contour_color,
                2,
                cv.LINE_AA,
            )

        cv.putText(
            result,
            f"strict: {len(strict)}  perspective candidates: {len(perspective)}",
            (10, 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            result,
            f"dynamic S>={self.threshold.saturation:.1f} "
            f"V>={self.threshold.value:.1f}",
            (10, 92),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv.LINE_AA,
        )
        return result