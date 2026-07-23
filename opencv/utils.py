from dataclasses import dataclass

from config import *

import numpy as np
import cv2 as cv

@dataclass
class AdaptiveSVThreshold:
    saturation: float
    value: float

    def update(
        self,
        hsv: np.ndarray,
        hue_mask: np.ndarray,
        config: DetectorConfig,
    ) -> None:
        """用当前帧的颜色候选实时更新并平滑 S/V 阈值。"""
        saturation_channel = hsv[:, :, 1]
        value_channel = hsv[:, :, 2]
        hue_candidates = hue_mask > 0

        raw_saturation = otsu_threshold(
            saturation_channel[hue_candidates],
            config.min_histogram_samples,
        )
        if raw_saturation is not None:
            target = float(np.clip(raw_saturation, *config.dynamic_s_range))
            alpha = config.threshold_smoothing
            self.saturation = (1.0 - alpha) * self.saturation + alpha * target

        # 亮度只统计已经具有一定颜色强度的像素，避免灰色赛道占主导。
        saturated_candidates = hue_candidates & (
            saturation_channel >= round(self.saturation)
        )
        value_samples = value_channel[saturated_candidates]
        raw_value = otsu_threshold(value_samples, config.min_histogram_samples)
        if raw_value is not None:
            lower_quartile = float(np.percentile(value_samples, 25))
            target = float(
                np.clip(
                    min(raw_value, lower_quartile),
                    *config.dynamic_v_range,
                )
            )
            alpha = config.threshold_smoothing
            self.value = (1.0 - alpha) * self.value + alpha * target


@dataclass(frozen=True)
class FrameMasks:
    brightness: np.ndarray
    red: np.ndarray
    blue: np.ndarray
    white: np.ndarray

    @property
    def color(self) -> np.ndarray:
        return cv.bitwise_or(self.red, self.blue)


@dataclass(frozen=True)
class Detection:
    polygon: np.ndarray
    target_color: str
    mode: str
    apex_index: int = -1

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        return cv.boundingRect(self.polygon)

    @property
    def area(self) -> float:
        return float(cv.contourArea(self.polygon))


@dataclass
class FrameAnalysis:
    debug_view: np.ndarray
    strict_detections: list[Detection]
    perspective_detections: list[Detection]
    
def otsu_threshold(samples: np.ndarray, min_samples: int) -> float | None:
    """从一维像素样本的直方图计算 Otsu 分界。"""
    if samples.size < min_samples:
        return None

    values = np.ascontiguousarray(samples.reshape(-1, 1), dtype=np.uint8)
    threshold, _ = cv.threshold(
        values,
        0,
        255,
        cv.THRESH_BINARY | cv.THRESH_OTSU,
    )
    return float(threshold)


def largest_component_area(binary: np.ndarray) -> int:
    component_count, _, stats, _ = cv.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return 0
    return int(stats[1:, cv.CC_STAT_AREA].max())


def crop_and_enhance_target(
    frame: np.ndarray,
    polygon: np.ndarray,
    config: CropConfig,
) -> np.ndarray:
    """裁出目标周围小图，并进行保边降噪、放大、对比度和锐度增强。"""
    x, y, width, height = cv.boundingRect(polygon)
    padding = max(
        config.min_padding,
        round(max(width, height) * config.padding_ratio),
    )
    frame_height, frame_width = frame.shape[:2]
    crop = frame[
        max(0, y - padding) : min(frame_height, y + height + padding),
        max(0, x - padding) : min(frame_width, x + width + padding),
    ].copy()
    if crop.size == 0:
        return crop

    denoised = cv.bilateralFilter(crop, d=5, sigmaColor=30, sigmaSpace=30)
    enlarged = cv.resize(
        denoised,
        None,
        fx=config.scale,
        fy=config.scale,
        interpolation=cv.INTER_LANCZOS4,
    )

    lab = cv.cvtColor(enlarged, cv.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv.split(lab)
    clahe = cv.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(4, 4),
    )
    contrasted = cv.cvtColor(
        cv.merge((clahe.apply(lightness), channel_a, channel_b)),
        cv.COLOR_LAB2BGR,
    )
    blurred = cv.GaussianBlur(contrasted, (0, 0), 1.0)
    return cv.addWeighted(
        contrasted,
        1.0 + config.sharpen_amount,
        blurred,
        -config.sharpen_amount,
        0,
    )
