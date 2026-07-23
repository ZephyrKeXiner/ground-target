from dataclasses import dataclass

@dataclass(frozen=True)
class DetectorConfig:
    """目标分割与几何筛选参数。"""

    min_shape_area: float = 300.0
    max_shape_area: float = 50_000.0

    red_hue_ranges: tuple[tuple[int, int], ...] = ((0, 15), (155, 179))
    blue_hue_range: tuple[int, int] = (90, 140)
    min_blue_saturation: int = 120

    min_color_pixels: int = 60
    min_color_ratio: float = 0.18
    min_component_pixels: int = 60
    min_component_ratio: float = 0.10

    perspective_min_color_ratio: float = 0.25
    perspective_min_component_ratio: float = 0.15
    perspective_min_solidity: float = 0.65
    perspective_max_aspect_ratio: float = 3.5
    perspective_min_white_ratio: float = 0.05

    white_max_saturation: int = 65
    white_min_value: int = 120

    initial_saturation: float = 80.0
    initial_value: float = 35.0
    dynamic_s_range: tuple[float, float] = (80.0, 100.0)
    dynamic_v_range: tuple[float, float] = (20.0, 35.0)
    threshold_smoothing: float = 0.20
    min_histogram_samples: int = 200


@dataclass(frozen=True)
class TrackerConfig:
    """跨帧匹配与确认参数。"""

    confirmation_frames: int = 3
    max_missed_frames: int = 2
    match_distance_ratio: float = 1.5
    max_size_ratio: float = 3.0


@dataclass(frozen=True)
class CropConfig:
    """确认后目标小图的裁剪与增强参数。"""

    padding_ratio: float = 0.5
    min_padding: int = 15
    scale: float = 2.0
    clahe_clip_limit: float = 1.0
    sharpen_amount: float = 0.1