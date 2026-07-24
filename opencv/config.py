from dataclasses import dataclass

@dataclass(frozen=True)
class DetectorConfig:
    """目标分割与几何筛选参数。"""

    min_shape_area: float = 400
    max_shape_area: float = 50_000.0

    red_hue_ranges: tuple[tuple[int, int], ...] = ((0, 15), (155, 179))
    blue_hue_range: tuple[int, int] = (90, 140)
    min_blue_saturation: int = 120

    min_color_pixels: int = 60
    min_color_ratio: float = 0.18
    min_component_pixels: int = 60
    min_component_ratio: float = 0.10

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
    max_missed_frames: int = 15
    match_distance_ratio: float = 2.0
    max_size_ratio: float = 3.0


@dataclass(frozen=True)
class RegistryConfig:
    """目标离开视野后重新出现时的场景与外观匹配参数。"""

    rectified_size: int = 96
    signature_size: int = 48
    number_roi: tuple[float, float, float, float] = (
        0.25,
        0.20,
        0.75,
        0.80,
    )
    max_hash_distance: int = 14
    min_correlation: float = 0.78
    max_signatures_per_target: int = 8

    scene_max_width: int = 960
    scene_orb_features: int = 1800
    scene_fast_threshold: int = 12
    scene_ratio_test: float = 0.72
    scene_min_matches: int = 12
    scene_min_inliers: int = 12
    scene_min_inlier_ratio: float = 0.50
    scene_max_position_error: float = 0.65
    scene_max_scale_ratio: float = 3.0
    max_scene_signatures_per_target: int = 6


@dataclass(frozen=True)
class CropConfig:
    """确认后目标小图的裁剪与增强参数。"""

    padding_ratio: float = 0.5
    min_padding: int = 15
    scale: float = 2.0
    clahe_clip_limit: float = 1.0
    sharpen_amount: float = 0.1
