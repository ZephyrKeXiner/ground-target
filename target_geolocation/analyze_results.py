from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Optional

from target_geolocation.core import EARTH_RADIUS_M


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def local_offset_m(
    latitude_deg: float,
    longitude_deg: float,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> tuple[float, float]:
    north = math.radians(latitude_deg - reference_latitude_deg) * EARTH_RADIUS_M
    east = (
        math.radians(longitude_deg - reference_longitude_deg)
        * EARTH_RADIUS_M
        * math.cos(math.radians(reference_latitude_deg))
    )
    return north, east


def nested(record: dict[str, Any], *keys: str) -> Optional[float]:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def pearson(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) < 5 or len(left) != len(right):
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return sum(
        a * b for a, b in zip(centered_left, centered_right)
    ) / denominator


def load_results(paths: Iterable[str]) -> tuple[list[dict[str, Any]], Counter[str]]:
    valid: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if record.get("type") != "target_geolocation":
                    continue
                if record.get("valid"):
                    valid.append(record)
                else:
                    invalid_reasons[str(record.get("reason", "unknown"))] += 1
    return valid, invalid_reasons


def print_sensitivity_summary(records: list[dict[str, Any]]) -> None:
    labels = {
        "roll_1deg": "横滚或安装横滚误差1°",
        "pitch_1deg": "俯仰或安装俯仰误差1°",
        "yaw_1deg": "航向误差1°",
        "anchor_u_1px": "检测点横向误差1像素",
        "anchor_v_1px": "检测点纵向误差1像素",
        "agl_1m": "高度误差1米",
        "timestamp_10ms": "时间误差10毫秒",
    }
    print("\n平均灵敏度（数值越大，该误差源越危险）：")
    for key, label in labels.items():
        values = [
            nested(record, "debug", "sensitivity", key, "magnitude_m")
            for record in records
        ]
        present = [value for value in values if value is not None]
        if present:
            print(f"  {label}: {statistics.fmean(present):.3f} m")


def analyze(paths: list[str], truth: Optional[tuple[float, float]]) -> None:
    records, invalid_reasons = load_results(paths)
    print(f"有效定位: {len(records)}")
    print(f"无效定位: {sum(invalid_reasons.values())}")
    for reason, count in invalid_reasons.most_common(5):
        print(f"  失败 {count} 次: {reason}")
    if not records:
        return

    latitudes = [float(record["target"]["latitude_deg"]) for record in records]
    longitudes = [float(record["target"]["longitude_deg"]) for record in records]
    if truth is None:
        reference_lat = statistics.median(latitudes)
        reference_lon = statistics.median(longitudes)
        print("\n未提供目标真值，只能分析重复精度，不能判断绝对偏差原因。")
    else:
        reference_lat, reference_lon = truth

    errors = [
        local_offset_m(lat, lon, reference_lat, reference_lon)
        for lat, lon in zip(latitudes, longitudes)
    ]
    north_errors = [north for north, _ in errors]
    east_errors = [east for _, east in errors]
    radial_errors = [math.hypot(north, east) for north, east in errors]
    mean_north = statistics.fmean(north_errors)
    mean_east = statistics.fmean(east_errors)
    bias = math.hypot(mean_north, mean_east)
    scatter = math.sqrt(
        statistics.fmean(
            (north - mean_north) ** 2 + (east - mean_east) ** 2
            for north, east in errors
        )
    )

    if truth is None:
        print(f"相对中位点RMS散布: {scatter:.2f} m")
        print(f"相对中位点P95: {percentile(radial_errors, 0.95):.2f} m")
    else:
        print(f"\n平均偏差: 北 {mean_north:+.2f} m，东 {mean_east:+.2f} m")
        print(f"系统偏差大小: {bias:.2f} m")
        print(f"去除平均偏差后的RMS散布: {scatter:.2f} m")
        print(
            "绝对误差: "
            f"中位数 {statistics.median(radial_errors):.2f} m，"
            f"P95 {percentile(radial_errors, 0.95):.2f} m，"
            f"RMSE {math.sqrt(statistics.fmean(value * value for value in radial_errors)):.2f} m"
        )

    warning_counts: Counter[str] = Counter()
    for record in records:
        warning_counts.update(record.get("quality", {}).get("warnings", []))
    if warning_counts:
        print("\n高频质量告警：")
        for warning, count in warning_counts.most_common(5):
            print(f"  {count}/{len(records)}: {warning}")

    print_sensitivity_summary(records)

    if truth is None:
        return

    features = {
        "高度": ("quality", "vehicle_agl_m"),
        "横滚角": ("debug", "vehicle", "roll_deg"),
        "俯仰角": ("debug", "vehicle", "pitch_deg"),
        "姿态样本时间差": ("quality", "pose_age_ms"),
        "射线向下分量": ("quality", "ray_down_component"),
        "bbox横坐标": ("anchor_uv", "0"),
    }
    correlations: list[tuple[float, str, str]] = []
    for label, keys in features.items():
        feature_values: list[float] = []
        selected_north: list[float] = []
        selected_east: list[float] = []
        selected_radial: list[float] = []
        for record, north, east, radial in zip(
            records, north_errors, east_errors, radial_errors
        ):
            if keys == ("anchor_uv", "0"):
                anchor = record.get("anchor_uv")
                value = float(anchor[0]) if anchor and len(anchor) == 2 else None
            else:
                value = nested(record, *keys)
            if value is None:
                continue
            feature_values.append(value)
            selected_north.append(north)
            selected_east.append(east)
            selected_radial.append(radial)
        for axis, values in (
            ("北向误差", selected_north),
            ("东向误差", selected_east),
            ("误差大小", selected_radial),
        ):
            coefficient = pearson(feature_values, values)
            if coefficient is not None:
                correlations.append((abs(coefficient), label, f"{axis} r={coefficient:+.2f}"))

    strong = sorted(correlations, reverse=True)[:5]
    if strong:
        print("\n最强相关性（相关不等于因果）：")
        for absolute, label, detail in strong:
            if absolute >= 0.30:
                print(f"  {label} ↔ {detail}")

    along_errors: list[float] = []
    cross_errors: list[float] = []
    delay_estimates_s: list[float] = []
    for record, (north, east) in zip(records, errors):
        velocity = record.get("debug", {}).get("vehicle", {}).get(
            "velocity_ned_mps"
        )
        if not velocity or len(velocity) < 2:
            continue
        vn, ve = float(velocity[0]), float(velocity[1])
        speed = math.hypot(vn, ve)
        if speed < 0.5:
            continue
        along = (north * vn + east * ve) / speed
        cross = (-north * ve + east * vn) / speed
        along_errors.append(along)
        cross_errors.append(cross)
        delay_estimates_s.append(along / speed)

    print("\n可能原因：")
    clues: list[str] = []
    if bias > max(2.0, scatter * 1.5):
        clues.append("偏差明显大于散布，优先检查相机安装角、内参、目标真值和固定方向偏置")
    if along_errors and abs(statistics.median(along_errors)) > max(
        1.0, statistics.median(abs(value) for value in cross_errors)
    ):
        delay_ms = statistics.median(delay_estimates_s) * 1000.0
        clues.append(
            f"误差主要沿飞行方向，疑似时间同步/图像管线延迟；等效延迟约{delay_ms:+.1f}ms"
        )
    altitude_correlation = next(
        (
            detail
            for absolute, label, detail in correlations
            if label == "高度" and "误差大小" in detail and absolute >= 0.5
        ),
        None,
    )
    if altitude_correlation:
        clues.append("误差随高度明显变化，优先检查焦距、安装角和AGL高度")
    attitude_related = any(
        absolute >= 0.5 and label in ("横滚角", "俯仰角")
        for absolute, label, _ in correlations
    )
    if attitude_related:
        clues.append("误差与飞机姿态相关，优先检查相机外参、姿态时间对齐和滚动快门")
    if not clues:
        clues.append("样本还不足以锁定单一原因；增加不同高度、航向和速度的已知目标飞越数据")
    for clue in clues:
        print(f"  - {clue}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze geolocation accuracy and print likely error sources"
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument(
        "--truth",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        help="surveyed target latitude and longitude",
    )
    args = parser.parse_args()
    try:
        analyze(args.results, tuple(args.truth) if args.truth else None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
