from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from target_geolocation.controller import TelemetryBuffer, process_frame
from target_geolocation.core import CameraCalibration, GeolocationError


class ReplayMessage:
    def __init__(self, message_type: str, fields: dict[str, Any]) -> None:
        self._message_type = message_type
        for name, value in fields.items():
            setattr(self, name, value)

    def get_type(self) -> str:
        return self._message_type


def replay_events(
    *,
    events_path: Path,
    calibration: CameraCalibration,
    geolocation_config: dict[str, Any],
):
    telemetry = TelemetryBuffer()
    with events_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {events_path}:{line_number}: {exc}"
                ) from exc

            event_type = event.get("event")
            if event_type == "telemetry":
                receive_s = float(event["receive_monotonic_ns"]) / 1e9
                telemetry.ingest(
                    ReplayMessage(event["message_type"], event["message"]),
                    receive_s,
                )
                continue

            if event_type != "bbox_frame":
                continue

            frame = event["frame"]
            receive_s = float(event["receive_monotonic_ns"]) / 1e9
            try:
                results = process_frame(
                    frame=frame,
                    receive_monotonic_s=receive_s,
                    telemetry=telemetry,
                    calibration=calibration,
                    geolocation_config=geolocation_config,
                )
            except (GeolocationError, ValueError, TypeError) as exc:
                results = [
                    {
                        "type": "target_geolocation",
                        "frame_id": frame.get("frame_id"),
                        "valid": False,
                        "reason": str(exc),
                    }
                ]

            for result in results:
                result["replayed"] = True
                yield result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay recorded telemetry+bbox events with a new calibration/config"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument(
        "--config", default="target_geolocation/config.json"
    )
    parser.add_argument("--output", help="optional replay result NDJSON")
    args = parser.parse_args()

    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        calibration = CameraCalibration.from_mapping(config["camera"])
        geolocation_config = config.get("geolocation", {})
    except (OSError, KeyError, json.JSONDecodeError, GeolocationError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    output_file = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")

    count = 0
    try:
        for result in replay_events(
            events_path=Path(args.events),
            calibration=calibration,
            geolocation_config=geolocation_config,
        ):
            line = json.dumps(
                result, ensure_ascii=False, separators=(",", ":")
            )
            print(line)
            if output_file is not None:
                output_file.write(line + "\n")
            count += 1
    except (OSError, ValueError) as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 3
    finally:
        if output_file is not None:
            output_file.close()

    print(f"replayed results: {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
