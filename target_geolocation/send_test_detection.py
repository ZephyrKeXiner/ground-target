from __future__ import annotations

import argparse
import json
import socket
import time


def parse_destination(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError("destination must be HOST:PORT")
    return host, int(port)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one synthetic bbox JSON frame")
    parser.add_argument("--destination", type=parse_destination, default=("127.0.0.1", 15100))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--bbox", nargs=4, type=float)
    args = parser.parse_args()

    if args.bbox is None:
        center_x, center_y = args.width / 2.0, args.height / 2.0
        bbox = [center_x - 10, center_y - 10, center_x + 10, center_y + 10]
    else:
        bbox = args.bbox

    message = {
        "version": 1,
        "camera_id": "down_cam",
        "frame_id": 1,
        "capture_monotonic_ns": time.monotonic_ns(),
        "image": {"width": args.width, "height": args.height},
        "detections": [
            {
                "detection_id": 0,
                "track_id": 1,
                "class_name": "test_target",
                "confidence": 1.0,
                "bbox_xyxy": bbox,
            }
        ],
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(
        json.dumps(message, separators=(",", ":")).encode("utf-8"),
        args.destination,
    )
    print(json.dumps(message, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

