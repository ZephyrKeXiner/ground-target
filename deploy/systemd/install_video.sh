#!/bin/sh
set -eu

PROJECT_DIR=/home/argus/ai
UNIT_SOURCE="$PROJECT_DIR/deploy/systemd"
ENV_DIR=/etc/ground-target
ENV_FILE="$ENV_DIR/video.env"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this installer with sudo: sudo $0" >&2
    exit 1
fi

install -d -m 0755 "$ENV_DIR"
if [ ! -e "$ENV_FILE" ]; then
    install -m 0644 "$UNIT_SOURCE/video.env.example" "$ENV_FILE"
    echo "Installed default video environment: $ENV_FILE"
else
    echo "Preserved existing video environment: $ENV_FILE"
fi

install -m 0644 "$UNIT_SOURCE/ground-target-video.service" /etc/systemd/system/
systemd-analyze verify /etc/systemd/system/ground-target-video.service
systemctl daemon-reload
systemctl enable --now ground-target-video.service

echo "Full-resolution recording service is enabled and running."
echo "Video directory: /home/argus/ai/training_videos"
