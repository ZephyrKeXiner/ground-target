#!/bin/sh
set -eu

PROJECT_DIR=/home/argus/ai
UNIT_SOURCE="$PROJECT_DIR/deploy/systemd"
ENV_DIR=/etc/ground-target
ENV_FILE="$ENV_DIR/ground-target.env"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this installer with sudo: sudo $0" >&2
    exit 1
fi

install -d -m 0755 "$ENV_DIR"
if [ ! -e "$ENV_FILE" ]; then
    install -m 0644 "$UNIT_SOURCE/ground-target.env.example" "$ENV_FILE"
    echo "Installed default environment: $ENV_FILE"
else
    echo "Preserved existing environment: $ENV_FILE"
fi

install -m 0644 "$UNIT_SOURCE/ground-target.target" /etc/systemd/system/
install -m 0644 "$UNIT_SOURCE/ground-target-prepare.service" /etc/systemd/system/
install -m 0644 "$UNIT_SOURCE/ground-target-controller.service" /etc/systemd/system/
install -m 0644 "$UNIT_SOURCE/ground-target-yolo.service" /etc/systemd/system/

systemd-analyze verify \
    /etc/systemd/system/ground-target.target \
    /etc/systemd/system/ground-target-prepare.service \
    /etc/systemd/system/ground-target-controller.service \
    /etc/systemd/system/ground-target-yolo.service
systemctl daemon-reload
systemctl enable ground-target.target

cd "$PROJECT_DIR"
if runuser -u argus -- "$PROJECT_DIR/.venv/bin/python" \
    -m target_geolocation.service_entrypoint validate; then
    systemctl restart ground-target.target
    echo "ground-target.target is enabled and running"
else
    echo "Units are installed and enabled, but were not started." >&2
    echo "Complete $PROJECT_DIR/target_geolocation/config.json, then run:" >&2
    echo "  sudo systemctl restart ground-target.target" >&2
fi
