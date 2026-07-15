from __future__ import annotations

import os
import socket
import time


class SystemdNotifier:
    """Small sd_notify/watchdog client with no external dependency."""

    def __init__(self) -> None:
        self.address = os.environ.get("NOTIFY_SOCKET")
        watchdog_usec = int(os.environ.get("WATCHDOG_USEC", "0") or 0)
        self.watchdog_interval_s = (
            max(0.5, watchdog_usec / 2_000_000.0) if watchdog_usec else 5.0
        )
        self.last_watchdog_s = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.address)

    def notify(self, message: str) -> None:
        if not self.address:
            return
        address = self.address
        if address.startswith("@"):
            address = "\0" + address[1:]
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.connect(address)
                sock.sendall(message.encode("utf-8"))
        except OSError:
            # Losing a status notification must not crash the flight process.
            pass

    def ready(self, status: str) -> None:
        self.notify(f"READY=1\nSTATUS={status}")
        self.last_watchdog_s = time.monotonic()

    def watchdog(self, status: str | None = None) -> None:
        now = time.monotonic()
        if now - self.last_watchdog_s < self.watchdog_interval_s:
            return
        message = "WATCHDOG=1"
        if status:
            message += f"\nSTATUS={status}"
        self.notify(message)
        self.last_watchdog_s = now

    def stopping(self, status: str) -> None:
        self.notify(f"STOPPING=1\nSTATUS={status}")
