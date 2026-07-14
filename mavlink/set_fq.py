from pymavlink import mavutil

fc = mavutil.mavlink_connection(
    "udpin:0.0.0.0:14550",
    source_system=245,
    autoreconnect=True,
)

print("等待 ArduPilot 心跳……")

heartbeat = fc.wait_heartbeat(timeout=15)
if heartbeat is None:
    raise TimeoutError("没有收到 ArduPilot HEARTBEAT")

print(
    f"连接成功：system={fc.target_system}, "
    f"component={fc.target_component}"
)

import time

def set_message_rate(connection, message_id, frequency_hz):
    interval_us = int(1_000_000 / frequency_hz)

    command = mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL

    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        command,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )

    deadline = time.monotonic() + 3

    while time.monotonic() < deadline:
        ack = connection.recv_match(
            type="COMMAND_ACK",
            blocking=True,
            timeout=0.5,
        )

        if ack is None:
            continue

        if ack.command == command:
            print(f"SET_MESSAGE_INTERVAL ACK result={ack.result}")
            return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED

    print("没有收到 SET_MESSAGE_INTERVAL ACK")
    return False
  
success = set_message_rate(
    fc,
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    20,
)

print("设置成功：", success)