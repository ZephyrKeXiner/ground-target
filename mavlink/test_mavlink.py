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

while True:
    msg = fc.recv_match(
      type='ATTITUDE',
      blocking=True,
      timeout=2,
    )

    if msg is not None and msg.get_type() != "BAD_DATA":
        msg_dict = msg.to_dict()
        print("Roll: ", msg_dict['roll'])
        print("Pitch: ", msg_dict['pitch'])
        print("yaw: ", msg_dict['yaw'])
        print()