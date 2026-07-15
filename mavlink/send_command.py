from pymavlink import mavutil
import time

connection = mavutil.mavlink_connection(
  "udpin:0.0.0.0:14550",
  source_system=245,
  autoreconnect=True
)

heartbeat = connection.wait_heartbeat(
  blocking=True,
  timeout=10,
)

RESULT_NAMES = {
    0: "ACCEPTED",
    1: "TEMPORARILY_REJECTED",
    2: "DENIED",
    3: "UNSUPPORTED",
    4: "FAILED",
    5: "IN_PROGRESS",
    6: "CANCELLED",
    7: "COMMAND_LONG_ONLY",
    8: "COMMAND_INT_ONLY",
}

def send_command(command, param=None, timeout=3, retries=3):
  params = list(param or [])
  
  if len(params) > 7:
    raise ValueError("COMMAND_LONG最多只有7个参数")
  
  params += [0.0]*(7-len(params))
  
  for confirmation in range(retries):
    print(
      f"发送command: {command}",
      f"confirmation={confirmation}, params={params}"
    )

    connection.mav.command_long_send(
      connection.target_system,
      connection.target_component,
      command,
      confirmation,
      *params
    )
  
    deadline = time.monotonic() + timeout
    in_progress = False
    
    while time.monotonic()<deadline:
      remaining = deadline - time.monotonic()
      
      ack = connection.recv_match(
        type="COMMAND_ACK",
        blocking=True,
        timeout=max(0, remaining)
      )
      
      if ack is None:
        break
      
      if ack.command != command:
        continue
      
      result_name = RESULT_NAMES.get(
        ack.result,
        f"UNKNOWN_{ack.result}"
      )
      
      print(
        f"收到ACK：command={ack.command}, "
        f"result={ack.result} ({result_name}), "
        f"progress={getattr(ack, 'progress', None)}"
      )
      
      if ack.result == mavutil.mavlink.MAV_RESULT_IN_PROGRESS:
        in_progress = True
        deadline = time.monotonic() + 30
        continue
              
      return ack

    if in_progress:
        raise TimeoutError(f"命令{command}执行中，但未收到最终ACK")

    print(f"第{confirmation + 1}次等待ACK超时")

  raise TimeoutError(f"命令{command}发送失败：未收到ACK")

ack = send_command(
    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
    [
        9,
        1500
    ]
)