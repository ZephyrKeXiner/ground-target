import time
import math
from pymavlink import mavutil

master = mavutil.mavlink_connection(
  "udpin:0.0.0.0:14550",
  source_system=245,
  autoreconnect=True
)

heartbeat = master.wait_heartbeat(
  blocking=True,
  timeout=10,
)

def set_param(name, value, timeout=3, retries=3):
    """
    设置一个ArduPilot参数，并等待PARAM_VALUE确认。

    返回飞控实际保存的参数值。
    """

    expected = float(value)

    for attempt in range(retries):
        print(
            f"设置参数：{name}={expected}，"
            f"第{attempt + 1}次"
        )

        master.param_set_send(name, expected)

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            msg = master.recv_match(
                type="PARAM_VALUE",
                blocking=True,
                timeout=max(0, deadline - time.monotonic())
            )

            if msg is None:
                break

            param_id = msg.param_id

            if isinstance(param_id, bytes):
                param_id = param_id.decode(
                    "ascii",
                    errors="ignore"
                ).rstrip("\x00")

            # 可能收到其他参数
            if param_id != name:
                continue

            actual = float(msg.param_value)

            print(
                f"飞控确认：{param_id}={actual}, "
                f"type={msg.param_type}"
            )

            if math.isclose(
                actual,
                expected,
                rel_tol=1e-6,
                abs_tol=1e-6
            ):
                return actual

            raise RuntimeError(
                f"参数{name}写入不一致："
                f"期望{expected}，实际{actual}"
            )

        print(f"设置{name}等待确认超时")

    raise TimeoutError(f"设置参数失败：{name}")
  
def get_param(name, timeout=3, retries=3):
    """读取一个ArduPilot参数。"""

    for attempt in range(retries):
        master.param_fetch_one(name)

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            msg = master.recv_match(
                type="PARAM_VALUE",
                blocking=True,
                timeout=max(0, deadline - time.monotonic())
            )

            if msg is None:
                break

            param_id = msg.param_id

            if isinstance(param_id, bytes):
                param_id = param_id.decode(
                    "ascii",
                    errors="ignore"
                ).rstrip("\x00")

            if param_id != name:
                # 可能收到其他参数，继续等待
                continue

            print(
                f"读取参数：{param_id} = {msg.param_value}, "
                f"type={msg.param_type}"
            )

            return msg.param_value

        print(f"读取{name}第{attempt + 1}次超时")

    raise TimeoutError(f"无法读取参数：{name}")
  
safety_default = int(get_param("BRD_SAFETY_DEFLT"))
safety_mask = int(get_param("BRD_SAFETY_MASK"))

print("BRD_SAFETY_DEFLT =", safety_default)
print("BRD_SAFETY_MASK =", safety_mask)

print(
    "A1是否忽略安全开关:",
    bool(safety_mask & (1 << 8))
)

mask = int(get_param("BRD_SAFETY_MASK"))

output9_bit = 1 << 8   # 256
new_mask = mask | output9_bit

print("旧MASK:", mask)
print("新MASK:", new_mask)

set_param("BRD_SAFETY_MASK", new_mask)