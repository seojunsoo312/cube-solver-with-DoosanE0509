#!/usr/bin/env python3
"""큐브 솔버용 그리퍼 제어 노드.

기본 아이디어는 /home/junsoo/Downloads/gripper 의 gripper_node.py 와 동일하다.
- DrlStart 서비스로 DRL 스크립트를 전송
- DRL 내부에서 flange_serial_open/write/close를 한 번에 수행
- Modbus RTU 패킷으로 RH-P12-RN(A) 레지스터 제어

rules 문서의 파지 펄스 사용을 위해 아래 기본 동작을 제공한다.
- /gripper/release   -> pulse 100
- /gripper/grab_cube -> pulse 420
- /gripper/grab_rotate -> pulse 400
- /gripper/grab_repose -> pulse 410
- /gripper/move_pulse (std_msgs/Int32 구독) -> 임의 pulse 이동
"""
from __future__ import annotations

import json
import struct
import threading
import time

import rclpy
from dsr_msgs2.srv import DrlStart
from dsr_msgs2.srv import GetOutputRegisterInt, SetOutputRegisterInt
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String
from std_srvs.srv import SetBool, Trigger


class ModbusRTU:
    @staticmethod
    def crc16(data: bytes) -> bytes:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return struct.pack("<H", crc)

    @classmethod
    def fc06(cls, slave_id: int, addr: int, value: int) -> bytes:
        body = bytes([slave_id, 0x06]) + struct.pack(">HH", addr, value)
        return body + cls.crc16(body)

    @classmethod
    def fc16(cls, slave_id: int, start: int, values: list[int]) -> bytes:
        n = len(values)
        body = bytes([slave_id, 0x10]) + struct.pack(">HH", start, n) + bytes([n * 2])
        for v in values:
            body += struct.pack(">H", v)
        return body + cls.crc16(body)

    @classmethod
    def fc03(cls, slave_id: int, start: int, count: int) -> bytes:
        body = bytes([slave_id, 0x03]) + struct.pack(">HH", start, count)
        return body + cls.crc16(body)


class Reg:
    TORQUE_ENABLE = 256
    GOAL_CURRENT = 275
    GOAL_POSITION = 282
    PRESENT_POSITION = 284


def build_drl(packets: list[bytes], motion_wait: float = 0.0) -> str:
    lines = [
        "flange_serial_open("
        "baudrate=57600, bytesize=DR_EIGHTBITS, "
        "parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)",
        "wait(0.2)",
    ]
    for i, pkt in enumerate(packets):
        lines.append(f"flange_serial_write(bytes({list(pkt)}))")
        if i < len(packets) - 1:
            lines.append("wait(0.1)")
    if motion_wait > 0:
        lines.append(f"wait({motion_wait})")
    lines.append("flange_serial_close()")
    return "\n".join(lines) + "\n"


class GripperControlNode(Node):
    def __init__(self):
        super().__init__("gripper_control_node")
        cb = ReentrantCallbackGroup()

        self.declare_parameter("robot_ns", "dsr01")
        self.declare_parameter("svc_timeout", 10.0)
        self.declare_parameter("state_hz", 10.0)
        self.declare_parameter("motion_wait", 1.5)
        self.declare_parameter("done_settle_sec", 0.2)
        self.declare_parameter("feedback_poll_hz", 8.0)
        self.declare_parameter("feedback_max_wait_sec", 4.0)
        self.declare_parameter("feedback_pos_tolerance", 30)
        self.declare_parameter("feedback_plc_addr_pos", 120)
        self.declare_parameter("feedback_plc_addr_code", 121)
        self.declare_parameter("feedback_strict", False)
        self.declare_parameter("init_current", 400)
        self.declare_parameter("cube_current", 300)

        self.declare_parameter("pulse_open", 100)
        self.declare_parameter("pulse_cube", 420)
        self.declare_parameter("pulse_rotate", 400)
        self.declare_parameter("pulse_repose", 410)

        self.declare_parameter("slave_id", 1)

        ns = str(self.get_parameter("robot_ns").value)
        self._timeout = float(self.get_parameter("svc_timeout").value)
        self._state_hz = float(self.get_parameter("state_hz").value)
        self._motion_wait = float(self.get_parameter("motion_wait").value)
        self._done_settle_sec = float(self.get_parameter("done_settle_sec").value)
        self._feedback_poll_hz = float(self.get_parameter("feedback_poll_hz").value)
        self._feedback_max_wait_sec = float(self.get_parameter("feedback_max_wait_sec").value)
        self._feedback_pos_tolerance = int(self.get_parameter("feedback_pos_tolerance").value)
        self._feedback_plc_addr_pos = int(self.get_parameter("feedback_plc_addr_pos").value)
        self._feedback_plc_addr_code = int(self.get_parameter("feedback_plc_addr_code").value)
        self._feedback_strict = bool(self.get_parameter("feedback_strict").value)
        self._cur_init = int(self.get_parameter("init_current").value)
        self._cur_cube = int(self.get_parameter("cube_current").value)
        self._pulse_open = int(self.get_parameter("pulse_open").value)
        self._pulse_cube = int(self.get_parameter("pulse_cube").value)
        self._pulse_rotate = int(self.get_parameter("pulse_rotate").value)
        self._pulse_repose = int(self.get_parameter("pulse_repose").value)
        self._slave_id = int(self.get_parameter("slave_id").value)

        self._cli_drl = self.create_client(
            DrlStart, f"/{ns}/drl/drl_start", callback_group=cb
        )
        self._cli_get_out_reg = self.create_client(
            GetOutputRegisterInt, f"/{ns}/plc/get_output_register_int", callback_group=cb
        )
        self._cli_set_out_reg = self.create_client(
            SetOutputRegisterInt, f"/{ns}/plc/set_output_register_int", callback_group=cb
        )

        self._state_pub = self.create_publisher(JointState, "/gripper/state", 10)
        self._event_pub = self.create_publisher(String, "/gripper/event", 20)
        self.create_timer(1.0 / self._state_hz, self._publish_state, callback_group=cb)

        self.create_subscription(
            Int32, "/gripper/move_pulse", self._sub_move_pulse, 10, callback_group=cb
        )
        self.create_subscription(
            String, "/gripper/command", self._sub_command, 20, callback_group=cb
        )

        self.create_service(Trigger, "/gripper/open", self._srv_open, callback_group=cb)
        self.create_service(Trigger, "/gripper/release", self._srv_open, callback_group=cb)
        self.create_service(Trigger, "/gripper/grab_cube", self._srv_grab_cube, callback_group=cb)
        self.create_service(
            Trigger, "/gripper/grab_rotate", self._srv_grab_rotate, callback_group=cb
        )
        self.create_service(
            Trigger, "/gripper/grab_repose", self._srv_grab_repose, callback_group=cb
        )
        self.create_service(Trigger, "/gripper/stop", self._srv_stop, callback_group=cb)
        self.create_service(SetBool, "/gripper/enable", self._srv_enable, callback_group=cb)

        self._stroke = self._pulse_open
        self._torque = False
        self._ready = False
        self._action_lock = threading.Lock()

        self._init_timer = self.create_timer(0.5, self._init_once, callback_group=cb)
        self.get_logger().info("gripper_control_node 시작. 초기화 대기 중...")

    def _publish_event(self, cmd_id: int, action: str, status: str, message: str = "") -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "cmd_id": int(cmd_id),
                "action": str(action),
                "status": str(status),
                "message": str(message),
                "stamp": float(time.time()),
            },
            ensure_ascii=True,
        )
        self._event_pub.publish(msg)

    def _call_service(self, client, request, label: str):
        if not client.service_is_ready():
            self.get_logger().error(f"서비스 미연결: {label}")
            return None

        event = threading.Event()
        result = [None]

        def done_cb(fut):
            result[0] = fut
            event.set()

        fut = client.call_async(request)
        fut.add_done_callback(done_cb)
        if not event.wait(timeout=self._timeout):
            self.get_logger().error(f"타임아웃({self._timeout}s): {label}")
            return None

        try:
            return result[0].result()
        except Exception as e:
            self.get_logger().error(f"서비스 오류 [{label}]: {e}")
            return None

    def _run_packets(self, packets: list[bytes], motion_wait: float, label: str) -> bool:
        req = DrlStart.Request()
        req.robot_system = 0
        req.code = build_drl(packets, motion_wait=motion_wait)
        res = self._call_service(self._cli_drl, req, label)
        return bool(res and res.success)

    def _set_output_register_int(self, address: int, value: int) -> bool:
        req = SetOutputRegisterInt.Request()
        req.address = int(address)
        req.value = int(value)
        res = self._call_service(self._cli_set_out_reg, req, "set_output_register_int")
        return bool(res and res.success)

    def _get_output_register_int(self, address: int, timeout_ms: int = 1000) -> tuple[bool, int]:
        req = GetOutputRegisterInt.Request()
        req.address = int(address)
        req.timeout_ms = int(timeout_ms)
        res = self._call_service(self._cli_get_out_reg, req, "get_output_register_int")
        if not res or not res.success:
            return False, 0
        return True, int(res.value)

    def _build_drl_read_present_position(self) -> str:
        read_pkt = list(ModbusRTU.fc03(self._slave_id, Reg.PRESENT_POSITION, 2))
        # DRL에서 수신 바이트 길이가 충분하면 pos를 PLC output register에 기록한다.
        # 코드: 1=ok, -1=rx error
        return (
            "flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)\n"
            "wait(0.05)\n"
            f"set_output_register_int({self._feedback_plc_addr_code}, 0)\n"
            f"flange_serial_write(bytes({read_pkt}))\n"
            "wait(0.02)\n"
            "rx = flange_serial_read(timeout=0.2)\n"
            "if len(rx) >= 7:\n"
            "    p = rx[3] * 256 + rx[4]\n"
            f"    set_output_register_int({self._feedback_plc_addr_pos}, p)\n"
            f"    set_output_register_int({self._feedback_plc_addr_code}, 1)\n"
            "else:\n"
            f"    set_output_register_int({self._feedback_plc_addr_code}, -1)\n"
            "flange_serial_close()\n"
        )

    def _read_present_position_feedback(self) -> tuple[bool, int]:
        req = DrlStart.Request()
        req.robot_system = 0
        req.code = self._build_drl_read_present_position()
        res = self._call_service(self._cli_drl, req, "read_present_position")
        if not res or not res.success:
            return False, 0

        ok_code, code = self._get_output_register_int(self._feedback_plc_addr_code, timeout_ms=500)
        if not ok_code or code != 1:
            return False, 0
        ok_pos, pos = self._get_output_register_int(self._feedback_plc_addr_pos, timeout_ms=500)
        if not ok_pos:
            return False, 0
        return True, pos

    def _wait_until_target_position(self, target_pulse: int) -> tuple[bool, str]:
        poll_dt = 1.0 / self._feedback_poll_hz if self._feedback_poll_hz > 0 else 0.125
        deadline = time.monotonic() + self._feedback_max_wait_sec
        stable_count = 0
        need_stable = 2

        while time.monotonic() < deadline:
            ok, pos = self._read_present_position_feedback()
            if ok and abs(int(pos) - int(target_pulse)) <= self._feedback_pos_tolerance:
                stable_count += 1
                if stable_count >= need_stable:
                    return True, f"feedback_ok(pos={pos}, target={target_pulse})"
            else:
                stable_count = 0
            time.sleep(poll_dt)

        return False, f"feedback_timeout(target={target_pulse})"

    def _init_once(self):
        self._init_timer.cancel()
        waited = 0.0
        while (
            (not self._cli_drl.service_is_ready())
            or (not self._cli_get_out_reg.service_is_ready())
            or (not self._cli_set_out_reg.service_is_ready())
        ) and waited < 10.0:
            time.sleep(0.5)
            waited += 0.5

        if (
            (not self._cli_drl.service_is_ready())
            or (not self._cli_get_out_reg.service_is_ready())
            or (not self._cli_set_out_reg.service_is_ready())
        ):
            self.get_logger().error("drl/plc 서비스 연결 실패")
            return

        pkts = [
            ModbusRTU.fc06(self._slave_id, Reg.TORQUE_ENABLE, 1),
            ModbusRTU.fc06(self._slave_id, Reg.GOAL_CURRENT, self._cur_init),
        ]
        if self._run_packets(pkts, motion_wait=0.0, label="gripper_init"):
            self._torque = True
            self._ready = True
            self.get_logger().info("그리퍼 초기화 완료")
        else:
            self.get_logger().error("그리퍼 초기화 실패")

    def _move(self, pulse: int, current: int) -> tuple[bool, str]:
        if not self._ready:
            return False, "초기화 미완료"
        if not self._torque:
            return False, "토크 OFF 상태"

        p = max(0, min(int(pulse), 1000))
        c = max(1, min(int(current), 1000))
        pkts = [
            ModbusRTU.fc06(self._slave_id, Reg.GOAL_CURRENT, c),
            ModbusRTU.fc16(self._slave_id, Reg.GOAL_POSITION, [p, 0]),
        ]
        # 목표 전송 자체는 짧게 보내고, 완료 판단은 실제 위치 피드백으로 한다.
        ok = self._run_packets(pkts, motion_wait=0.0, label=f"move({p},{c})")
        if ok:
            ok_fb, fb_msg = self._wait_until_target_position(target_pulse=p)
            if not ok_fb:
                if self._feedback_strict:
                    return False, fb_msg
                # 피드백이 지원되지 않는 컨트롤러/환경에서는 시간 기반으로 degrade.
                self.get_logger().warn(
                    f"피드백 판정 실패({fb_msg}). "
                    f"fallback으로 motion_wait={self._motion_wait}s 적용"
                )
                if self._motion_wait > 0:
                    time.sleep(self._motion_wait)
            # 피드백으로 목표 도달 확인 후, 필요시 최소 안정화 시간만 추가
            if self._done_settle_sec > 0:
                time.sleep(self._done_settle_sec)
            self._stroke = p
            return True, f"완료(pulse={p}, current={c})"
        return False, "실패"

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper_joint"]
        msg.position = [float(self._stroke)]
        msg.velocity = [0.0]
        msg.effort = [float(self._cur_init if self._torque else 0)]
        self._state_pub.publish(msg)

    def _sub_move_pulse(self, msg: Int32):
        with self._action_lock:
            ok, text = self._move(msg.data, self._cur_cube)
        if ok:
            self.get_logger().info(f"/gripper/move_pulse 적용: {text}")
        else:
            self.get_logger().warn(f"/gripper/move_pulse 실패: {text}")

    def _sub_command(self, msg: String):
        try:
            data = json.loads(msg.data)
            cmd_id = int(data.get("cmd_id", -1))
            action = str(data.get("action", "")).strip().lower()
        except Exception as e:
            self.get_logger().warn(f"/gripper/command 파싱 실패: {e}")
            self._publish_event(-1, "parse", "failed", f"parse_error: {e}")
            return

        if action in {"release", "open"}:
            with self._action_lock:
                ok, text = self._move(self._pulse_open, self._cur_init)
        elif action == "grab_cube":
            with self._action_lock:
                ok, text = self._move(self._pulse_cube, self._cur_cube)
        elif action == "grab_rotate":
            with self._action_lock:
                ok, text = self._move(self._pulse_rotate, self._cur_cube)
        elif action == "grab_repose":
            with self._action_lock:
                ok, text = self._move(self._pulse_repose, self._cur_cube)
        elif action == "move_pulse":
            pulse = int(data.get("pulse", self._pulse_open))
            current = int(data.get("current", self._cur_cube))
            with self._action_lock:
                ok, text = self._move(pulse, current)
        else:
            ok, text = False, f"unsupported action: {action}"

        self._publish_event(cmd_id, action, "done" if ok else "failed", text)

    def _srv_open(self, _req: Trigger.Request, res: Trigger.Response):
        with self._action_lock:
            res.success, res.message = self._move(self._pulse_open, self._cur_init)
        return res

    def _srv_grab_cube(self, _req: Trigger.Request, res: Trigger.Response):
        with self._action_lock:
            res.success, res.message = self._move(self._pulse_cube, self._cur_cube)
        return res

    def _srv_grab_rotate(self, _req: Trigger.Request, res: Trigger.Response):
        with self._action_lock:
            res.success, res.message = self._move(self._pulse_rotate, self._cur_cube)
        return res

    def _srv_grab_repose(self, _req: Trigger.Request, res: Trigger.Response):
        with self._action_lock:
            res.success, res.message = self._move(self._pulse_repose, self._cur_cube)
        return res

    def _srv_stop(self, _req: Trigger.Request, res: Trigger.Response):
        pkts = [ModbusRTU.fc06(self._slave_id, Reg.TORQUE_ENABLE, 0)]
        res.success = self._run_packets(pkts, motion_wait=0.0, label="torque_off")
        res.message = "토크 OFF" if res.success else "실패"
        if res.success:
            self._torque = False
        return res

    def _srv_enable(self, req: SetBool.Request, res: SetBool.Response):
        val = 1 if req.data else 0
        pkts = [ModbusRTU.fc06(self._slave_id, Reg.TORQUE_ENABLE, val)]
        label = "torque_on" if req.data else "torque_off"
        res.success = self._run_packets(pkts, motion_wait=0.0, label=label)
        res.message = "완료" if res.success else "실패"
        if res.success:
            self._torque = req.data
        return res

    def destroy_node(self):
        if self._ready:
            pkts = [ModbusRTU.fc06(self._slave_id, Reg.TORQUE_ENABLE, 0)]
            self._run_packets(pkts, motion_wait=0.0, label="shutdown")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GripperControlNode()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
