#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from dsr_msgs2.srv import DrlStart, GetDrlState
from rclpy.node import Node
from std_srvs.srv import Trigger


class PerceptionTimedCaptureRunner(Node):
    def __init__(self) -> None:
        super().__init__("perception_timed_capture_runner")

        self.declare_parameter("robot_ns", "dsr01")
        self.declare_parameter("robot_system", 0)
        self.declare_parameter(
            "script_path",
            "/home/junsoo/cube_solver_ws/src/cube_solver/cube_solver/motions/Perception.drl",
        )
        self.declare_parameter(
            "capture_offsets_sec",
            [11.8, 14.5, 17.7, 20.8, 47.4, 52.5],
        )
        self.declare_parameter("capture_service", "/cube/capture_once")
        self.declare_parameter("record_dir", "/home/junsoo/cube_solver_ws/capture")
        self.declare_parameter("timeout_sec", 120.0)
        self.declare_parameter("auto_run", False)

        self._robot_ns = str(self.get_parameter("robot_ns").value).strip("/")
        self._robot_system = int(self.get_parameter("robot_system").value)
        self._script_path = Path(str(self.get_parameter("script_path").value)).expanduser()
        self._capture_offsets = [
            float(x) for x in self.get_parameter("capture_offsets_sec").value
        ]
        self._capture_service = str(self.get_parameter("capture_service").value)
        self._record_dir = Path(str(self.get_parameter("record_dir").value)).expanduser()
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)
        self._auto_run = bool(self.get_parameter("auto_run").value)

        self._record_dir.mkdir(parents=True, exist_ok=True)

        drl_ns = self._robot_ns if self._robot_ns else "dsr01"
        self._drl_cli = self.create_client(DrlStart, f"/{drl_ns}/drl/drl_start")
        self._state_cli = self.create_client(GetDrlState, f"/{drl_ns}/drl/get_drl_state")
        self._capture_cli = self.create_client(Trigger, self._capture_service)
        self.create_service(Trigger, "/cube/run_perception_timed_capture", self._srv_run)

        while not self._drl_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스 대기 중: drl/drl_start")
        while not self._state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스 대기 중: drl/get_drl_state")
        while not self._capture_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"서비스 대기 중: {self._capture_service}")

        self.get_logger().info(
            "perception_timed_capture_runner 준비 완료. "
            "service: /cube/run_perception_timed_capture"
        )

        if self._auto_run:
            threading.Thread(target=self._run_and_log, daemon=True).start()

    def _read_script(self) -> tuple[bool, str]:
        if not self._script_path.is_file():
            return False, f"script 파일 없음: {self._script_path}"
        try:
            code = self._script_path.read_text(encoding="utf-8")
        except OSError as e:
            return False, f"script 파일 읽기 실패: {e}"
        if not code.strip():
            return False, "script가 비어 있습니다."
        return True, code

    def _start_drl(self, code: str) -> tuple[bool, str]:
        req = DrlStart.Request()
        req.robot_system = self._robot_system
        req.code = code
        fut = self._drl_cli.call_async(req)
        evt = threading.Event()
        fut.add_done_callback(lambda _f: evt.set())
        if not evt.wait(timeout=self._timeout_sec):
            return False, "DrlStart 타임아웃"
        try:
            res = fut.result()
        except Exception as e:
            return False, f"DrlStart 예외: {e}"
        if not res or not res.success:
            return False, "DrlStart 실패 응답"
        return True, "ok"

    def _capture_once(self) -> tuple[bool, str]:
        req = Trigger.Request()
        fut = self._capture_cli.call_async(req)
        evt = threading.Event()
        fut.add_done_callback(lambda _f: evt.set())
        if not evt.wait(timeout=self._timeout_sec):
            return False, "capture_once 타임아웃"
        try:
            res = fut.result()
        except Exception as e:
            return False, f"capture_once 예외: {e}"
        if not res or not res.success:
            msg = res.message if res else "빈 응답"
            return False, f"capture_once 실패: {msg}"
        return True, res.message

    def _wait_drl_end(self) -> tuple[bool, str]:
        # get_drl_state: PLAY=0, STOP=1
        deadline = time.monotonic() + self._timeout_sec
        while time.monotonic() < deadline:
            req = GetDrlState.Request()
            fut = self._state_cli.call_async(req)
            evt = threading.Event()
            fut.add_done_callback(lambda _f: evt.set())
            if not evt.wait(timeout=2.0):
                time.sleep(0.1)
                continue
            try:
                res = fut.result()
            except Exception:
                time.sleep(0.1)
                continue
            if not res or not res.success:
                time.sleep(0.1)
                continue
            if int(res.drl_state) != 0:
                return True, "ok"
            time.sleep(0.1)
        return False, "DRL 종료 대기 타임아웃"

    def _save_run_record(self, start_ts: str, rows: list[dict]) -> None:
        out = {
            "started_at": start_ts,
            "script_path": str(self._script_path),
            "offsets_sec": self._capture_offsets,
            "captures": rows,
        }
        filename = f"timed_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self._record_dir / filename
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_sequence(self) -> tuple[bool, str]:
        ok, code_or_msg = self._read_script()
        if not ok:
            return False, code_or_msg

        ok, msg = self._start_drl(code_or_msg)
        if not ok:
            return False, msg

        start = time.monotonic()
        start_ts = datetime.now().isoformat(timespec="seconds")
        rows: list[dict] = []

        for i, off in enumerate(self._capture_offsets, start=1):
            sleep_sec = off - (time.monotonic() - start)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            actual = time.monotonic() - start
            self.get_logger().info(
                f"[{i}/{len(self._capture_offsets)}] capture_once 호출 "
                f"(target={off:.1f}s, actual={actual:.2f}s)"
            )
            ok, msg = self._capture_once()
            rows.append(
                {
                    "index": i,
                    "target_sec": off,
                    "actual_sec": round(actual, 3),
                    "success": ok,
                    "message": msg,
                }
            )
            if not ok:
                self._save_run_record(start_ts, rows)
                return False, msg

        ok, msg = self._wait_drl_end()
        self._save_run_record(start_ts, rows)
        if not ok:
            return False, msg
        return True, "Perception.drl 타이밍 캡처 완료"

    def _run_and_log(self) -> None:
        ok, msg = self._run_sequence()
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)

    def _srv_run(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        ok, msg = self._run_sequence()
        res.success = ok
        res.message = msg
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionTimedCaptureRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
