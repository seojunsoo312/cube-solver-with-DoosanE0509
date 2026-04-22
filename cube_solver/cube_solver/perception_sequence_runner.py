#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from pathlib import Path

import rclpy
from dsr_msgs2.srv import DrlStart, GetDrlState
from rclpy.node import Node
from std_srvs.srv import Trigger


class PerceptionSequenceRunner(Node):
    def __init__(self) -> None:
        super().__init__("perception_sequence_runner")

        self.declare_parameter("robot_ns", "dsr01")
        self.declare_parameter("robot_system", 0)
        self.declare_parameter(
            "motions_dir",
            str(Path(__file__).resolve().parent / "motions"),
        )
        self.declare_parameter(
            "step_files",
            [
                "Perception_step1.drl",
                "Perception_step2.drl",
                "Perception_step3.drl",
                "Perception_step4.drl",
                "Perception_step5.drl",
                "Perception_step6.drl",
            ],
        )
        self.declare_parameter("capture_service", "/cube/capture_once")
        self.declare_parameter("timeout_sec", 60.0)
        self.declare_parameter("auto_run", False)

        self._robot_ns = str(self.get_parameter("robot_ns").value).strip("/")
        self._robot_system = int(self.get_parameter("robot_system").value)
        self._motions_dir = Path(str(self.get_parameter("motions_dir").value)).expanduser()
        self._step_files = [str(x) for x in self.get_parameter("step_files").value]
        self._capture_service = str(self.get_parameter("capture_service").value)
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)
        self._auto_run = bool(self.get_parameter("auto_run").value)

        drl_ns = self._robot_ns if self._robot_ns else "dsr01"
        self._drl_cli = self.create_client(DrlStart, f"/{drl_ns}/drl/drl_start")
        self._state_cli = self.create_client(GetDrlState, f"/{drl_ns}/drl/get_drl_state")
        self._capture_cli = self.create_client(Trigger, self._capture_service)
        self.create_service(Trigger, "/cube/run_perception_sequence", self._srv_run)

        while not self._drl_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스 대기 중: drl/drl_start")
        while not self._state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스 대기 중: drl/get_drl_state")
        while not self._capture_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"서비스 대기 중: {self._capture_service}")

        self.get_logger().info(
            "perception_sequence_runner 준비 완료. "
            "service: /cube/run_perception_sequence"
        )

        if self._auto_run:
            threading.Thread(target=self._run_and_log, daemon=True).start()

    def _read_step_code(self, filename: str) -> tuple[bool, str]:
        path = self._motions_dir / filename
        if not path.is_file():
            return False, f"step 파일 없음: {path}"
        try:
            code = path.read_text(encoding="utf-8")
        except OSError as e:
            return False, f"step 파일 읽기 실패: {e}"
        if not code.strip():
            return False, f"step 파일이 비어 있음: {path}"
        return True, code

    def _start_drl(self, code: str, label: str) -> tuple[bool, str]:
        req = DrlStart.Request()
        req.robot_system = self._robot_system
        req.code = code
        fut = self._drl_cli.call_async(req)
        evt = threading.Event()
        fut.add_done_callback(lambda _f: evt.set())
        if not evt.wait(timeout=self._timeout_sec):
            return False, f"DrlStart 타임아웃: {label}"
        try:
            res = fut.result()
        except Exception as e:
            return False, f"DrlStart 예외: {e}"
        if not res or not res.success:
            return False, f"DrlStart 실패 응답: {label}"
        return True, "ok"

    def _wait_drl_done(self, label: str) -> tuple[bool, str]:
        # get_drl_state: PLAY=0
        play_state = 0
        deadline = time.monotonic() + self._timeout_sec
        seen_play = False
        non_play_streak = 0
        start_t = time.monotonic()
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

            st = int(res.drl_state)
            if st == play_state:
                seen_play = True
                non_play_streak = 0
            elif seen_play:
                return True, "ok"
            else:
                # 짧은 DRL은 첫 폴링 시 이미 종료(non-PLAY)일 수 있다.
                non_play_streak += 1
                if non_play_streak >= 5 and (time.monotonic() - start_t) > 0.8:
                    self.get_logger().warn(
                        f"{label}: PLAY 상태를 관측하지 못했지만 빠르게 종료된 것으로 간주합니다."
                    )
                    return True, "ok"
            time.sleep(0.1)
        return False, f"DRL 완료 대기 타임아웃: {label}"

    def _capture_once(self, label: str) -> tuple[bool, str]:
        req = Trigger.Request()
        fut = self._capture_cli.call_async(req)
        evt = threading.Event()
        fut.add_done_callback(lambda _f: evt.set())
        if not evt.wait(timeout=self._timeout_sec):
            return False, f"capture_once 타임아웃: {label}"
        try:
            res = fut.result()
        except Exception as e:
            return False, f"capture_once 예외: {e}"
        if not res or not res.success:
            msg = res.message if res else "빈 응답"
            return False, f"capture_once 실패: {msg}"
        return True, res.message

    def _run_sequence(self) -> tuple[bool, str]:
        for idx, step_file in enumerate(self._step_files, start=1):
            ok, code_or_msg = self._read_step_code(step_file)
            if not ok:
                return False, code_or_msg

            label = f"{idx}/{len(self._step_files)} {step_file}"
            self.get_logger().info(f"DRL 실행: {label}")
            ok, msg = self._start_drl(code_or_msg, label)
            if not ok:
                return False, msg
            ok, msg = self._wait_drl_done(label)
            if not ok:
                return False, msg

            ok, msg = self._capture_once(label)
            if not ok:
                return False, msg
            self.get_logger().info(f"캡처 완료: {msg}")
        return True, "Perception 6단계 + 자동 캡처 완료"

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
    node = PerceptionSequenceRunner()
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
