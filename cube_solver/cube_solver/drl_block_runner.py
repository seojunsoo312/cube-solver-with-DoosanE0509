#!/usr/bin/env python3
"""DRL 블록 1회 실행 테스트 노드.

용도:
- DART에서 만든 동작(예: U to D 11단계)을 DRL 한 블록으로 실행 테스트
- script_path에 있는 .drl 텍스트를 /{namespace}/drl/drl_start로 1회 전송

motions/ 청크 실행:
- script_path와 같은 디렉터리에 ``cube_motion_preamble.drl``이 있으면,
  그 내용을 본문 앞에 자동으로 붙여 한 스크립트로 전송한다.
- 비활성화: ``prepend_path``를 빈 문자열로 두고, 동일 폴더에 프리앰블 파일을 두지 않는다.
- 강제 지정: ``prepend_path``에 프리앰블 파일 절대/상대 경로를 넣는다.
"""
from __future__ import annotations

from pathlib import Path
import threading

import rclpy
from dsr_msgs2.srv import DrlStart
from rclpy.node import Node
from std_srvs.srv import Trigger

from .motion_drl_loader import load_merged_drl


class DrlBlockRunner(Node):
    def __init__(self) -> None:
        super().__init__("drl_block_runner")

        self.declare_parameter("robot_ns", "dsr01")
        self.declare_parameter("robot_system", 0)
        self.declare_parameter("script_path", "")
        self.declare_parameter("prepend_path", "")
        self.declare_parameter("auto_run", True)
        self.declare_parameter("timeout_sec", 20.0)

        self._robot_ns = str(self.get_parameter("robot_ns").value).strip("/")
        self._robot_system = int(self.get_parameter("robot_system").value)
        self._script_path = str(self.get_parameter("script_path").value).strip()
        self._prepend_path = str(self.get_parameter("prepend_path").value).strip()
        self._auto_run = bool(self.get_parameter("auto_run").value)
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)

        svc_name = f"/{self._robot_ns}/drl/drl_start"
        self._cli = self.create_client(DrlStart, svc_name)
        self.create_service(Trigger, "/cube/run_drl_block", self._srv_run)

        while not self._cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"서비스 대기 중: {svc_name}")

        self.get_logger().info(
            "drl_block_runner 준비 완료. "
            "service: /cube/run_drl_block, "
            f"script_path='{self._script_path}'"
        )

        if self._auto_run:
            threading.Thread(target=self._run_once_and_report, daemon=True).start()

    def _load_script(self) -> tuple[bool, str]:
        if not self._script_path:
            return False, "script_path가 비어 있습니다."
        return load_merged_drl(Path(self._script_path), prepend_path=self._prepend_path)

    def _run_once(self) -> tuple[bool, str]:
        ok, code_or_err = self._load_script()
        if not ok:
            return False, code_or_err

        req = DrlStart.Request()
        req.robot_system = self._robot_system
        req.code = code_or_err

        fut = self._cli.call_async(req)
        evt = threading.Event()
        fut.add_done_callback(lambda _f: evt.set())
        if not evt.wait(timeout=self._timeout_sec):
            return False, f"DrlStart 타임아웃({self._timeout_sec}s)"

        try:
            res = fut.result()
        except Exception as e:
            return False, f"DrlStart 예외: {e}"
        if not res or not res.success:
            return False, "DrlStart 실패 응답"
        return True, "DrlStart 성공"

    def _run_once_and_report(self) -> None:
        ok, msg = self._run_once()
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)

    def _srv_run(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        ok, msg = self._run_once()
        res.success = ok
        res.message = msg
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DrlBlockRunner()
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

