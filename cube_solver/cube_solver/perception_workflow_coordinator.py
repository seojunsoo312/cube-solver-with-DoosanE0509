#!/usr/bin/env python3
from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class PerceptionWorkflowCoordinator(Node):
    def __init__(self) -> None:
        super().__init__("perception_workflow_coordinator")

        self.declare_parameter("run_drl_service", "/cube/run_drl_block")
        self.declare_parameter("scan_service", "/cube/scan_and_solve")
        self.declare_parameter("publish_solution_service", "/cube/publish_solution_once")
        self.declare_parameter("service_timeout_sec", 120.0)

        run_drl_service = str(self.get_parameter("run_drl_service").value)
        scan_service = str(self.get_parameter("scan_service").value)
        publish_solution_service = str(
            self.get_parameter("publish_solution_service").value
        )
        self._service_timeout_sec = float(
            self.get_parameter("service_timeout_sec").value
        )

        self._run_drl_cli = self.create_client(Trigger, run_drl_service)
        self._scan_cli = self.create_client(Trigger, scan_service)
        self._publish_solution_cli = self.create_client(Trigger, publish_solution_service)

        self.create_service(Trigger, "/cube/trigger_perception", self._srv_trigger_perception)
        self.create_service(Trigger, "/cube/trigger_solve", self._srv_trigger_solve)

        for cli, name in (
            (self._run_drl_cli, run_drl_service),
            (self._scan_cli, scan_service),
            (self._publish_solution_cli, publish_solution_service),
        ):
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"서비스 대기 중: {name}")

        self.get_logger().info(
            "perception_workflow_coordinator 준비 완료. "
            "services: /cube/trigger_perception, /cube/trigger_solve"
        )

    def _call_trigger(self, client: rclpy.client.Client, label: str) -> tuple[bool, str]:
        req = Trigger.Request()
        fut = client.call_async(req)
        evt = threading.Event()
        fut.add_done_callback(lambda _f: evt.set())
        if not evt.wait(timeout=self._service_timeout_sec):
            return False, f"{label} 타임아웃({self._service_timeout_sec}s)"
        try:
            res = fut.result()
        except Exception as e:
            return False, f"{label} 예외: {e}"
        if not res or not res.success:
            msg = res.message if res else "빈 응답"
            return False, f"{label} 실패: {msg}"
        return True, res.message

    def _srv_trigger_perception(
        self, _req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        ok, msg = self._call_trigger(self._run_drl_cli, "run_drl_block")
        if not ok:
            res.success = False
            res.message = msg
            return res

        ok, msg = self._call_trigger(self._scan_cli, "scan_and_solve")
        res.success = ok
        res.message = msg
        return res

    def _srv_trigger_solve(
        self, _req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        ok, msg = self._call_trigger(self._publish_solution_cli, "publish_solution_once")
        res.success = ok
        res.message = msg
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionWorkflowCoordinator()
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
