#!/usr/bin/env python3
"""큐브 마스터 노드.

rules 문서의 동작 순서를 반영해 kociemba 솔루션 문자열을 실제 로봇/그리퍼 명령으로 실행한다.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import rclpy
from dsr_msgs2.srv import DrlStart, GetDrlState, MoveJoint, MoveLine, MoveWait
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .gripper_drl_controller import GripperController
from .motion_drl_loader import load_merged_drl


class CubeMasterNode(Node):
    # rules 좌표/명령 정의
    INITIAL_STATE = [373.030, 0.000, 74.660, 22.85, -180.0, 22.85]
    Z_DOWN_GRAB_TWO = [373.010, 0.000, 20.000, 22.85, -180.0, 22.85]
    Z_LITTLE_UP_GRAB_TWO = [373.020, 0.000, 27.730, 6.68, -180.0, 6.68]
    L_DROP_POS = [373.030, 0.000, 48.530, 90.0, 90.0, 90.0]
    R_DROP_POS = [373.030, 0.000, 48.530, 90.0, -90.0, 90.0]

    Z_40MM_DOWN = [0.0, 0.0, -40.0, 0.0, 0.0, 0.0]
    Z_40MM_UP = [0.0, 0.0, 40.0, 0.0, 0.0, 0.0]
    Z_10MM_DOWN = [0.0, 0.0, -10.0, 0.0, 0.0, 0.0]
    X_17MM_DOWN_Z_10MM_DOWN = [-17.0, 0.0, -15.0, 0.0, 0.0, 0.0]
    X_17MM_UP_Z_10MM_DOWN = [17.0, 0.0, -15.0, 0.0, 0.0, 0.0]

    VALID_SUFFIX = {"": 90.0, "'": -90.0, "2": 180.0}
    VALID_FACE = {"U", "R", "F", "D", "L", "B"}

    def __init__(self) -> None:
        super().__init__("cube_master_node")

        self.declare_parameter("robot_ns", "")
        self.declare_parameter("use_drl_chunks", True)
        self.declare_parameter(
            "motions_dir",
            str(Path(__file__).resolve().parent / "motions"),
        )
        self.declare_parameter("drl_prepend_path", "")
        self.declare_parameter("drl_chunk_timeout_sec", 120.0)
        # If get_drl_state polling fails, set drl_post_chunk_sleep_sec e.g. 8.0
        self.declare_parameter("drl_post_chunk_sleep_sec", 0.0)
        self.declare_parameter("robot_system", 0)
        self.declare_parameter("auto_execute_solution", True)
        # 퍼셉션 없이 수동 테스트: true면 /cube/solution_manual도 실행(퍼셉션과 동시에 켜면 이중 실행될 수 있음)
        self.declare_parameter("subscribe_solution_manual", False)
        self.declare_parameter("joint_vel", 15.0)
        self.declare_parameter("joint_acc", 15.0)
        self.declare_parameter("line_vel", [60.0, 60.0])
        self.declare_parameter("line_acc", [120.0, 120.0])
        self.declare_parameter("step_sleep", 0.15)
        self.declare_parameter("gripper_wait_sec", 1.0)
        self.declare_parameter("service_timeout_sec", 10.0)
        self.declare_parameter("gripper_timeout_sec", 10.0)
        self.declare_parameter("pulse_open", 200)
        self.declare_parameter("pulse_cube", 420)
        self.declare_parameter("pulse_rotate", 420)
        self.declare_parameter("pulse_repose", 420)

        self._robot_ns = str(self.get_parameter("robot_ns").value).strip("/")
        self._use_drl_chunks = bool(self.get_parameter("use_drl_chunks").value)
        self._motions_dir = Path(
            str(self.get_parameter("motions_dir").value)
        ).expanduser()
        self._drl_prepend_path = str(self.get_parameter("drl_prepend_path").value).strip()
        self._drl_chunk_timeout_sec = float(
            self.get_parameter("drl_chunk_timeout_sec").value
        )
        self._drl_post_chunk_sleep_sec = float(
            self.get_parameter("drl_post_chunk_sleep_sec").value
        )
        self._robot_system = int(self.get_parameter("robot_system").value)
        self._auto_execute = bool(self.get_parameter("auto_execute_solution").value)
        self._subscribe_solution_manual = bool(
            self.get_parameter("subscribe_solution_manual").value
        )
        self._joint_vel = float(self.get_parameter("joint_vel").value)
        self._joint_acc = float(self.get_parameter("joint_acc").value)
        self._line_vel = [float(x) for x in self.get_parameter("line_vel").value]
        self._line_acc = [float(x) for x in self.get_parameter("line_acc").value]
        self._step_sleep = float(self.get_parameter("step_sleep").value)
        self._gripper_wait_sec = float(self.get_parameter("gripper_wait_sec").value)
        self._service_timeout_sec = float(self.get_parameter("service_timeout_sec").value)
        self._gripper_timeout_sec = float(self.get_parameter("gripper_timeout_sec").value)
        self._pulse_open = int(self.get_parameter("pulse_open").value)
        self._pulse_cube = int(self.get_parameter("pulse_cube").value)
        self._pulse_rotate = int(self.get_parameter("pulse_rotate").value)
        self._pulse_repose = int(self.get_parameter("pulse_repose").value)
        self._cb_group = ReentrantCallbackGroup()

        self._move_joint_cli = None
        self._move_line_cli = None
        self._move_wait_cli = None
        self._drl_cli = None
        self._get_drl_state_cli = None

        if self._use_drl_chunks:
            drl_ns = self._robot_ns if self._robot_ns else "dsr01"
            self._drl_cli = self.create_client(
                DrlStart,
                f"/{drl_ns}/drl/drl_start",
                callback_group=self._cb_group,
            )
            # DRL 청크는 drl_start 응답만으로는 모션 종료를 보장하지 않음 → 다음 청크 전 move_wait 필수
            self._get_drl_state_cli = self.create_client(
                GetDrlState,
                f"/{drl_ns}/drl/get_drl_state",
                callback_group=self._cb_group,
            )
            self._move_wait_cli = self.create_client(
                MoveWait,
                f"/{drl_ns}/motion/move_wait",
                callback_group=self._cb_group,
            )
        else:
            self._move_joint_cli = self.create_client(
                MoveJoint, self._svc("/motion/move_joint"), callback_group=self._cb_group
            )
            self._move_line_cli = self.create_client(
                MoveLine, self._svc("/motion/move_line"), callback_group=self._cb_group
            )
            self._move_wait_cli = self.create_client(
                MoveWait, self._svc("/motion/move_wait"), callback_group=self._cb_group
            )

        self.create_subscription(
            String, "/cube/solution", self._on_solution, 10, callback_group=self._cb_group
        )
        if self._subscribe_solution_manual:
            self.create_subscription(
                String,
                "/cube/solution_manual",
                self._on_solution,
                10,
                callback_group=self._cb_group,
            )
        self.create_service(
            Trigger, "/cube/execute_last_solution", self._srv_execute_last_solution,
            callback_group=self._cb_group
        )

        self._last_solution = ""
        self._lock = threading.Lock()
        self._busy = False

        self._wait_services()

        self._gripper: GripperController | None = None
        self._gripper_initialized = False
        if not self._use_drl_chunks:
            gripper_ns = self._robot_ns if self._robot_ns else "dsr01"
            self._gripper = GripperController(
                self,
                namespace=gripper_ns,
                timeout_sec=self._gripper_timeout_sec,
            )

        mode = "drl_chunks" if self._use_drl_chunks else "ros_motion+gripper_drl"
        self.get_logger().info(
            f"cube_master_node 준비 완료 (실행 모드: {mode}). "
            "구독: /cube/solution"
            + (", /cube/solution_manual" if self._subscribe_solution_manual else "")
            + ", 서비스: /cube/execute_last_solution"
        )
        if self._use_drl_chunks:
            self.get_logger().info(f"motions_dir={self._motions_dir}")

    def _svc(self, base: str) -> str:
        if self._robot_ns:
            return f"/{self._robot_ns}{base}"
        return base

    def _wait_services(self) -> None:
        if self._use_drl_chunks:
            assert self._drl_cli is not None
            assert self._get_drl_state_cli is not None
            assert self._move_wait_cli is not None
            for cli, label in (
                (self._drl_cli, "drl/drl_start"),
                (self._get_drl_state_cli, "drl/get_drl_state"),
                (self._move_wait_cli, "motion/move_wait"),
            ):
                while not cli.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(f"서비스 대기 중: {label}")
            return

        services = [
            (self._move_joint_cli, self._svc("/motion/move_joint")),
            (self._move_line_cli, self._svc("/motion/move_line")),
            (self._move_wait_cli, self._svc("/motion/move_wait")),
        ]
        for cli, name in services:
            assert cli is not None
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"서비스 대기 중: {name}")

    def _move_wait(self, timeout_sec: float | None = None) -> bool:
        assert self._move_wait_cli is not None
        tmo = float(self._service_timeout_sec if timeout_sec is None else timeout_sec)
        req = MoveWait.Request()
        fut = self._move_wait_cli.call_async(req)
        done_evt = threading.Event()
        fut.add_done_callback(lambda _f: done_evt.set())
        if not done_evt.wait(timeout=tmo):
            self.get_logger().error(f"move_wait 타임아웃({tmo}s)")
            return False
        res = fut.result()
        if not res or not res.success:
            self.get_logger().error("move_wait 실패")
            return False
        return True

    def _move_joint(self, pos: list[float], rel: bool = False) -> bool:
        assert self._move_joint_cli is not None
        req = MoveJoint.Request()
        req.pos = [float(x) for x in pos]
        req.vel = self._joint_vel
        req.acc = self._joint_acc
        req.time = 0.0
        req.radius = 0.0
        req.mode = 1 if rel else 0
        req.blend_type = 0
        req.sync_type = 0
        fut = self._move_joint_cli.call_async(req)
        done_evt = threading.Event()
        fut.add_done_callback(lambda _f: done_evt.set())
        if not done_evt.wait(timeout=self._service_timeout_sec):
            self.get_logger().error(
                f"move_joint 타임아웃({self._service_timeout_sec}s, rel={rel}, pos={pos})"
            )
            return False
        res = fut.result()
        if not res or not res.success:
            self.get_logger().error(f"move_joint 실패(rel={rel}, pos={pos})")
            return False
        return self._move_wait()

    def _move_line(self, pos: list[float], rel: bool = False) -> bool:
        assert self._move_line_cli is not None
        req = MoveLine.Request()
        req.pos = [float(x) for x in pos]
        req.vel = [self._line_vel[0], self._line_vel[1]]
        req.acc = [self._line_acc[0], self._line_acc[1]]
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0
        req.mode = 1 if rel else 0
        req.blend_type = 0
        req.sync_type = 0
        fut = self._move_line_cli.call_async(req)
        done_evt = threading.Event()
        fut.add_done_callback(lambda _f: done_evt.set())
        if not done_evt.wait(timeout=self._service_timeout_sec):
            self.get_logger().error(
                f"move_line 타임아웃({self._service_timeout_sec}s, rel={rel}, pos={pos})"
            )
            return False
        res = fut.result()
        if not res or not res.success:
            self.get_logger().error(f"move_line 실패(rel={rel}, pos={pos})")
            return False
        return self._move_wait()

    def _joint6_rel(self, degree: float) -> bool:
        return self._move_joint([0.0, 0.0, 0.0, 0.0, 0.0, float(degree)], rel=True)

    def _ensure_gripper_ready(self) -> bool:
        if self._gripper is None:
            return False
        if self._gripper_initialized:
            return True
        self.get_logger().info("그리퍼 초기화 시도...")
        ok = self._gripper.initialize()
        if ok:
            self._gripper_initialized = True
            self.get_logger().info("그리퍼 초기화 완료")
            return True
        self.get_logger().error("그리퍼 초기화 실패")
        return False

    def _gripper_do(self, action: str) -> bool:
        if not self._ensure_gripper_ready():
            return False

        if action in {"release", "open"}:
            ok = self._gripper.move(self._pulse_open)
        elif action == "grab_cube":
            ok = self._gripper.move(self._pulse_cube)
        elif action == "grab_rotate":
            ok = self._gripper.move(self._pulse_rotate)
        elif action == "grab_repose":
            ok = self._gripper.move(self._pulse_repose)
        else:
            self.get_logger().error(f"지원하지 않는 gripper action: {action}")
            return False

        if not ok:
            self.get_logger().error(f"gripper {action} 실패")
            return False

        # DRL 요청 성공 시점과 실제 기구 동작 완료 시점 차이를 보정한다.
        if self._gripper_wait_sec > 0:
            time.sleep(self._gripper_wait_sec)
        return True

    def _gripper_release(self) -> bool:
        return self._gripper_do("release")

    def _gripper_grab_cube(self) -> bool:
        return self._gripper_do("grab_cube")

    def _gripper_grab_rotate(self) -> bool:
        return self._gripper_do("grab_rotate")

    def _gripper_grab_repose(self) -> bool:
        return self._gripper_do("grab_repose")

    def _run_steps(self, name: str, steps: list[Callable[[], bool]]) -> bool:
        self.get_logger().info(f"▶ {name}")
        for i, step in enumerate(steps, start=1):
            if not step():
                self.get_logger().error(f"{name} 실패 (step {i})")
                return False
            if self._step_sleep > 0:
                time.sleep(self._step_sleep)
        return True

    # ---- rules 동작 시퀀스 ----
    def _rotate_top(self, delta_j6: float) -> bool:
        return self._run_steps(
            f"TOP 회전 {delta_j6:+.0f}",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.Z_40MM_DOWN, rel=True),
                self._gripper_grab_rotate,
                lambda: self._joint6_rel(delta_j6),
                self._gripper_release,
                lambda: self._move_line(self.Z_40MM_UP, rel=True),
            ],
        )

    def _r_to_u(self) -> bool:
        return self._run_steps(
            "R to U",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.Z_40MM_DOWN, rel=True),
                self._gripper_grab_repose,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.R_DROP_POS, rel=False),
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
            ],
        )

    def _l_to_u(self) -> bool:
        return self._run_steps(
            "L to U",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.Z_40MM_DOWN, rel=True),
                self._gripper_grab_repose,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.L_DROP_POS, rel=False),
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
            ],
        )

    def _f_to_u(self) -> bool:
        return self._run_steps(
            "F to U",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._joint6_rel(90.0),
                lambda: self._move_line(self.Z_40MM_DOWN, rel=True),
                self._gripper_grab_repose,
                lambda: self._move_line(self.Z_40MM_UP, rel=True),
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.R_DROP_POS, rel=False),
                self._gripper_release,
            ],
        )

    def _b_to_u(self) -> bool:
        return self._run_steps(
            "B to U",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._joint6_rel(90.0),
                lambda: self._move_line(self.Z_40MM_DOWN, rel=True),
                self._gripper_grab_repose,
                lambda: self._move_line(self.Z_40MM_UP, rel=True),
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.L_DROP_POS, rel=False),
                self._gripper_release,
            ],
        )

    def _u_to_f(self) -> bool:
        return self._run_steps(
            "U to F",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.R_DROP_POS, rel=False),
                self._gripper_grab_repose,
                lambda: self._move_line(self.Z_40MM_UP, rel=True),
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._joint6_rel(90.0),
                lambda: self._move_line(self.X_17MM_DOWN_Z_10MM_DOWN, rel=True),
                lambda: self._move_line(self.Z_10MM_DOWN, rel=True),
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
            ],
        )

    def _u_to_b(self) -> bool:
        return self._run_steps(
            "U to B",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.L_DROP_POS, rel=False),
                self._gripper_grab_repose,
                lambda: self._move_line(self.Z_40MM_UP, rel=True),
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._joint6_rel(90.0),
                lambda: self._move_line(self.X_17MM_UP_Z_10MM_DOWN, rel=True),
                lambda: self._move_line(self.Z_10MM_DOWN, rel=True),
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
            ],
        )

    def _ready_d(self) -> bool:
        return self._run_steps(
            "ready_D",
            [
                self._gripper_release,
                lambda: self._move_line(self.INITIAL_STATE, rel=False),
                lambda: self._move_line(self.Z_DOWN_GRAB_TWO, rel=False),
                self._gripper_grab_cube,
                lambda: self._move_line(self.Z_LITTLE_UP_GRAB_TWO, rel=False),
            ],
        )

    def _do_d(self, delta: float) -> bool:
        return self._run_steps(
            f"D {delta:+.0f}",
            [
                lambda: self._joint6_rel(delta),
                lambda: self._move_line(self.Z_40MM_UP, rel=True),
                lambda: self._joint6_rel(-delta),
                lambda: self._move_line(self.Z_40MM_DOWN, rel=True),
                self._gripper_release,
            ],
        )

    def _drl_basename_for(self, face: str, suffix: str) -> str:
        """토큰 1개당 motions/ 아래 단일 통합 DRL (예: R' -> R_prime.drl)."""
        if suffix == "":
            return f"{face}.drl"
        if suffix == "'":
            return f"{face}_prime.drl"
        if suffix == "2":
            return f"{face}_two.drl"
        raise ValueError(f"지원하지 않는 suffix: {suffix}")

    def _drl_start_code(self, code: str, label: str) -> bool:
        assert self._drl_cli is not None
        if not self._drl_cli.service_is_ready():
            self.get_logger().error("DrlStart 서비스가 준비되지 않았습니다.")
            return False
        req = DrlStart.Request()
        req.robot_system = self._robot_system
        req.code = code
        fut = self._drl_cli.call_async(req)
        done_evt = threading.Event()
        fut.add_done_callback(lambda _f: done_evt.set())
        if not done_evt.wait(timeout=self._drl_chunk_timeout_sec):
            self.get_logger().error(
                f"DrlStart 타임아웃 ({label}, {self._drl_chunk_timeout_sec}s)"
            )
            return False
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"DrlStart 예외 ({label}): {e}")
            return False
        if not res or not res.success:
            self.get_logger().error(f"DrlStart 실패 응답: {label}")
            return False
        return True

    def _get_drl_state_once(self) -> tuple[bool, int]:
        assert self._get_drl_state_cli is not None
        req = GetDrlState.Request()
        fut = self._get_drl_state_cli.call_async(req)
        done_evt = threading.Event()
        fut.add_done_callback(lambda _f: done_evt.set())
        if not done_evt.wait(timeout=2.0):
            return False, -1
        try:
            res = fut.result()
        except Exception:
            return False, -1
        if not res or not res.success:
            return False, -1
        return True, int(res.drl_state)

    def _wait_drl_chunk_motion_done(self, label: str) -> bool:
        """Wait until DRL leaves PLAY (0). motion/move_wait often returns immediately for DRL."""
        DRL_PLAY = 0
        assert self._move_wait_cli is not None
        self._move_wait(timeout_sec=min(2.0, self._drl_chunk_timeout_sec))

        deadline = time.monotonic() + float(self._drl_chunk_timeout_sec)
        time.sleep(0.1)
        seen_play = False

        while time.monotonic() < deadline:
            ok, st = self._get_drl_state_once()
            if ok:
                if st == DRL_PLAY:
                    seen_play = True
                elif seen_play:
                    self.get_logger().info(f"DRL chunk done (get_drl_state): {label}")
                    if self._drl_post_chunk_sleep_sec > 0:
                        time.sleep(self._drl_post_chunk_sleep_sec)
                    return True
            time.sleep(0.15)

        if self._drl_post_chunk_sleep_sec > 0:
            self.get_logger().warn(
                f"{label}: get_drl_state wait timeout, "
                f"sleeping drl_post_chunk_sleep_sec={self._drl_post_chunk_sleep_sec}s"
            )
            time.sleep(self._drl_post_chunk_sleep_sec)
            return True

        self.get_logger().error(
            f"{label}: DRL wait timeout (saw_play={seen_play}). "
            "Try: -p drl_post_chunk_sleep_sec:=8.0"
        )
        return False

    def _execute_token_drl(self, token: str) -> bool:
        token = token.strip()
        if not token:
            return True
        face = token[0]
        suffix = token[1:]
        if face not in self.VALID_FACE:
            raise ValueError(f"지원하지 않는 face: {face}")
        if suffix not in self.VALID_SUFFIX:
            raise ValueError(f"지원하지 않는 suffix: {suffix}")

        if not self._motions_dir.is_dir():
            self.get_logger().error(f"motions_dir이 디렉터리가 아님: {self._motions_dir}")
            return False

        basename = self._drl_basename_for(face, suffix)
        path = self._motions_dir / basename
        ok, merged_or_err = load_merged_drl(path, prepend_path=self._drl_prepend_path)
        if not ok:
            self.get_logger().error(f"{basename} 로드 실패: {merged_or_err}")
            return False
        self.get_logger().info(f"DRL 실행: {basename}")
        if not self._drl_start_code(merged_or_err, basename):
            return False
        if not self._wait_drl_chunk_motion_done(basename):
            return False
        return True

    def _execute_token_ros(self, token: str) -> bool:
        token = token.strip()
        if not token:
            return True
        face = token[0]
        suffix = token[1:]
        if face not in self.VALID_FACE:
            raise ValueError(f"지원하지 않는 face: {face}")
        if suffix not in self.VALID_SUFFIX:
            raise ValueError(f"지원하지 않는 suffix: {suffix}")
        delta = self.VALID_SUFFIX[suffix]

        if face == "U":
            return self._rotate_top(delta)
        if face == "R":
            return self._r_to_u() and self._rotate_top(delta) and self._l_to_u()
        if face == "L":
            return self._l_to_u() and self._rotate_top(delta) and self._r_to_u()
        if face == "F":
            return self._f_to_u() and self._rotate_top(delta) and self._u_to_f()
        if face == "B":
            return self._b_to_u() and self._rotate_top(delta) and self._u_to_b()
        return self._ready_d() and self._do_d(delta)

    def _execute_token(self, token: str) -> bool:
        if self._use_drl_chunks:
            return self._execute_token_drl(token)
        return self._execute_token_ros(token)

    def execute_solution(self, solution: str) -> tuple[bool, str]:
        moves = [m for m in solution.strip().split() if m]
        if not moves:
            return False, "빈 솔루션 문자열"

        self.get_logger().info(f"총 {len(moves)}개 동작 실행 시작: {moves}")
        for i, token in enumerate(moves, start=1):
            self.get_logger().info(f"[{i}/{len(moves)}] {token}")
            try:
                ok = self._execute_token(token)
            except ValueError as e:
                return False, str(e)
            if not ok:
                return False, f"{token} 실행 실패"

        self.get_logger().info("🎉 솔루션 실행 완료")
        return True, "ok"

    def _on_solution(self, msg: String) -> None:
        solution = msg.data.strip()
        if not solution:
            return
        self._last_solution = solution
        self.get_logger().info(f"/cube/solution 수신: {solution}")
        if not self._auto_execute:
            return

        # 동기 실행: 수신 콜백에서 즉시 끝까지 수행한다.
        with self._lock:
            if self._busy:
                self.get_logger().warn("이미 실행 중이어서 무시: /cube/solution")
                return
            self._busy = True

        try:
            ok, msg_text = self.execute_solution(solution)
            if not ok:
                self.get_logger().error(f"솔루션 실행 실패: {msg_text}")
        finally:
            with self._lock:
                self._busy = False

    def _srv_execute_last_solution(
        self, _req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        if not self._last_solution:
            res.success = False
            res.message = "실행할 last_solution 이 없습니다."
            return res

        with self._lock:
            if self._busy:
                res.success = False
                res.message = "이미 솔루션 실행 중입니다."
                return res

        ok, msg = self.execute_solution(self._last_solution)
        res.success = ok
        res.message = msg
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubeMasterNode()
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
