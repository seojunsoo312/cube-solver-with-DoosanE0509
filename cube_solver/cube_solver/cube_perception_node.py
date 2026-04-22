"""큐브 퍼셉션 노드.

RealSense + OpenCV로 6면(U,R,F,D,L,B)을 순차 캡처하여
54자 상태 문자열(URFDLB)을 /cube/state_raw 로 발행하고,
kociemba.solve() 결과를 /cube/solution 으로 발행한다.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    import kociemba  # type: ignore
except Exception:
    kociemba = None


FRAME_TIMEOUT_MS = 15000
VALID_FACES = "URFDLB"
SOLVER_FACE_ORDER = ("U", "R", "F", "D", "L", "B")
VALID_MOVE_FACES = {"U", "R", "F", "D", "L", "B"}
VALID_MOVE_SUFFIX = {"", "'", "2"}
DU_REORDER_INDEX = (2, 5, 8, 1, 4, 7, 0, 3, 6)


def wait_frames(pipeline: rs.pipeline, timeout_ms: int = FRAME_TIMEOUT_MS):
    return pipeline.wait_for_frames(timeout_ms)


def warmup_pipeline(pipeline: rs.pipeline, tries: int = 15) -> None:
    last_err = None
    for _ in range(tries):
        try:
            wait_frames(pipeline)
            return
        except RuntimeError as e:
            last_err = e
            time.sleep(0.3)
    raise RuntimeError(
        "RealSense에서 프레임을 받지 못했습니다.\n"
        "- USB 케이블 재연결, 다른 USB3 포트 사용\n"
        "- realsense-viewer 등 다른 프로그램 종료\n"
        "- rs-enumerate-devices 로 장치 인식 확인"
    ) from last_err


def get_color_name(h: int, s: int, v: int) -> str:
    if v < 20:
        return "?"
    if s < 52 and v > 88:
        return "W"

    def classify_warm() -> Optional[str]:
        # 노랑: 주황보다 먼저 체크 (H 21~22가 노랑으로 들어오는 경우 보정)
        if 18 <= h <= 38 and s >= 42 and v >= 50:
            return "Y"
        # 주황: Hue 5~17
        if 5 <= h <= 17 and s >= 50 and v >= 50:
            return "O"
        # 빨강: Hue 양 끝 (0~4, 165~180)
        if (h <= 4 or h >= 165) and s >= 40:
            return "R"
        # warm 영역 fallback
        if (h <= 6 or h >= 160) and s >= 30:
            return "R"
        return None

    warm = classify_warm()
    if warm is not None:
        return warm

    if 40 <= h < 90 and s >= 50:
        return "G"
    if 100 <= h < 130 and s >= 50:
        return "B"
    # 마지막 fallback: warm 계열인데 채도가 낮아 누락된 경우
    if (h <= 15 or h >= 150) and s >= 25 and v >= 30:
        return "R"
    return "?"


def sample_mean_hsv(hsv_img: np.ndarray, pt: tuple[int, int], half_window: int = 5):
    x, y = pt
    h_img, w_img = hsv_img.shape[:2]
    x1 = max(0, x - half_window)
    y1 = max(0, y - half_window)
    x2 = min(w_img, x + half_window + 1)
    y2 = min(h_img, y + half_window + 1)
    roi = hsv_img[y1:y2, x1:x2]
    h, s, v = cv2.mean(roi)[:3]
    return int(h), int(s), int(v)


def reorder_du_face(face_data: str) -> str:
    if len(face_data) != 9:
        return face_data
    return "".join(face_data[i] for i in DU_REORDER_INDEX)


def validate_urfdlb(state: str) -> tuple[bool, str]:
    if len(state) != 54:
        return False, f"길이 오류: {len(state)} (기대 54)"
    if any(c not in VALID_FACES for c in state):
        return False, "허용되지 않은 문자 포함(URFDLB 이외)"

    counts = Counter(state)
    bad = [f"{f}:{counts.get(f, 0)}" for f in VALID_FACES if counts.get(f, 0) != 9]
    if bad:
        return False, "면별 개수 불일치(" + ", ".join(bad) + ")"
    return True, "ok"


def validate_solution_string(solution: str) -> tuple[bool, str]:
    tokens = [t for t in solution.split() if t]
    if not tokens:
        return False, "빈 문자열입니다."
    for token in tokens:
        face = token[0]
        suffix = token[1:]
        if face not in VALID_MOVE_FACES:
            return False, f"지원하지 않는 face: {face}"
        if suffix not in VALID_MOVE_SUFFIX:
            return False, f"지원하지 않는 suffix: {suffix} (token={token})"
    return True, "ok"


class CubePerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("cube_perception")

        self.declare_parameter("grid_start_x", 130)
        self.declare_parameter("grid_start_y", 130)
        self.declare_parameter("grid_gap", 75)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("manual_mode", False)
        self.declare_parameter("face_order", ["R", "B", "L", "F", "D", "U"])
        self.declare_parameter("capture_save_dir", "/home/junsoo/cube_solver_ws/capture")
        self.declare_parameter("publish_solution_after_scan", False)

        sx = int(self.get_parameter("grid_start_x").value)
        sy = int(self.get_parameter("grid_start_y").value)
        gap = int(self.get_parameter("grid_gap").value)
        hz = float(self.get_parameter("publish_hz").value)
        manual_mode = bool(self.get_parameter("manual_mode").value)
        faces = [str(x) for x in self.get_parameter("face_order").value]
        capture_save_dir = str(self.get_parameter("capture_save_dir").value)
        publish_solution_after_scan = bool(
            self.get_parameter("publish_solution_after_scan").value
        )

        self._grid_points: list[tuple[int, int]] = []
        for i in range(3):
            for j in range(3):
                self._grid_points.append((sx + j * gap, sy + i * gap))

        self._faces = faces
        self._manual_mode = manual_mode
        self._lock = threading.Lock()
        self._busy = False
        self._state_urfdlb = ""
        self._state_color = ""
        self._solution = ""
        self._solution_pending = False
        self._capture_index = 0
        self._captured_faces: dict[str, str] = {}
        self._capture_save_dir = Path(capture_save_dir).expanduser()
        self._capture_save_dir.mkdir(parents=True, exist_ok=True)
        self._publish_solution_after_scan = publish_solution_after_scan

        self._pub_state = self.create_publisher(String, "/cube/state_raw", 10)
        self._pub_solution = self.create_publisher(String, "/cube/solution", 10)
        self._pub_color = self.create_publisher(String, "/cube/state_color", 10)

        period = 1.0 / hz if hz > 0 else 0.2
        self.create_timer(period, self._publish_cached)
        self.create_service(Trigger, "/cube/scan_and_solve", self._srv_scan_and_solve)
        self.create_service(Trigger, "/cube/capture_once", self._srv_capture_once)
        self.create_service(Trigger, "/cube/publish_solution_once", self._srv_publish_solution_once)
        self.create_subscription(String, "/cube/solution_manual", self._sub_solution_manual, 10)

        self.get_logger().info(
            "cube_perception 준비 완료. "
            "캡처 서비스: /cube/scan_and_solve, /cube/capture_once, /cube/publish_solution_once, "
            "수동 입력 토픽: /cube/solution_manual"
        )
        if manual_mode:
            self.get_logger().info(
                "manual_mode=true: 카메라 스캔을 건너뜁니다. "
                "터미널 입력 또는 /cube/solution_manual 로 솔루션을 넣으세요."
            )
            threading.Thread(target=self._manual_input_loop, daemon=True).start()
        elif bool(self.get_parameter("auto_start").value):
            self._start_scan_thread("startup")

    def _publish_cached(self) -> None:
        with self._lock:
            state = self._state_urfdlb
            sol = self._solution
            color = self._state_color
            publish_solution_once = self._solution_pending
            if publish_solution_once:
                self._solution_pending = False

        msg_state = String()
        msg_state.data = state
        self._pub_state.publish(msg_state)

        if publish_solution_once:
            msg_sol = String()
            msg_sol.data = sol
            self._pub_solution.publish(msg_sol)

        msg_color = String()
        msg_color.data = color
        self._pub_color.publish(msg_color)

    def _start_scan_thread(self, reason: str) -> None:
        t = threading.Thread(
            target=self._scan_and_solve_once,
            args=(reason,),
            daemon=True,
        )
        t.start()

    def _manual_input_loop(self) -> None:
        while rclpy.ok():
            try:
                raw = input("[cube_perception manual] solution 입력 (예: R U R' U', quit 종료): ").strip()
            except EOFError:
                return
            except Exception:
                return

            if not raw:
                continue
            if raw.lower() in {"quit", "exit", "q"}:
                self.get_logger().info("manual_mode 입력 루프를 종료합니다.")
                return
            self._apply_manual_solution(raw, source="stdin")

    def _sub_solution_manual(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self._apply_manual_solution(text, source="/cube/solution_manual")

    def _apply_manual_solution(self, solution: str, source: str) -> None:
        normalized = " ".join(solution.split())
        ok, reason = validate_solution_string(normalized)
        if not ok:
            self.get_logger().warn(f"[{source}] 잘못된 솔루션: {reason}")
            return
        with self._lock:
            self._solution = normalized
            self._solution_pending = True
        self.get_logger().info(f"[{source}] 수동 솔루션 반영: {normalized}")

    def _capture_face_data(self, pipeline: rs.pipeline) -> tuple[str, np.ndarray]:
        samples: list[str] = []
        images: list[np.ndarray] = []
        for _ in range(5):
            frames = wait_frames(pipeline)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            face_data = ""
            for pt in self._grid_points:
                h, s, v = sample_mean_hsv(hsv_img, pt, half_window=5)
                face_data += get_color_name(h, s, v)
            if len(face_data) == 9:
                samples.append(face_data)
                images.append(img)

        if not samples:
            raise RuntimeError("유효한 컬러 프레임을 얻지 못했습니다.")

        voted = ""
        for i in range(9):
            col = Counter(s[i] for s in samples if len(s) == 9)
            voted += col.most_common(1)[0][0]
        return voted, images[-1]

    def _save_capture_record(self, face: str, face_data: str, img: np.ndarray) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seq = self._capture_index + 1
        image_path = self._capture_save_dir / f"{timestamp}_{seq}_{face}.png"
        log_path = self._capture_save_dir / "captures.log"

        vis = img.copy()
        for idx, pt in enumerate(self._grid_points):
            code = face_data[idx] if idx < len(face_data) else "?"
            cv2.circle(vis, pt, 5, (0, 255, 0), -1)
            cv2.putText(
                vis,
                code,
                (pt[0] + 8, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        cv2.imwrite(str(image_path), vis)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{timestamp},index={seq},face={face},data={face_data},image={image_path}\n"
            )

    def _finalize_captured_faces(self) -> tuple[bool, str]:
        total_color_capture_order = "".join(self._captured_faces[f] for f in self._faces)
        total_color_solver_order = "".join(self._captured_faces[f] for f in SOLVER_FACE_ORDER)
        center_color_to_face = {self._captured_faces[f][4]: f for f in self._faces}

        if len(center_color_to_face) != 6 or "?" in total_color_capture_order:
            with self._lock:
                self._state_color = total_color_capture_order
                self._state_urfdlb = ""
                self._solution = ""
            return False, "URFDLB 변환 실패(센터 중복 또는 '?' 포함)"

        urfdlb = "".join(center_color_to_face[c] for c in total_color_solver_order)
        valid, reason_msg = validate_urfdlb(urfdlb)
        if not valid:
            with self._lock:
                self._state_color = total_color_capture_order
                self._state_urfdlb = urfdlb
                self._solution = ""
            return False, f"큐브 상태 검증 실패: {reason_msg}"

        solution = ""
        if kociemba is None:
            self.get_logger().warn(
                "kociemba 모듈이 없어 풀이를 생략합니다. (state_raw만 발행)"
            )
        else:
            try:
                solution = str(kociemba.solve(urfdlb)).strip()
            except Exception as e:
                with self._lock:
                    self._state_color = total_color_capture_order
                    self._state_urfdlb = urfdlb
                    self._solution = ""
                return False, f"kociemba.solve 실패: {e}"

        with self._lock:
            self._state_color = total_color_capture_order
            self._state_urfdlb = urfdlb
            self._solution = solution
            self._solution_pending = self._publish_solution_after_scan
        self.get_logger().info(f"state_color : {total_color_capture_order}")
        self.get_logger().info(f"state_raw   : {urfdlb}")
        if solution:
            self.get_logger().info(f"solution    : {solution}")
        return True, "자동 캡처/변환/풀이 완료"

    def _srv_capture_once(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        with self._lock:
            if self._busy:
                res.success = False
                res.message = "이미 캡처/풀이 진행 중입니다."
                return res
            self._busy = True

        try:
            if self._capture_index >= len(self._faces):
                self._capture_index = 0
                self._captured_faces = {}

            face = self._faces[self._capture_index]
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            try:
                pipeline.start(config)
                warmup_pipeline(pipeline)
                face_data, capture_img = self._capture_face_data(pipeline)
            finally:
                try:
                    pipeline.stop()
                except Exception:
                    pass

            if face in {"D", "U"}:
                face_data = reorder_du_face(face_data)

            self._captured_faces[face] = face_data
            try:
                self._save_capture_record(face, face_data, capture_img)
            except Exception as e:
                self.get_logger().warn(f"capture 기록 저장 실패: {e}")
            self._capture_index += 1
            self.get_logger().info(f"[capture_once] {face}면(색): {face_data}")

            if self._capture_index < len(self._faces):
                res.success = True
                res.message = f"{face} 캡처 완료 ({self._capture_index}/{len(self._faces)})"
                return res

            ok, msg = self._finalize_captured_faces()
            self._capture_index = 0
            self._captured_faces = {}
            res.success = ok
            res.message = msg
            return res
        except Exception as e:
            res.success = False
            res.message = f"capture_once 예외: {e}"
            return res
        finally:
            with self._lock:
                self._busy = False

    def _srv_scan_and_solve(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        ok, msg = self._scan_and_solve_once("service")
        res.success = ok
        res.message = msg
        return res

    def _srv_publish_solution_once(
        self, _req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        with self._lock:
            if not self._solution:
                res.success = False
                res.message = "발행할 solution이 없습니다."
                return res
            self._solution_pending = True
            value = self._solution
        res.success = True
        res.message = f"solution 1회 발행 예약: {value}"
        return res

    def _scan_and_solve_once(self, reason: str) -> tuple[bool, str]:
        with self._lock:
            if self._busy:
                return False, "이미 캡처/풀이 진행 중입니다."
            self._busy = True

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        text_bgr = {
            "W": (240, 240, 240),
            "Y": (0, 255, 255),
            "R": (0, 0, 255),
            "O": (0, 165, 255),
            "G": (0, 255, 0),
            "B": (255, 0, 0),
            "?": (180, 180, 180),
        }

        try:
            self.get_logger().info(
                f"[{reason}] 스캔 시작: "
                f"{'->'.join(self._faces)} 순서로 맞추고 's' 저장, 'q' 취소"
            )
            pipeline.start(config)
            warmup_pipeline(pipeline)

            captured_faces: dict[str, str] = {}
            center_color_to_face: dict[str, str] = {}

            for face in self._faces:
                self.get_logger().info(f"[{face}] 면 준비 후 's'")
                while True:
                    try:
                        frames = wait_frames(pipeline)
                    except RuntimeError as e:
                        self.get_logger().warn(f"프레임 오류: {e}. 재시도합니다.")
                        time.sleep(1.0)
                        continue

                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    img = np.asanyarray(color_frame.get_data())
                    display_img = img.copy()
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    current_face_data = ""
                    for pt in self._grid_points:
                        cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
                        h, s, v = sample_mean_hsv(hsv_img, pt, half_window=5)
                        color_code = get_color_name(h, s, v)
                        current_face_data += color_code
                        cv2.putText(
                            display_img,
                            color_code,
                            (pt[0] + 10, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            text_bgr.get(color_code, (255, 255, 255)),
                            2,
                        )

                    cv2.putText(
                        display_img,
                        "Color map: W Y R O G B | 's' capture | 'q' cancel",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("RealSense Cube Scanner (ROS2)", display_img)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        if face in {"D", "U"}:
                            current_face_data = reorder_du_face(current_face_data)
                        captured_faces[face] = current_face_data
                        center_color = current_face_data[4]
                        center_color_to_face[center_color] = face
                        self.get_logger().info(f"{face}면(색): {current_face_data}")
                        break
                    if key == ord("q"):
                        return False, "사용자 취소(q)"

            total_color_capture_order = "".join(captured_faces[f] for f in self._faces)
            total_color_solver_order = "".join(captured_faces[f] for f in SOLVER_FACE_ORDER)

            if len(center_color_to_face) != 6 or "?" in total_color_capture_order:
                with self._lock:
                    self._state_color = total_color_capture_order
                    self._state_urfdlb = ""
                    self._solution = ""
                return False, "URFDLB 변환 실패(센터 중복 또는 '?' 포함)"

            urfdlb = "".join(center_color_to_face[c] for c in total_color_solver_order)
            valid, reason_msg = validate_urfdlb(urfdlb)
            if not valid:
                with self._lock:
                    self._state_color = total_color_capture_order
                    self._state_urfdlb = urfdlb
                    self._solution = ""
                return False, f"큐브 상태 검증 실패: {reason_msg}"

            solution = ""
            if kociemba is None:
                self.get_logger().warn(
                    "kociemba 모듈이 없어 풀이를 생략합니다. (state_raw만 발행)"
                )
            else:
                try:
                    solution = str(kociemba.solve(urfdlb)).strip()
                except Exception as e:
                    with self._lock:
                        self._state_color = total_color_capture_order
                        self._state_urfdlb = urfdlb
                        self._solution = ""
                    return False, f"kociemba.solve 실패: {e}"

            with self._lock:
                self._state_color = total_color_capture_order
                self._state_urfdlb = urfdlb
                self._solution = solution
                self._solution_pending = self._publish_solution_after_scan

            self.get_logger().info(f"state_color : {total_color_capture_order}")
            self.get_logger().info(f"state_raw   : {urfdlb}")
            if solution:
                self.get_logger().info(f"solution    : {solution}")
            return True, "캡처/변환/풀이 완료"
        except Exception as e:
            return False, f"예외 발생: {e}"
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()
            with self._lock:
                self._busy = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubePerceptionNode()
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
