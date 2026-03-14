from ultralytics import YOLO
import argparse
import supervision as sv
import cv2
import numpy as np
import math
import time
import threading
import logging
import json
import requests
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sys
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

# Add the project root to sys.path so we can import 'detection_services'
# even when running the script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from detection_services.llm.gemini_analyzer import GeminiAnalyzer
from detection_services.schemas.alert_schema import DetectionAlert
from newservice.detection_service.client import CallingServiceClient

load_dotenv()

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "surveillance.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class Config:
    """All tuneable parameters in one place."""

    # Model
    model_path: str = "yolo26n.pt"
    camera_index: int = 1
    calling_service_base_url: str = os.getenv(
        "NEW_CALLING_SERVICE_BASE_URL",
        "https://untrusting-oxymoronically-annita.ngrok-free.dev",
    ).strip()
    alert_phone_number: str = os.getenv("ALERT_PHONE_NUMBER", "+917011072161").strip()

    # Accident detection
    accident_distance: int = 100
    sudden_stop_threshold: float = 3.0
    closing_speed_threshold: float = 8.0
    iou_overlap_threshold: float = 0.15
    accident_cooldown_frames: int = 90  # ~3 s at 30 fps
    pair_clear_distance: int = 200  # vehicles must separate this far to re-trigger

    # Suspicious surround detection
    surround_distance: int = 150
    min_people_for_surround: int = 3
    min_approaching_for_surround: int = 2
    angular_spread_threshold: float = math.pi  # 180° in radians

    # Performance
    detect_every_n_frames: int = 1  # set to 2 or 3 to skip frames

    # Frame buffer
    frame_buffer_size: int = 5

    # Clip recording
    clip_seconds_before: int = 3
    clip_seconds_after: int = 3
    clip_fps: int = 15

    # Track history lengths
    track_history_len: int = 10
    velocity_history_len: int = 5


# ──────────────────────────────────────────────
# Event Logger — persists events to JSON lines
# ──────────────────────────────────────────────
class EventLogger:
    """Append-only JSON-lines logger for detected events."""

    def __init__(self, path: Path = LOG_DIR / "events.jsonl"):
        self.path = path

    def log(self, event_type: str, details: dict | None = None):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "details": details or {},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Event logged: %s", event_type)


# ──────────────────────────────────────────────
# Surveillance System
# ──────────────────────────────────────────────
class SurveillanceSystem:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()

        # Model & tracker
        self.model = YOLO(self.cfg.model_path)
        self.tracker = sv.ByteTrack()

        # Camera
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.cfg.camera_index}"
            )

        # Gemini Analyzer
        self.llm_analyzer = GeminiAnalyzer()
        self.calling_service_client = CallingServiceClient(
            base_url=self.cfg.calling_service_base_url,
            timeout=5.0,
        )

        # Memory stores
        self.track_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.track_history_len)
        )
        self.velocity_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.velocity_history_len)
        )
        self.prev_distances: dict[str, float] = {}

        # Frame buffer for clip recording & LLM verification
        buf_len = max(
            self.cfg.frame_buffer_size,
            self.cfg.clip_seconds_before * self.cfg.clip_fps,
        )
        self.frame_buffer: deque = deque(maxlen=buf_len)

        # Accident state
        self.accident_detected_flag = False
        self.frames_after_accident = 0
        self.last_accident_frame = -self.cfg.accident_cooldown_frames
        self.frame_count = 0

        # Per-pair deduplication: tracks which vehicle pairs already triggered
        # Each entry is a frozenset({id1, id2})
        self.reported_pairs: set[frozenset] = set()

        # LLM verification threading
        self.llm_lock = threading.Lock()
        self.llm_busy = False
        self.last_llm_result: DetectionAlert | None = None

        # Event logger
        self.event_logger = EventLogger()

        # FPS tracking
        self._fps_time = time.time()
        self._fps_count = 0
        self._fps_display = 0.0

        # Clip output directory
        self.clip_dir = Path("clips")
        self.clip_dir.mkdir(exist_ok=True)

    # ── Velocity & Acceleration ─────────────────

    def compute_velocity(self, track_id: int, center: tuple) -> float:
        history = self.track_history[track_id]
        history.append(center)

        if len(history) < 2:
            return 0.0

        p1 = np.array(history[-1])
        p2 = np.array(history[-2])
        dist = float(np.linalg.norm(p1 - p2))

        self.velocity_history[track_id].append(dist)
        return dist

    def compute_acceleration(self, track_id: int) -> float:
        """Return acceleration (change in velocity). Negative = decelerating."""
        vhist = self.velocity_history[track_id]
        if len(vhist) < 2:
            return 0.0
        return vhist[-1] - vhist[-2]

    def compute_closing_speed(self, id1: int, id2: int) -> float:
        """Positive = approaching each other, negative = separating."""
        h1 = self.track_history[id1]
        h2 = self.track_history[id2]
        if len(h1) < 2 or len(h2) < 2:
            return 0.0
        prev_dist = float(np.linalg.norm(np.array(h1[-2]) - np.array(h2[-2])))
        curr_dist = float(np.linalg.norm(np.array(h1[-1]) - np.array(h2[-1])))
        return prev_dist - curr_dist

    @staticmethod
    def compute_iou(box1, box2) -> float:
        """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    # ── LLM Verification ────────────────────────

    def verify_accident_with_llm(self, frame: np.ndarray):
        """Send a frame to Gemini for accident verification and routing (called in thread)."""
        if not self.llm_analyzer.client:
            logger.warning("LLM client not available — skipping verification")
            return

        try:
            prompt = (
                "A vehicle accident was just detected by our heuristic tracking system. "
                "Analyze this frame to confirm. If there is a clear emergency, map the details "
                "to the correct government department, priority, and category according to the schema. "
                "Look for: crashes, traffic disruptions, sudden stops, or debris."
            )

            result = self.llm_analyzer.analyze_frame(frame, prompt)
            
            if result:
                self.last_llm_result = result
                
                # Log the structured prediction
                self.event_logger.log(
                    "llm_verification",
                    result.model_dump()
                )
                logger.info("LLM Result: %s", result.model_dump_json(indent=2))

                if result.is_valid and result.message:
                    self._trigger_voice_alert(result.message)

        except Exception as e:
            logger.error("LLM verification failed: %s", e)
        finally:
            with self.llm_lock:
                self.llm_busy = False

    def _trigger_voice_alert(self, message: str):
        """Send the LLM-generated spoken alert to the hosted calling service."""
        if not self.cfg.alert_phone_number:
            logger.error("ALERT_PHONE_NUMBER is not configured; skipping voice alert")
            return

        try:
            logger.info(
                "Triggering voice alert to %s via %s",
                self.cfg.alert_phone_number,
                self.cfg.calling_service_base_url,
            )
            response = self.calling_service_client.send_broadcast_message(
                message=message,
                number=self.cfg.alert_phone_number,
            )
            logger.info("Voice alert triggered successfully: %s", response)
        except Exception as e:
            logger.error("Failed to trigger voice alert: %s", e)

    def _trigger_llm_verification(self, frame: np.ndarray):
        """Launch LLM verification in a background thread (non-blocking)."""
        with self.llm_lock:
            if self.llm_busy:
                logger.info("LLM already busy — skipping this verification")
                return
            self.llm_busy = True

        thread = threading.Thread(
            target=self.verify_accident_with_llm,
            args=(frame,),
            daemon=True,
        )
        thread.start()

    # ── Clip Saver ───────────────────────────────

    def save_clip(self, extra_frames: list | None = None):
        """Save a short video clip around the current moment."""
        frames = list(self.frame_buffer)
        if extra_frames:
            frames.extend(extra_frames)
        if not frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = self.clip_dir / f"event_{timestamp}.mp4"
        h, w = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.cfg.clip_fps,
            (w, h),
        )
        for f in frames:
            writer.write(f)
        writer.release()

        logger.info("Clip saved: %s (%d frames)", clip_path, len(frames))

    # ── Accident Detection ───────────────────────

    def _clear_separated_pairs(self, vehicle_data: dict):
        """Remove pairs from reported_pairs once vehicles separate enough."""
        to_remove = set()
        for pair in self.reported_pairs:
            id1, id2 = tuple(pair)
            # If either vehicle left the scene, clear the pair
            if id1 not in vehicle_data or id2 not in vehicle_data:
                to_remove.add(pair)
                continue
            c1 = np.array(vehicle_data[id1]["center"])
            c2 = np.array(vehicle_data[id2]["center"])
            dist = float(np.linalg.norm(c1 - c2))
            if dist > self.cfg.pair_clear_distance:
                to_remove.add(pair)
        for pair in to_remove:
            self.reported_pairs.discard(pair)
            logger.debug("Cleared reported pair %s (separated)", pair)

    def detect_accident(self, vehicle_data: dict) -> tuple[bool, frozenset | None]:
        """Returns (accident_flag, offending_pair) or (False, None)."""
        ids = list(vehicle_data.keys())
        cfg = self.cfg

        # First, clear any pairs that have separated
        self._clear_separated_pairs(vehicle_data)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair = frozenset({id1, id2})

                # Skip if this exact pair was already reported
                if pair in self.reported_pairs:
                    continue

                v1_info = vehicle_data[id1]
                v2_info = vehicle_data[id2]

                c1, c2 = v1_info["center"], v2_info["center"]
                dist = float(np.linalg.norm(np.array(c1) - np.array(c2)))

                if dist >= cfg.accident_distance:
                    continue

                # ── Signal 1: Bounding box overlap (IoU) ──
                iou = self.compute_iou(v1_info["bbox"], v2_info["bbox"])
                if iou > cfg.iou_overlap_threshold:
                    logger.debug("IoU overlap %.2f between %s and %s", iou, id1, id2)
                    return True, pair

                # ── Signal 2: High closing speed ──
                closing = self.compute_closing_speed(id1, id2)
                if closing > cfg.closing_speed_threshold:
                    logger.debug("High closing speed %.1f between %s and %s", closing, id1, id2)
                    return True, pair

                # ── Signal 3: Both suddenly stopped after movement ──
                v1, v2 = v1_info["velocity"], v2_info["velocity"]
                a1 = self.compute_acceleration(id1)
                a2 = self.compute_acceleration(id2)

                if (
                    v1 < cfg.sudden_stop_threshold
                    and v2 < cfg.sudden_stop_threshold
                    and (a1 < -2 or a2 < -2)  # significant deceleration
                ):
                    logger.debug("Sudden stop detected: %s and %s", id1, id2)
                    return True, pair

        return False, None

    # ── Suspicious Surround Detection ────────────

    def detect_surrounding(self, person_data: dict) -> bool:
        ids = list(person_data.keys())
        cfg = self.cfg

        if len(ids) < cfg.min_people_for_surround:
            return False

        for i in range(len(ids)):
            target_id = ids[i]
            target = np.array(person_data[target_id]["center"])

            approaching_angles: list[float] = []

            for j in range(len(ids)):
                if i == j:
                    continue

                other_id = ids[j]
                other = np.array(person_data[other_id]["center"])

                curr_dist = float(np.linalg.norm(target - other))
                key = f"{target_id}_{other_id}"

                if key in self.prev_distances:
                    prev_dist = self.prev_distances[key]

                    if curr_dist < prev_dist and curr_dist < cfg.surround_distance:
                        # This person is approaching — record the angle
                        angle = math.atan2(
                            other[1] - target[1], other[0] - target[0]
                        )
                        approaching_angles.append(angle)

                self.prev_distances[key] = curr_dist

            # Check: enough people approaching AND from spread-out directions
            if len(approaching_angles) >= cfg.min_approaching_for_surround:
                if self._angular_spread(approaching_angles) >= cfg.angular_spread_threshold:
                    return True

        return False

    @staticmethod
    def _angular_spread(angles: list[float]) -> float:
        """Max angular gap between sorted angles — large spread = surrounding."""
        if len(angles) < 2:
            return 0.0
        sorted_a = sorted(angles)
        max_gap = 0.0
        for k in range(len(sorted_a) - 1):
            max_gap = max(max_gap, sorted_a[k + 1] - sorted_a[k])
        # Also check wrap-around gap
        wrap_gap = (2 * math.pi) - (sorted_a[-1] - sorted_a[0])
        max_gap = max(max_gap, wrap_gap)
        # The "spread" is the arc NOT covered by the largest gap
        return (2 * math.pi) - max_gap

    # ── FPS Counter ──────────────────────────────

    def _update_fps(self):
        self._fps_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._fps_display = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_time = time.time()

    # ── Main Loop ────────────────────────────────

    def run(self):
        logger.info("Starting AI Surveillance System...")
        cfg = self.cfg

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame — exiting")
                    break

                self.frame_count += 1
                self.frame_buffer.append(frame.copy())
                self._update_fps()

                # ── Optional frame skipping ──
                if self.frame_count % cfg.detect_every_n_frames != 0:
                    # Still show the last annotated frame if skipping
                    cv2.imshow("AI Surveillance System", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                # ── YOLO detection + tracking ──
                results = self.model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = self.tracker.update_with_detections(detections)

                annotated = frame.copy()
                vehicle_data = {}
                person_data = {}

                for i in range(len(detections.xyxy)):
                    bbox = detections.xyxy[i]
                    x1, y1, x2, y2 = bbox
                    class_id = detections.class_id[i]
                    track_id = detections.tracker_id[i]
                    label = self.model.names[class_id]

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    velocity = self.compute_velocity(track_id, (center_x, center_y))

                    # Vehicles
                    if label in ("car", "truck", "bus", "motorcycle"):
                        vehicle_data[track_id] = {
                            "center": (center_x, center_y),
                            "velocity": velocity,
                            "bbox": (x1, y1, x2, y2),
                        }

                    # People
                    if label == "person":
                        person_data[track_id] = {
                            "center": (center_x, center_y),
                        }

                    # Draw bounding box
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    text = f"{label} ID:{track_id}"
                    cv2.putText(
                        annotated,
                        text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # ── Event Detection ──
                accident, accident_pair = self.detect_accident(vehicle_data)
                suspicious = self.detect_surrounding(person_data)

                # ── Accident handling with per-pair dedup + cooldown ──
                in_cooldown = (
                    self.frame_count - self.last_accident_frame
                ) < cfg.accident_cooldown_frames

                if accident and accident_pair is not None and not in_cooldown and not self.accident_detected_flag:
                    # Mark this pair as reported so it won't re-trigger
                    self.reported_pairs.add(accident_pair)
                    pair_ids = tuple(accident_pair)

                    logger.info(
                        "ACCIDENT DETECTED between %s — capturing for LLM",
                        pair_ids,
                    )
                    self.accident_detected_flag = True
                    self.frames_after_accident = 0
                    self.last_accident_frame = self.frame_count

                    self.event_logger.log(
                        "accident_heuristic",
                        {
                            "vehicle_pair": [int(x) for x in pair_ids],
                            "vehicle_count": len(vehicle_data),
                        },
                    )

                if self.accident_detected_flag:
                    self.frames_after_accident += 1
                    if self.frames_after_accident == 2:
                        trigger_frame = annotated.copy()
                        logger.info("Sending detection frame to LLM...")
                        self._trigger_llm_verification(trigger_frame)
                        self.save_clip()

                        self.accident_detected_flag = False
                        self.frames_after_accident = 0

                if accident and not in_cooldown:
                    cv2.putText(
                        annotated,
                        "ACCIDENT DETECTED",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

                if suspicious:
                    self.event_logger.log(
                        "suspicious_surround",
                        {"person_count": len(person_data)},
                    )
                    cv2.putText(
                        annotated,
                        "SUSPICIOUS SURROUND",
                        (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3,
                    )

                # ── Show LLM result on screen if available ──
                if self.last_llm_result is not None:
                    color = (0, 0, 255) if self.last_llm_result.is_valid else (0, 255, 0)
                    llm_text = (
                        f"LLM: {'CONFIRMED EMERGENCY' if self.last_llm_result.is_valid else 'NORMAL / FALSE ALARM'} "
                    )
                    cv2.putText(
                        annotated, llm_text, (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )

                # ── FPS overlay ──
                cv2.putText(
                    annotated,
                    f"FPS: {self._fps_display:.1f}",
                    (annotated.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("AI Surveillance System", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Surveillance system stopped")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accident and suspicious activity detection service")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument(
        "--calling-service-base-url",
        "--calling-agent-base-url",
        dest="calling_service_base_url",
        default=os.getenv(
            "NEW_CALLING_SERVICE_BASE_URL",
            "https://untrusting-oxymoronically-annita.ngrok-free.dev",
        ),
        help="Hosted calling-service base URL. Detector sends only number and message.",
    )
    parser.add_argument(
        "--alert-number",
        default=os.getenv("ALERT_PHONE_NUMBER", "+917011072161"),
        help="Phone number that should receive the outbound alert call.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(
        camera_index=args.camera_index,
        calling_service_base_url=args.calling_service_base_url,
        alert_phone_number=args.alert_number,
    )
    system = SurveillanceSystem(config)
    system.run()
