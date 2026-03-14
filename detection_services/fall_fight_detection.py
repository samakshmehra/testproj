from ultralytics import YOLO
import argparse
import cv2
import numpy as np
import math
import time
import threading
import logging
import json
import requests
from dataclasses import dataclass
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import os
import urllib3
from urllib.parse import urlparse
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
        logging.FileHandler(LOG_DIR / "pose_surveillance.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class PoseConfig:
    """All tuneable parameters in one place."""

    # Model
    model_path: str = "yolo26n-pose.pt"
    camera_index: int = 1
    calling_service_base_url: str = os.getenv(
        "NEW_CALLING_SERVICE_BASE_URL",
        "https://untrusting-oxymoronically-annita.ngrok-free.dev",
    ).strip()
    alert_phone_number: str = os.getenv("ALERT_PHONE_NUMBER", "7982373129").strip()
    verify_ssl: bool = os.getenv("DETECTION_VERIFY_SSL", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # Fight detection
    fight_proximity: float = 150.0       # px — how close people need to be
    fight_movement_threshold: float = 120.0  # total wrist movement to flag a fight
    fight_window_frames: int = 5         # sliding window of wrist history for smoothing

    # Fall detection
    fall_angle_threshold: float = 30.0   # degrees — torso angle below this = "fallen"
    fall_confirmation_frames: int = 3    # must stay fallen for N frames to confirm
    min_keypoint_confidence: float = 0.3 # ignore low-confidence keypoints

    # Deduplication / cooldown
    fight_cooldown_frames: int = 90      # ~3 s at 30 fps
    fall_cooldown_frames: int = 90
    pair_clear_distance: float = 250.0   # people must separate this far to re-trigger fight

    # Performance
    detect_every_n_frames: int = 1

    # Frame buffer & clips
    frame_buffer_size: int = 45          # ~1.5 s at 30 fps
    clip_fps: int = 15

    # Keypoint indices (COCO pose)
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16
    NOSE: int = 0


# ──────────────────────────────────────────────
# Event Logger
# ──────────────────────────────────────────────
class EventLogger:
    """Append-only JSON-lines logger for detected events."""

    def __init__(self, path: Path = LOG_DIR / "pose_events.jsonl"):
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
# Pose Surveillance System
# ──────────────────────────────────────────────
class PoseSurveillanceSystem:
    def __init__(self, config: PoseConfig | None = None):
        self.cfg = config or PoseConfig()

        # Model
        self.model = YOLO(self.cfg.model_path)

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
            timeout=40.0,
        )

        # ── Per-person wrist history (keyed by person index in frame) ──
        # For proper tracking we use a sliding window of wrist positions
        self.wrist_history: deque[list[np.ndarray]] = deque(
            maxlen=self.cfg.fight_window_frames
        )

        # ── Fall confirmation: person_index → consecutive fallen frame count ──
        self.fall_frame_counts: defaultdict[int, int] = defaultdict(int)

        # ── Deduplication ──
        # Fight: set of frozenset({person_i, person_j}) pairs already reported
        self.reported_fight_pairs: set[frozenset] = set()
        self.last_fight_frame: int = -self.cfg.fight_cooldown_frames

        # Fall: set of person indices already reported as fallen
        self.reported_falls: set[int] = set()
        self.last_fall_frame: int = -self.cfg.fall_cooldown_frames

        # Frame state
        self.frame_count: int = 0
        self.frame_buffer: deque = deque(maxlen=self.cfg.frame_buffer_size)

        # LLM threading
        self.llm_lock = threading.Lock()
        self.llm_busy = False
        self.last_llm_result: DetectionAlert | None = None

        # Event logger
        self.event_logger = EventLogger()

        # FPS tracking
        self._fps_time = time.time()
        self._fps_count = 0
        self._fps_display = 0.0

        # Clip output
        self.clip_dir = Path("clips")
        self.clip_dir.mkdir(exist_ok=True)

    # ── Fight Detection ──────────────────────────

    def detect_fight(
        self, wrists: list[np.ndarray], person_centers: list[list[float]]
    ) -> tuple[bool, frozenset | None]:
        """
        Detect a fight between two close people with rapid arm movement.
        Returns (flag, offending_pair) or (False, None).
        """
        cfg = self.cfg

        if len(person_centers) < 2:
            return False, None

        # Store current wrists in sliding window
        self.wrist_history.append(wrists)

        # ── Step 1: Find close pairs (that haven't been reported) ──
        close_pairs: list[tuple[int, int, float]] = []
        for i in range(len(person_centers)):
            for j in range(i + 1, len(person_centers)):
                pair = frozenset({i, j})
                if pair in self.reported_fight_pairs:
                    continue
                dist = float(np.linalg.norm(
                    np.array(person_centers[i]) - np.array(person_centers[j])
                ))
                if dist < cfg.fight_proximity:
                    close_pairs.append((i, j, dist))

        if not close_pairs:
            return False, None

        # ── Step 2: Check wrist movement over the sliding window ──
        if len(self.wrist_history) < 2:
            return False, None

        # Compute total wrist movement between oldest and newest frames
        oldest_wrists = self.wrist_history[0]
        newest_wrists = self.wrist_history[-1]

        total_movement = 0.0
        count = min(len(oldest_wrists), len(newest_wrists))
        for k in range(count):
            if (
                np.any(oldest_wrists[k] == 0) or np.any(newest_wrists[k] == 0)
            ):
                continue  # skip undetected keypoints
            total_movement += float(
                np.linalg.norm(newest_wrists[k] - oldest_wrists[k])
            )

        if total_movement > cfg.fight_movement_threshold:
            # Return the closest pair as the offender
            close_pairs.sort(key=lambda x: x[2])
            p_i, p_j, _ = close_pairs[0]
            return True, frozenset({p_i, p_j})

        return False, None

    def _clear_separated_fight_pairs(self, person_centers: list[list[float]]):
        """Remove fight pairs that have separated or left the scene."""
        to_remove = set()
        for pair in self.reported_fight_pairs:
            ids = tuple(pair)
            i, j = ids[0], ids[1]
            # If either person left the scene
            if i >= len(person_centers) or j >= len(person_centers):
                to_remove.add(pair)
                continue
            dist = float(np.linalg.norm(
                np.array(person_centers[i]) - np.array(person_centers[j])
            ))
            if dist > self.cfg.pair_clear_distance:
                to_remove.add(pair)
        for pair in to_remove:
            self.reported_fight_pairs.discard(pair)
            logger.debug("Cleared fight pair %s (separated)", pair)

    # ── Fall Detection ───────────────────────────

    def detect_fall(self, keypoints: np.ndarray) -> tuple[bool, list[int]]:
        """
        Detect falls using torso angle + confirmation window.
        Returns (flag, list_of_fallen_person_indices).
        """
        cfg = self.cfg
        fallen_indices: list[int] = []

        for person_idx, person in enumerate(keypoints):
            # Skip already-reported falls
            if person_idx in self.reported_falls:
                continue

            left_shoulder = person[cfg.LEFT_SHOULDER]
            right_shoulder = person[cfg.RIGHT_SHOULDER]
            left_hip = person[cfg.LEFT_HIP]
            right_hip = person[cfg.RIGHT_HIP]
            left_ankle = person[cfg.LEFT_ANKLE]
            right_ankle = person[cfg.RIGHT_ANKLE]

            # Skip if key joints are undetected (all zeros)
            if (
                np.all(left_shoulder == 0)
                or np.all(right_shoulder == 0)
                or np.all(left_hip == 0)
                or np.all(right_hip == 0)
            ):
                self.fall_frame_counts[person_idx] = 0
                continue

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2

            dx = hip_center[0] - shoulder_center[0]
            dy = hip_center[1] - shoulder_center[1]
            torso_angle = abs(np.degrees(np.arctan2(dy, dx)))

            # ── Additional signal: are ankles near head height? ──
            # A truly fallen person has ankles at roughly the same Y as shoulders
            ankle_near_shoulder = False
            if not np.all(left_ankle == 0) and not np.all(right_ankle == 0):
                ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                shoulder_y = shoulder_center[1]
                if abs(ankle_y - shoulder_y) < 50:  # within 50px vertically
                    ankle_near_shoulder = True

            is_horizontal = torso_angle < cfg.fall_angle_threshold

            if is_horizontal:
                self.fall_frame_counts[person_idx] += 1
            else:
                self.fall_frame_counts[person_idx] = 0

            # Confirm only if fallen for N consecutive frames
            if self.fall_frame_counts[person_idx] >= cfg.fall_confirmation_frames:
                fallen_indices.append(person_idx)

        return len(fallen_indices) > 0, fallen_indices

    # ── LLM Verification ─────────────────────────

    def verify_event_with_llm(self, frame: np.ndarray, event_type: str):
        """Send a frame to Gemini for event verification and routing (runs in thread)."""
        if not self.llm_analyzer.client:
            logger.warning("LLM client not available — skipping verification")
            return

        try:
            prompt = (
                f"A possible '{event_type}' was just detected by our heuristic tracking system. "
                "Analyze this frame to confirm. If there is a clear emergency, map the details "
                "to the correct government department, priority, and category according to the schema. "
                f"For fights: look for aggressive physical contact, punching, pushing. "
                f"For falls: look for a person lying on the ground, collapsed posture. "
                f"Ignore people who are simply bending, sitting, or exercising."
            )

            result = self.llm_analyzer.analyze_frame(frame, prompt)

            if result:
                self.last_llm_result = result
                
                # Log the structured prediction
                self.event_logger.log(
                    "llm_verification",
                    {
                        "event_type": event_type,
                        **result.model_dump()
                    }
                )
                logger.info("LLM Result: %s", result.model_dump_json(indent=2))

                # Use the LLM-generated spoken message for the common calling server.
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

    def _trigger_llm_verification(self, frame: np.ndarray, event_type: str):
        """Launch LLM verification in a background thread (non-blocking)."""
        with self.llm_lock:
            if self.llm_busy:
                logger.info("LLM already busy — skipping")
                return
            self.llm_busy = True

        thread = threading.Thread(
            target=self.verify_event_with_llm,
            args=(frame, event_type),
            daemon=True,
        )
        thread.start()

    # ── Clip Saver ───────────────────────────────

    def save_clip(self, event_type: str):
        """Save a short video clip around the current moment."""
        frames = list(self.frame_buffer)
        if not frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = self.clip_dir / f"{event_type}_{timestamp}.mp4"
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
        logger.info("Starting Pose Surveillance System...")
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
                    cv2.imshow("Fight + Fall Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                # ── YOLO Pose inference ──
                results = self.model(frame, verbose=False)[0]
                annotated = results.plot()

                wrists: list[np.ndarray] = []
                person_centers: list[list[float]] = []

                if results.keypoints is not None:
                    keypoints = results.keypoints.xy.cpu().numpy()

                    for person in keypoints:
                        wrists.append(person[cfg.LEFT_WRIST])
                        wrists.append(person[cfg.RIGHT_WRIST])

                        center_x = float(np.mean(person[:, 0]))
                        center_y = float(np.mean(person[:, 1]))
                        person_centers.append([center_x, center_y])

                    # ── Clear separated fight pairs ──
                    self._clear_separated_fight_pairs(person_centers)

                    # ── Fight detection ──
                    fight_in_cooldown = (
                        self.frame_count - self.last_fight_frame
                    ) < cfg.fight_cooldown_frames

                    fight, fight_pair = self.detect_fight(wrists, person_centers)

                    if fight and fight_pair is not None and not fight_in_cooldown:
                        self.reported_fight_pairs.add(fight_pair)
                        pair_ids = tuple(fight_pair)
                        self.last_fight_frame = self.frame_count

                        logger.info("FIGHT DETECTED between persons %s", pair_ids)
                        self.event_logger.log(
                            "fight_detected",
                            {
                                "person_pair": list(pair_ids),
                                "person_count": len(person_centers),
                            },
                        )

                        # LLM verification + clip
                        trigger_frame = annotated.copy()
                        self._trigger_llm_verification(trigger_frame, "fight")
                        self.save_clip("fight")

                    # ── Fall detection ──
                    fall_in_cooldown = (
                        self.frame_count - self.last_fall_frame
                    ) < cfg.fall_cooldown_frames

                    fall, fallen_indices = self.detect_fall(keypoints)

                    if fall and not fall_in_cooldown:
                        # Only report new falls
                        new_falls = [
                            idx for idx in fallen_indices
                            if idx not in self.reported_falls
                        ]
                        if new_falls:
                            self.reported_falls.update(new_falls)
                            self.last_fall_frame = self.frame_count

                            logger.info(
                                "FALL DETECTED for person(s) %s", new_falls
                            )
                            self.event_logger.log(
                                "fall_detected",
                                {
                                    "fallen_persons": new_falls,
                                    "person_count": len(person_centers),
                                },
                            )

                            trigger_frame = annotated.copy()
                            self._trigger_llm_verification(trigger_frame, "fall")
                            self.save_clip("fall")

                    # ── Clear reported falls for people who stood back up ──
                    standing_again = set()
                    for idx in self.reported_falls:
                        if idx < len(keypoints):
                            # If their fall counter reset, they stood up
                            if self.fall_frame_counts[idx] == 0:
                                standing_again.add(idx)
                        else:
                            # Person left the scene
                            standing_again.add(idx)
                    self.reported_falls -= standing_again

                    # ── Draw alerts ──
                    if fight and not fight_in_cooldown:
                        cv2.putText(
                            annotated,
                            "FIGHT DETECTED",
                            (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )

                    if fall and not fall_in_cooldown:
                        cv2.putText(
                            annotated,
                            "FALL DETECTED",
                            (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            3,
                        )

                # ── Show LLM result on screen ──
                if self.last_llm_result is not None:
                    color = (
                        (0, 0, 255) if self.last_llm_result.is_valid
                        else (0, 255, 0)
                    )
                    llm_text = (
                        f"LLM: {'CONFIRMED EMERGENCY' if self.last_llm_result.is_valid else 'NORMAL / FALSE ALARM'}"
                    )
                    cv2.putText(
                        annotated, llm_text, (50, 180),
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

                cv2.imshow("Fight + Fall Detection", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Pose surveillance system stopped")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def _is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_startup_config(config: PoseConfig) -> None:
    """Fail fast on invalid URL wiring and print resolved runtime config."""
    if not _is_valid_http_url(config.calling_service_base_url):
        raise ValueError(
            "Invalid calling service URL. Set NEW_CALLING_SERVICE_BASE_URL "
            "to a valid http(s) URL."
        )

    if not config.alert_phone_number:
        raise ValueError("ALERT_PHONE_NUMBER is empty.")

    logger.info("Resolved calling service URL: %s", config.calling_service_base_url)
    logger.info("Outbound SSL verification enabled: %s", config.verify_ssl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fight and fall detection service")
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
        default=os.getenv("ALERT_PHONE_NUMBER", "7982373129"),
        help="Phone number that should receive the outbound alert call.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = PoseConfig(
        camera_index=args.camera_index,
        calling_service_base_url=args.calling_service_base_url,
        alert_phone_number=args.alert_number,
    )
    try:
        validate_startup_config(config)
    except ValueError as exc:
        logger.error("Startup configuration error: %s", exc)
        sys.exit(2)
    system = PoseSurveillanceSystem(config)
    system.run()
