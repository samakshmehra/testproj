from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import requests as http_requests
from dotenv import load_dotenv
from flask import Request
from twilio.rest import Client

# Ensure package imports work when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from calling_agent.llm import GeminiLLM
from calling_agent.tts import GeminiTTS

load_dotenv()

GRIEVANCES_FILE = PROJECT_ROOT / "grievances.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "calling_agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class CallingServerRuntime:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.default_to_number = os.getenv("ALERT_PHONE_NUMBER")

        self.audio_dir = Path(tempfile.mkdtemp(prefix="calling_agent_audio_"))
        logger.info("Audio files stored in: %s", self.audio_dir)

        self.tts_service = GeminiTTS(self.audio_dir)
        self.llm_service = GeminiLLM()

        self.greeting_text = "Hey! Grievance helpline. What issue are you facing?"
        self.greeting_file: Path | None = None

        self._file_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._call_data: dict[str, dict[str, Any]] = {}

    def load_grievances(self) -> list[dict[str, Any]]:
        if not GRIEVANCES_FILE.exists():
            return []
        try:
            with open(GRIEVANCES_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def save_grievance(self, grievance: dict[str, Any]) -> None:
        with self._file_lock:
            data = self.load_grievances()
            data.append(grievance)
            with open(GRIEVANCES_FILE, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Grievance saved: %s", grievance.get("id"))

    def get_call(self, call_sid: str) -> dict[str, Any]:
        with self._call_lock:
            return self._call_data.setdefault(
                call_sid,
                {
                    "history": [],
                    "target_number": "unknown",
                },
            )

    def clear_call(self, call_sid: str) -> None:
        with self._call_lock:
            self._call_data.pop(call_sid, None)

    def ensure_twilio_ready(self) -> tuple[bool, str]:
        if not self.account_sid or not self.auth_token or not self.from_number:
            return False, "Missing Twilio credentials in environment."
        return True, ""

    def resolve_base_url(
        self, request_obj: Request, payload: dict[str, Any] | None = None
    ) -> str:
        payload = payload or {}
        ngrok_url = str(payload.get("ngrok_url", "")).strip()
        if ngrok_url:
            return ngrok_url.rstrip("/")

        forwarded_host = request_obj.headers.get("X-Forwarded-Host")
        if forwarded_host:
            proto = request_obj.headers.get("X-Forwarded-Proto", "http")
            return f"{proto}://{forwarded_host}".rstrip("/")

        return request_obj.url_root.rstrip("/")

    def twilio_client(self) -> Client:
        return Client(self.account_sid, self.auth_token)

    def download_recording(self, recording_url: str) -> bytes:
        if not self.account_sid or not self.auth_token:
            raise RuntimeError("Twilio credentials missing for recording download.")

        response = http_requests.get(
            recording_url + ".mp3",
            auth=(self.account_sid, self.auth_token),
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Downloaded recording (%d bytes)", len(response.content))
        return response.content

    def pregenerate_greeting(self) -> None:
        try:
            self.greeting_file = self.tts_service.generate_speech(
                self.greeting_text, "greeting"
            )
        except Exception as exc:
            logger.error("Greeting generation failed: %s", exc)


runtime = CallingServerRuntime()
