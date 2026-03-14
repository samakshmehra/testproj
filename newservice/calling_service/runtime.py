from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from fastapi import Request
from twilio.rest import Client

from .config import AUDIO_ROOT, COLLECTED_CALLS_FILE, STORAGE_ROOT, Settings
from .transcriber import GeminiAudioTranscriber
from .tts import GeminiTTS

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "new_calling_service.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class CallingServiceRuntime:
    def __init__(self) -> None:
        self.settings = Settings()

        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

        self.tts_service = GeminiTTS(self.settings.gemini_api_key, AUDIO_ROOT)
        self.transcriber = GeminiAudioTranscriber(self.settings.gemini_api_key)

        self._sessions: dict[str, dict[str, Any]] = {}
        self._session_lock = threading.Lock()
        self._file_lock = threading.Lock()

    def ensure_twilio_ready(self) -> tuple[bool, str]:
        if not (
            self.settings.twilio_account_sid
            and self.settings.twilio_auth_token
            and self.settings.twilio_phone_number
        ):
            return False, "Missing Twilio credentials in environment."
        return True, ""

    def twilio_client(self) -> Client:
        return Client(
            self.settings.twilio_account_sid,
            self.settings.twilio_auth_token,
        )

    def resolve_base_url(
        self,
        request: Request,
        explicit_base_url: str | None = None,
    ) -> str:
        if explicit_base_url:
            return explicit_base_url.rstrip("/")

        forwarded_host = request.headers.get("x-forwarded-host")
        if forwarded_host:
            proto = request.headers.get("x-forwarded-proto", "https")
            return f"{proto}://{forwarded_host}".rstrip("/")

        return str(request.base_url).rstrip("/")

    def create_broadcast_session(
        self,
        number: str,
        message: str,
        base_url: str,
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex[:12]
        audio_filename: str | None = None

        try:
            audio_path = self.tts_service.generate_speech(message, f"broadcast_{token}")
            audio_filename = audio_path.name
        except Exception as exc:
            logger.warning("Broadcast TTS generation failed, falling back to Twilio Say: %s", exc)

        session = {
            "token": token,
            "flow": "broadcast",
            "number": number,
            "message": message,
            "audio_filename": audio_filename,
            "base_url": base_url,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.save_session(token, session)
        return session

    def create_collect_session(
        self,
        number: str,
        prompt: str,
        base_url: str,
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex[:12]
        prompt_audio_filename: str | None = None

        try:
            audio_path = self.tts_service.generate_speech(prompt, f"collect_{token}")
            prompt_audio_filename = audio_path.name
        except Exception as exc:
            logger.warning("Prompt TTS generation failed, falling back to Twilio Say: %s", exc)

        session = {
            "token": token,
            "flow": "collect",
            "number": number,
            "prompt": prompt,
            "prompt_audio_filename": prompt_audio_filename,
            "base_url": base_url,
            "retry_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.save_session(token, session)
        return session

    def save_session(self, token: str, session: dict[str, Any]) -> None:
        with self._session_lock:
            self._sessions[token] = session

    def get_session(self, token: str) -> dict[str, Any] | None:
        with self._session_lock:
            return self._sessions.get(token)

    def clear_session(self, token: str) -> None:
        with self._session_lock:
            self._sessions.pop(token, None)

    def audio_file(self, filename: str) -> Path:
        return AUDIO_ROOT / filename

    def download_recording(self, recording_url: str) -> bytes:
        response = requests.get(
            f"{recording_url}.mp3",
            auth=(
                self.settings.twilio_account_sid,
                self.settings.twilio_auth_token,
            ),
            timeout=20,
        )
        response.raise_for_status()
        return response.content

    def load_collected_calls(self) -> list[dict[str, Any]]:
        if not COLLECTED_CALLS_FILE.exists():
            return []

        try:
            with open(COLLECTED_CALLS_FILE, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read collected calls file, starting fresh.")
            return []

        return data if isinstance(data, list) else []

    def save_collected_call(self, record: dict[str, Any]) -> None:
        with self._file_lock:
            data = self.load_collected_calls()
            data.append(record)
            with open(COLLECTED_CALLS_FILE, "w", encoding="utf-8") as file_obj:
                json.dump(data, file_obj, indent=2, ensure_ascii=False)


runtime = CallingServiceRuntime()

