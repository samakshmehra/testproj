from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SERVICE_ROOT = Path(__file__).resolve().parent
STORAGE_ROOT = SERVICE_ROOT / "storage"
AUDIO_ROOT = STORAGE_ROOT / "audio"
COLLECTED_CALLS_FILE = STORAGE_ROOT / "collected_calls.json"


@dataclass(slots=True)
class Settings:
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "").strip()
    default_target_number: str = os.getenv("ALERT_PHONE_NUMBER", "").strip()
    public_base_url: str = os.getenv(
        "NEW_CALLING_SERVICE_PUBLIC_BASE_URL",
        "https://untrusting-oxymoronically-annita.ngrok-free.dev",
    ).rstrip("/")
    gemini_api_key: str = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    host: str = os.getenv("NEW_CALLING_SERVICE_HOST", "0.0.0.0").strip()
    port: int = int(os.getenv("NEW_CALLING_SERVICE_PORT", "8000"))
