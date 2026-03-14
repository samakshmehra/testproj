from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CALLING_SERVICE_BASE_URL = os.getenv(
    "NEW_CALLING_SERVICE_BASE_URL",
    "https://untrusting-oxymoronically-annita.ngrok-free.dev",
).rstrip("/")

DEFAULT_ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER", "").strip()
