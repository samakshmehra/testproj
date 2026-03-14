"""
Grievance Reporting System — AI Conversational Agent via Twilio + Gemini

Flow: User speaks → Record → Gemini STT → Gemini LLM → Gemini TTS → Play back → loop

Run:
  1. python grievance_system.py
  2. ngrok http 5001
  3. curl -X POST http://localhost:5001/make_call \
       -H 'Content-Type: application/json' \
       -d '{"ngrok_url":"https://YOUR-URL.ngrok-free.app"}'
"""

import os
import sys
import json
import uuid
import logging
import threading
import tempfile
import requests as http_requests
from datetime import datetime
from pathlib import Path

# Fix paths so the app can be run from anywhere and still import properly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Also force JSON writing to the root of the project
GRIEVANCES_FILE = PROJECT_ROOT / "grievances.json"

from flask import Flask, request, send_file
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

# Import our new modular services
from calling_agent.tts import GeminiTTS
from calling_agent.llm import GeminiLLM

load_dotenv()

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "grievance.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────
app = Flask(__name__)

# ──────────────────────────────────────────────
# Credentials
# ──────────────────────────────────────────────
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TO_NUMBER = os.getenv("ALERT_PHONE_NUMBER")

# ──────────────────────────────────────────────
# Audio Storage
# ──────────────────────────────────────────────
AUDIO_DIR = Path(tempfile.mkdtemp(prefix="grievance_audio_"))
logger.info("Audio files stored in: %s", AUDIO_DIR)

# ──────────────────────────────────────────────
# AI Services
# ──────────────────────────────────────────────
tts_service = GeminiTTS(AUDIO_DIR)
llm_service = GeminiLLM()

# ──────────────────────────────────────────────
# Grievance Storage
# ──────────────────────────────────────────────
_file_lock = threading.Lock()


def _load_grievances() -> list[dict]:
    if not GRIEVANCES_FILE.exists():
        return []
    try:
        with open(GRIEVANCES_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_grievance(grievance: dict):
    with _file_lock:
        data = _load_grievances()
        data.append(grievance)
        with open(GRIEVANCES_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Grievance saved: %s", grievance.get("id"))


# ──────────────────────────────────────────────
# Per-call conversation state
# ──────────────────────────────────────────────
_call_data: dict[str, dict] = {}
_call_lock = threading.Lock()


def _get_call(call_sid: str) -> dict:
    with _call_lock:
        return _call_data.setdefault(call_sid, {
            "history": [],
            "caller_number": "unknown",
        })


def _clear_call(call_sid: str):
    with _call_lock:
        _call_data.pop(call_sid, None)


def download_recording(recording_url: str) -> bytes:
    """Download recording from Twilio."""
    url = recording_url + ".mp3"
    r = http_requests.get(url, auth=(ACCOUNT_SID, AUTH_TOKEN), timeout=10)
    r.raise_for_status()
    logger.info("Recording: %d bytes", len(r.content))
    return r.content


# ──────────────────────────────────────────────
# Pre-generate greeting at startup
# ──────────────────────────────────────────────
GREETING_TEXT = "Hey! Grievance helpline. What issue are you facing?"
_greeting_file: Path | None = None


def pregenerate_greeting():
    global _greeting_file
    try:
        _greeting_file = tts_service.generate_speech(GREETING_TEXT, "greeting")
    except Exception as e:
        logger.error("Greeting gen failed: %s", e)


# ──────────────────────────────────────────────
# Audio serving
# ──────────────────────────────────────────────
@app.route("/audio/<filename>")
def serve_audio(filename):
    file_path = AUDIO_DIR / filename
    if not file_path.exists():
        return "Not found", 404
    return send_file(str(file_path), mimetype="audio/wav")


# ══════════════════════════════════════════════
# CALL ROUTES
# ══════════════════════════════════════════════

@app.route("/voice", methods=["POST"])
def start_call():
    """Play greeting → record user speech."""
    call_sid = request.form.get("CallSid", "")
    caller = request.form.get("From", "unknown")
    call_data = _get_call(call_sid)
    call_data["caller_number"] = caller
    call_data["history"].append({"role": "assistant", "text": GREETING_TEXT})

    # Safely get the public URL (ngrok usually puts it in HTTP_X_FORWARDED_HOST or Host)
    if request.headers.get("X-Forwarded-Host"):
        base_url = f"{request.headers.get('X-Forwarded-Proto', 'http')}://{request.headers.get('X-Forwarded-Host')}"
    else:
        base_url = request.url_root.rstrip("/")

    resp = VoiceResponse()

    if _greeting_file and _greeting_file.exists():
        resp.play(f"{base_url}/audio/{_greeting_file.name}")
    else:
        resp.say(GREETING_TEXT)

    resp.record(
        max_length=30,
        timeout=2,
        action="/conversation",
        play_beep=False,
        trim="trim-silence",
    )
    return str(resp)


@app.route("/conversation", methods=["POST"])
def conversation():
    """STT → LLM → TTS → play → record → loop."""
    call_sid = request.form.get("CallSid", "")
    recording_url = request.form.get("RecordingUrl", "")
    call_data = _get_call(call_sid)
    
    # Safely get the public URL (ngrok usually puts it in HTTP_X_FORWARDED_HOST or Host)
    if request.headers.get("X-Forwarded-Host"):
        base_url = f"{request.headers.get('X-Forwarded-Proto', 'http')}://{request.headers.get('X-Forwarded-Host')}"
    else:
        base_url = request.url_root.rstrip("/")

    # ── STT & LLM ──
    audio = None
    if recording_url:
        try:
            audio = download_recording(recording_url)
        except Exception as e:
            logger.error("Failed to download recording: %s", e)

    if not audio:
        resp = VoiceResponse()
        resp.say("Sorry, didn't catch that.")
        resp.record(
            max_length=30, timeout=2, action="/conversation",
            play_beep=False, trim="trim-silence",
        )
        return str(resp)

    # Directly pass audio to Native Multimodal LLM
    llm_result = llm_service.generate_response(call_data["history"], audio_bytes=audio)
    
    user_text = llm_result.user_transcript or "(unintelligible)"
    call_data["history"].append({"role": "user", "text": user_text})
    logger.info("[%s] Caller (Native Audio): %s", call_sid[:8], user_text)

    call_data["history"].append({"role": "assistant", "text": llm_result.reply})
    logger.info("[%s] Agent: %s", call_sid[:8], llm_result.reply)

    # ── TTS ──
    resp = VoiceResponse()
    turn_id = f"t_{call_sid[:8]}_{len(call_data['history'])}"

    try:
        audio_file = tts_service.generate_speech(llm_result.reply, turn_id)
        resp.play(f"{base_url}/audio/{audio_file.name}")
    except Exception:
        resp.say(llm_result.reply)

    # ── Loop or finish ──
    if llm_result.is_complete:
        record_data = (
            llm_result.extracted_data.model_dump()
            if llm_result.extracted_data else {"is_valid": False, "issue": "unparseable"}
        )
        
        grievance = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "caller_number": call_data.get("caller_number", "unknown"),
            "extracted_data": record_data,
            "conversation": call_data.get("history", []),
        }
        _save_grievance(grievance)

        logger.info(
            "\n%s\nGRIEVANCE RECORDED\n%s\n  ID: %s\n  Valid: %s\n  Issue: %s\n  Location: %s\n  Priority: %s\n  Department: %s\n%s",
            "=" * 50, "=" * 50,
            grievance["id"], 
            record_data.get("is_valid"),
            record_data.get("issue"), 
            record_data.get("location"),
            record_data.get("priority"),
            record_data.get("department"),
            "=" * 50,
        )
        _clear_call(call_sid)
        resp.hangup()
    else:
        resp.record(
            max_length=30, timeout=2, action="/conversation",
            play_beep=False, trim="trim-silence",
        )

    return str(resp)


# ── Make call ────────────────────────────────

@app.route("/make_call", methods=["POST"])
def make_call():
    if not all([ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBER]):
        return {"status": "error", "message": "Missing credentials"}, 500

    ngrok_url = request.json.get("ngrok_url")
    if not ngrok_url:
        return {"status": "error", "message": "ngrok_url required"}, 400

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        call = client.calls.create(
            url=f"{ngrok_url}/voice", to=TO_NUMBER, from_=FROM_NUMBER,
        )
        logger.info("Call: SID=%s, to=%s", call.sid, TO_NUMBER)
        return {"status": "success", "call_sid": call.sid, "calling": TO_NUMBER}
    except Exception as e:
        logger.error("Call failed: %s", e)
        return {"status": "error", "message": str(e)}, 500


@app.route("/grievances", methods=["GET"])
def list_grievances():
    return {"count": len(_load_grievances()), "grievances": _load_grievances()}


# ══════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  GRIEVANCE AI — Gemini STT + LLM + TTS")
    print("=" * 55)
    print("  Pre-generating greeting...")
    pregenerate_greeting()
    print()
    print("  1. ngrok http 5001")
    print("  2. curl -X POST http://localhost:5001/make_call \\")
    print("       -H 'Content-Type: application/json' \\")
    print("""       -d '{"ngrok_url":"https://YOUR.ngrok-free.app"}'""")
    print("  3. http://localhost:5001/grievances")
    print("=" * 55)
    print()
    app.run(debug=True, port=5001, use_reloader=False)
