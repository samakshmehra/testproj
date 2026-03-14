"""
Calling Agent 2 — One-Way Alert System

Flow: Receive number & text -> Generate TTS -> Call number -> Play TTS -> Hangup

Run:
  1. python calling_agent/calling_agent_2.py
  2. ngrok http 5002
  3. POST /make_alert_call 
     {"number": "+91...", "text": "This is an alert!", "ngrok_url": "https://..."}
"""

import os
import sys
import uuid
import logging
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from flask import Flask, request, send_file
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

from calling_agent.tts import GeminiTTS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

AUDIO_DIR = Path(tempfile.mkdtemp(prefix="alert_audio_"))
logger.info("Audio files stored in: %s", AUDIO_DIR)

tts_service = GeminiTTS(AUDIO_DIR)

@app.route("/make_alert_call", methods=["POST"])
def make_alert_call():
    data = request.json or {}
    number = data.get("number")
    text = data.get("text")
    ngrok_url = data.get("ngrok_url")

    if not all([number, text, ngrok_url, ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER]):
        return {"status": "error", "message": "Missing required fields or Twilio credentials"}, 400

    try:
        # 1. Generate TTS using Gemini
        file_id = f"alert_{uuid.uuid4().hex[:8]}"
        audio_file = tts_service.generate_speech(text, file_id)

        # 2. Make Outbound Call
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        
        # We tell Twilio to fetch the TwiML instructions from our /twiml endpoint
        twiml_url = f"{ngrok_url.rstrip('/')}/twiml/{audio_file.name}"
        
        call = client.calls.create(
            url=twiml_url,
            to=number,
            from_=FROM_NUMBER
        )
        
        logger.info("Outbound call initiated: %s to %s", call.sid, number)
        return {"status": "success", "call_sid": call.sid, "message": f"Calling {number}"}
        
    except Exception as e:
        logger.error("Failed to make alert call: %s", e)
        return {"status": "error", "message": str(e)}, 500


@app.route("/twiml/<filename>", methods=["GET", "POST"])
def serve_twiml(filename):
    """Twilio hits this endpoint when the user picks up the phone."""
    
    # Check for ngrok forwarded host to safely build the audio file URL
    if request.headers.get("X-Forwarded-Host"):
        base_url = f"{request.headers.get('X-Forwarded-Proto', 'http')}://{request.headers.get('X-Forwarded-Host')}"
    else:
        base_url = request.url_root.rstrip("/")

    resp = VoiceResponse()
    
    # Play the newly generated TTS audio
    resp.play(f"{base_url}/audio/{filename}")
    
    # Hang up when done
    resp.hangup()
    
    return str(resp)


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve the raw .wav audio file to Twilio."""
    file_path = AUDIO_DIR / filename
    if not file_path.exists():
        return "Not found", 404
    return send_file(str(file_path), mimetype="audio/wav")


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  ALERT AGENT 2 — One-way TTS Caller")
    print("=" * 55)
    print("  1. ngrok http 5002")
    print("  2. curl -X POST http://localhost:5002/make_alert_call \\")
    print("       -H 'Content-Type: application/json' \\")
    print("""       -d '{\n            "number": "+91...",\n            "text": "Hello, this is an automated alert.",\n            "ngrok_url": "https://YOUR.ngrok-free.app"\n          }'""")
    print("=" * 55)
    print()
    # Runs on port 5002 so it doesn't conflict with Agent 1
    app.run(debug=True, port=5002, use_reloader=False)
