from __future__ import annotations

import os
import uuid
from datetime import datetime

from flask import Flask, request, send_file
from twilio.twiml.voice_response import VoiceResponse

from calling_agent.services.runtime import logger, runtime


def register_routes(app: Flask) -> None:
    @app.route("/audio/<filename>")
    def serve_audio(filename: str):
        file_path = runtime.audio_dir / filename
        if not file_path.exists():
            return "Not found", 404
        return send_file(str(file_path), mimetype="audio/wav")

    @app.route("/health", methods=["GET"])
    def health_check():
        return {"status": "ok"}

    @app.route("/grievances", methods=["GET"])
    def list_grievances():
        records = runtime.load_grievances()
        return {"count": len(records), "grievances": records}

    @app.route("/complaint/register", methods=["POST"])
    @app.route("/make_call", methods=["POST"])
    def complaint_register():
        ready, message = runtime.ensure_twilio_ready()
        if not ready:
            return {"status": "error", "message": message}, 500

        data = request.get_json(silent=True) or {}
        number = data.get("number") or runtime.default_to_number
        if not number:
            return {
                "status": "error",
                "message": "number is required (or set ALERT_PHONE_NUMBER in env).",
            }, 400

        base_url = runtime.resolve_base_url(request, data)
        webhook_url = f"{base_url}/voice/complaint"

        try:
            call = runtime.twilio_client().calls.create(
                url=webhook_url,
                to=number,
                from_=runtime.from_number,
            )
            logger.info("Complaint call initiated: %s -> %s", call.sid, number)
            return {
                "status": "success",
                "flow": "complaint_registration",
                "call_sid": call.sid,
                "calling": number,
                "webhook_url": webhook_url,
            }
        except Exception as exc:
            logger.error("Failed to start complaint call: %s", exc)
            return {"status": "error", "message": str(exc)}, 500

    @app.route("/voice/complaint", methods=["POST"])
    @app.route("/voice", methods=["POST"])
    def complaint_voice_start():
        call_sid = request.form.get("CallSid", "")
        target_number = request.form.get("To") or request.form.get("From", "unknown")

        call_data = runtime.get_call(call_sid)
        call_data["target_number"] = target_number
        call_data["history"].append(
            {"role": "assistant", "text": runtime.greeting_text}
        )

        base_url = runtime.resolve_base_url(request)
        response = VoiceResponse()

        if runtime.greeting_file and runtime.greeting_file.exists():
            response.play(f"{base_url}/audio/{runtime.greeting_file.name}")
        else:
            response.say(runtime.greeting_text)

        response.record(
            max_length=30,
            timeout=2,
            action="/voice/complaint/conversation",
            play_beep=False,
            trim="trim-silence",
        )
        return str(response)

    @app.route("/voice/complaint/conversation", methods=["POST"])
    @app.route("/conversation", methods=["POST"])
    def complaint_conversation():
        call_sid = request.form.get("CallSid", "")
        recording_url = request.form.get("RecordingUrl", "")
        call_data = runtime.get_call(call_sid)
        base_url = runtime.resolve_base_url(request)

        if not recording_url:
            response = VoiceResponse()
            response.say("Sorry, I didn't catch that.")
            response.record(
                max_length=30,
                timeout=2,
                action="/voice/complaint/conversation",
                play_beep=False,
                trim="trim-silence",
            )
            return str(response)

        try:
            audio = runtime.download_recording(recording_url)
        except Exception as exc:
            logger.error("Failed to download recording: %s", exc)
            response = VoiceResponse()
            response.say("Sorry, I had trouble hearing you. Please repeat.")
            response.record(
                max_length=30,
                timeout=2,
                action="/voice/complaint/conversation",
                play_beep=False,
                trim="trim-silence",
            )
            return str(response)

        llm_result = runtime.llm_service.generate_response(
            call_data["history"], audio_bytes=audio
        )
        user_text = llm_result.user_transcript or "(unintelligible)"
        call_data["history"].append({"role": "user", "text": user_text})
        call_data["history"].append({"role": "assistant", "text": llm_result.reply})

        logger.info("[%s] Caller: %s", call_sid[:8], user_text)
        logger.info("[%s] Agent: %s", call_sid[:8], llm_result.reply)

        response = VoiceResponse()
        turn_id = f"t_{call_sid[:8]}_{len(call_data['history'])}"
        try:
            audio_file = runtime.tts_service.generate_speech(llm_result.reply, turn_id)
            response.play(f"{base_url}/audio/{audio_file.name}")
        except Exception:
            response.say(llm_result.reply)

        if llm_result.is_complete:
            extracted = (
                llm_result.extracted_data.model_dump()
                if llm_result.extracted_data
                else {"is_valid": False, "issue": "unparseable"}
            )

            grievance = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().isoformat(),
                "caller_number": call_data.get("target_number", "unknown"),
                "extracted_data": extracted,
                "conversation": call_data.get("history", []),
            }
            runtime.save_grievance(grievance)
            runtime.clear_call(call_sid)
            response.hangup()
        else:
            response.record(
                max_length=30,
                timeout=2,
                action="/voice/complaint/conversation",
                play_beep=False,
                trim="trim-silence",
            )

        return str(response)

    @app.route("/info/send", methods=["POST"])
    @app.route("/make_alert_call", methods=["POST"])
    def info_sender():
        ready, message = runtime.ensure_twilio_ready()
        if not ready:
            return {"status": "error", "message": message}, 500

        data = request.get_json(silent=True) or {}
        number = data.get("number")
        text = data.get("message") or data.get("msg") or data.get("text")

        if not number or not text:
            return {
                "status": "error",
                "message": "Both number and message/text are required.",
            }, 400

        base_url = runtime.resolve_base_url(request, data)

        try:
            file_id = f"info_{uuid.uuid4().hex[:8]}"
            audio_file = runtime.tts_service.generate_speech(text, file_id)
            twiml_url = f"{base_url}/voice/info/{audio_file.name}"

            call = runtime.twilio_client().calls.create(
                url=twiml_url,
                to=number,
                from_=runtime.from_number,
            )
            logger.info("Info call initiated: %s -> %s", call.sid, number)
            return {
                "status": "success",
                "flow": "info_sender",
                "call_sid": call.sid,
                "calling": number,
                "twiml_url": twiml_url,
            }
        except Exception as exc:
            logger.error("Failed to start info sender call: %s", exc)
            return {"status": "error", "message": str(exc)}, 500

    @app.route("/voice/info/<filename>", methods=["GET", "POST"])
    @app.route("/twiml/<filename>", methods=["GET", "POST"])
    def voice_info(filename: str):
        base_url = runtime.resolve_base_url(request)
        response = VoiceResponse()
        response.play(f"{base_url}/audio/{filename}")
        response.hangup()
        return str(response)


def run_server(app: Flask) -> None:
    port = int(os.getenv("CALLING_AGENT_PORT", "5002"))
    runtime.pregenerate_greeting()

    print()
    print("=" * 70)
    print("  COMMON CALLING SERVER — Complaint + Info Sender")
    print("=" * 70)
    print("  Complaint endpoint:")
    print("    POST /complaint/register")
    print('    Body: {"number":"+91...", "ngrok_url":"https://YOUR.ngrok-free.app"}')
    print()
    print("  Info sender endpoint:")
    print("    POST /info/send")
    print('    Body: {"number":"+91...", "message":"Your alert message", "ngrok_url":"https://YOUR.ngrok-free.app"}')
    print("=" * 70)
    print()

    app.run(debug=True, port=port, use_reloader=False)
