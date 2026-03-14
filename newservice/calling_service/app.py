from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from twilio.twiml.voice_response import VoiceResponse

from .runtime import logger, runtime
from .schemas import BroadcastCallRequest, CollectDetailsCallRequest

app = FastAPI(
    title="New Calling Service",
    description="Minimal FastAPI service for Twilio broadcast and details-collection calls.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/audio/{filename}")
async def serve_audio(filename: str) -> FileResponse:
    file_path = runtime.audio_file(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(file_path, media_type="audio/wav")


@app.post("/api/calls/broadcast")
async def create_broadcast_call(
    payload: BroadcastCallRequest,
) -> JSONResponse:
    ready, message = runtime.ensure_twilio_ready()
    if not ready:
        raise HTTPException(status_code=500, detail=message)

    base_url = runtime.resolve_base_url()
    session = runtime.create_broadcast_session(
        number=payload.number,
        message=payload.message,
        base_url=base_url,
    )
    webhook_url = (
        f"{base_url}/webhooks/twilio/call-flow"
        f"?flow=broadcast&token={session['token']}"
    )

    try:
        call = runtime.twilio_client().calls.create(
            url=webhook_url,
            to=payload.number,
            from_=runtime.settings.twilio_phone_number,
            method="POST",
        )
    except Exception as exc:
        runtime.clear_session(session["token"])
        logger.error("Failed to create broadcast call: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "status": "success",
            "flow": "broadcast",
            "call_sid": call.sid,
            "number": payload.number,
            "webhook_url": webhook_url,
        }
    )


@app.post("/api/calls/collect-details")
async def create_collect_details_call(
    payload: CollectDetailsCallRequest,
) -> JSONResponse:
    ready, message = runtime.ensure_twilio_ready()
    if not ready:
        raise HTTPException(status_code=500, detail=message)

    base_url = runtime.resolve_base_url()
    session = runtime.create_collect_session(
        number=payload.number,
        prompt=payload.prompt,
        base_url=base_url,
    )
    webhook_url = (
        f"{base_url}/webhooks/twilio/call-flow"
        f"?flow=collect&token={session['token']}"
    )

    try:
        call = runtime.twilio_client().calls.create(
            url=webhook_url,
            to=payload.number,
            from_=runtime.settings.twilio_phone_number,
            method="POST",
        )
    except Exception as exc:
        runtime.clear_session(session["token"])
        logger.error("Failed to create collect-details call: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "status": "success",
            "flow": "collect",
            "call_sid": call.sid,
            "number": payload.number,
            "webhook_url": webhook_url,
        }
    )


@app.api_route("/webhooks/twilio/call-flow", methods=["GET", "POST"])
async def twilio_call_flow(request: Request) -> Response:
    flow = request.query_params.get("flow", "").strip()
    token = request.query_params.get("token", "").strip()
    stage = request.query_params.get("stage", "").strip()

    if not flow or not token:
        raise HTTPException(status_code=400, detail="Missing flow or token.")

    session = runtime.get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Call session not found.")

    base_url = session.get("base_url") or runtime.resolve_base_url(request)
    twiml = VoiceResponse()

    if flow == "broadcast":
        audio_filename = session.get("audio_filename")
        if audio_filename:
            twiml.play(f"{base_url}/audio/{audio_filename}")
        else:
            twiml.say(session["message"])
        twiml.hangup()
        runtime.clear_session(token)
        return Response(content=str(twiml), media_type="application/xml")

    if flow != "collect":
        raise HTTPException(status_code=400, detail="Unsupported flow.")

    form_data = await request.form()
    recording_url = str(form_data.get("RecordingUrl", "")).strip()
    call_sid = str(form_data.get("CallSid", "")).strip()

    if stage != "recording" and not recording_url:
        prompt_audio_filename = session.get("prompt_audio_filename")
        if prompt_audio_filename:
            twiml.play(f"{base_url}/audio/{prompt_audio_filename}")
        else:
            twiml.say(session["prompt"])

        next_webhook = (
            f"{base_url}/webhooks/twilio/call-flow"
            f"?flow=collect&token={token}&stage=recording"
        )
        twiml.record(
            action=next_webhook,
            method="POST",
            max_length=60,
            timeout=3,
            play_beep=True,
            trim="trim-silence",
        )
        return Response(content=str(twiml), media_type="application/xml")

    if not recording_url:
        retry_count = int(session.get("retry_count", 0))
        if retry_count >= 1:
            twiml.say("Sorry, no details were recorded. Please try again later.")
            twiml.hangup()
            runtime.clear_session(token)
            return Response(content=str(twiml), media_type="application/xml")

        session["retry_count"] = retry_count + 1
        runtime.save_session(token, session)
        twiml.say("I did not catch that. Please describe the issue after the beep.")
        retry_webhook = (
            f"{base_url}/webhooks/twilio/call-flow"
            f"?flow=collect&token={token}&stage=recording"
        )
        twiml.record(
            action=retry_webhook,
            method="POST",
            max_length=60,
            timeout=3,
            play_beep=True,
            trim="trim-silence",
        )
        return Response(content=str(twiml), media_type="application/xml")

    transcript: str | None = None
    try:
        audio_bytes = runtime.download_recording(recording_url)
        transcript = runtime.transcriber.transcribe(audio_bytes)
    except Exception as exc:
        logger.warning("Could not download or transcribe recording: %s", exc)

    runtime.save_collected_call(
        {
            "token": token,
            "flow": "collect",
            "call_sid": call_sid or None,
            "number": session["number"],
            "prompt": session["prompt"],
            "recording_url": f"{recording_url}.mp3",
            "transcript": transcript,
            "created_at": session["created_at"],
            "completed_at": datetime.utcnow().isoformat(),
        }
    )
    runtime.clear_session(token)
    twiml.say("Thank you. Your details have been recorded.")
    twiml.hangup()
    return Response(content=str(twiml), media_type="application/xml")
