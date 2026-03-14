from __future__ import annotations

import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiAudioTranscriber:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key) if api_key else None

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/mp3") -> str | None:
        if not self.client:
            return None

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this caller audio. Return only the spoken words.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )

        text = (response.text or "").strip()
        if not text:
            logger.warning("Transcription completed with empty text.")
            return None
        return text

