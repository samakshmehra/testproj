import os
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiSTT:
    def __init__(self):
        # Initializes using OS environment variables
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.prompt = "Transcribe the audio exactly as spoken. Output only the text."

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/mp3") -> str:
        """Transcribe audio bytes using Gemini."""
        if not self.client:
            logger.error("Gemini client not initialized for STT (no API key).")
            return ""
            
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    self.prompt,
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
            )
            text = response.text.strip()
            logger.info("STT Result: %s", text)
            return text
        except Exception as e:
            logger.error("STT failed: %s", e)
            return ""