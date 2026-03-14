import os
import wave
import logging
from pathlib import Path
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiTTS:
    def __init__(self, audio_dir: Path):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def generate_speech(self, text: str, filename: str, voice: str = "Kore") -> Path:
        """Generate speech using Gemini TTS and save to a wave file."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized for TTS (no API key).")
            
        file_path = self.audio_dir / f"{filename}.wav"
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )
            audio_data = response.candidates[0].content.parts[0].inline_data.data

            with wave.open(str(file_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)

            logger.info("TTS Generated: %s (%d bytes)", file_path.name, len(audio_data))
            return file_path
            
        except Exception as e:
            logger.error("TTS failed: %s", e)
            raise e