from __future__ import annotations

import logging
import wave
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiTTS:
    def __init__(self, api_key: str, audio_dir: Path):
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.client = genai.Client(api_key=api_key) if api_key else None

    def generate_speech(
        self,
        text: str,
        filename: str,
        voice: str = "Kore",
    ) -> Path:
        if not self.client:
            raise RuntimeError("Gemini API key is missing for TTS.")

        file_path = self.audio_dir / f"{filename}.wav"
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
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)

        logger.info("Generated audio file %s", file_path.name)
        return file_path

