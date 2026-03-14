import os
import logging
from google import genai
from google.genai import types

from calling_agent.schemas.grievance_schema import GrievanceChatResponse

logger = logging.getLogger(__name__)

class GeminiLLM:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = (
            "You are an AI assistant for a Smart City grievance system. You take calls from citizens "
            "who are reporting public issues (potholes, water leaks, power outages, etc.).\n\n"
            "Your goals:\n"
            "1. Ask what the problem is.\n"
            "2. Ask for the specific location.\n"
            "3. Collect the details. If you don't have BOTH the problem and the location, keep asking.\n"
            "4. Once you have both, set `is_complete` to true, wrap up warmly, and output the `extracted_data`.\n\n"
            "Rules for your replies:\n"
            "- ALWAYS reply in a maximum of 1 or 2 short sentences.\n"
            "- Be conversational, empathetic, and ultra-brief (helpful over the phone).\n"
            "- Don't invent details. Only use what the caller said."
        )

    def generate_response(self, history: list[dict], audio_bytes: bytes = None) -> GrievanceChatResponse:
        """Call Gemini to get the next turn or extraction, directly from audio if provided."""
        if not self.client:
            logger.error("Gemini client not initialized for LLM (no API key).")
            return GrievanceChatResponse(reply="Sorry, I am offline.")

        # Build transcript string
        transcript = "Chat Transcript so far:\n"
        for msg in history:
            role_tag = "Caller" if msg["role"] == "user" else "Agent"
            transcript += f"{role_tag}: {msg['text']}\n"

        if audio_bytes:
            contents = [
                f"{self.system_prompt}\n\n{transcript}\n\nBased on the attached audio of the newest Caller reply, first transcribe what they said into 'user_transcript', then provide your new 'reply', and see if you can extract data.",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
            ]
        else:
            contents = [f"{self.system_prompt}\n\n{transcript}\nAgent:"]

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", # Revert to Flash 2.5 standard, it is much better at native multi-modal audio processing
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GrievanceChatResponse,
                    temperature=0.3
                ),
            )
            
            if response.parsed:
                 result = response.parsed
            else:
                 result = GrievanceChatResponse.model_validate_json(response.text)
                 
            logger.info("LLM Reply: %s", result.reply)
            return result
            
        except Exception as e:
            logger.error("LLM Generation failed: %s", e)
            return GrievanceChatResponse(reply="I'm sorry, I didn't quite get that. Could you repeat?")