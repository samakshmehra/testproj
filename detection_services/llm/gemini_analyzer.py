import os
import io
from PIL import Image
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

from detection_services.schemas.alert_schema import DetectionAlert

load_dotenv()

class GeminiAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
    
    def analyze_frame(self, frame_bgr: np.ndarray, context_prompt: str) -> DetectionAlert | None:
        """
        Takes an OpenCV BGR frame, converts it to PIL Image, and sends it to Gemini 
        with the structured output schema DetectionAlert.
        """
        if not self.client:
            print("WARNING: GEMINI_API_KEY not set or client not initialized. Skipping LLM analysis.")
            return None
            
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = frame_bgr[..., ::-1]
            image = Image.fromarray(frame_rgb)
            
            prompt = (
                "You are an AI assistant for a Smart City emergency surveillance system.\n"
                f"CONTEXT DETECTED BY HEURISTICS: {context_prompt}\n\n"
                "TASK: Analyze the provided image of the scene and securely generate a structured alert.\n\n"
                "GUIDELINES for the fields:\n"
                "- 'is_valid': VERY IMPORTANT. Set to true ONLY IF there is a genuine incident or emergency visible. If the frame shows normal behavior, false alarms, or safe conditions, set this to false.\n"
                "- If 'is_valid' is false: Leave all other fields (issue, message, priority, department, category, resolution_days) as null/empty.\n"
                "- If 'is_valid' is true, please provide the following:\n"
                "  * 'issue': Provide a clear, 1-sentence factual description of what is actually visible in the frame (e.g., 'Vehicle accident visible at intersection').\n"
                "  * 'message': Write a concise, commanding, and professional script to be spoken via Text-To-Speech in a phone call to the authorities. Start with 'Emergency:' or 'Alert:' if critical.\n"
                "  * 'priority': Since these are LIVE physical anomalies (fights, falls, accidents), the priority should almost always be 'high' or 'very_high', unless it is an exceptionally minor issue.\n"
                "  * 'department', 'category': Select the most appropriate option based on the severity and nature of the incident. These MUST perfectly match one of the allowed string Literals from the schema.\n"
                "  * 'resolution_days': Estimate a realistic integer for how many days this type of issue usually takes to formally resolve or process (e.g., 1 for immediate safety incidents, 7 for infrastructure repairs).\n\n"
                "Output the result exactly matching the requested JSON schema for the detection alert."
            )
            
            # Note: Assuming gemini-1.5-flash or gemini-1.5-pro which support structured output
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DetectionAlert,
                    temperature=0.1
                )
            )
            
            # Since response_schema is provided, the parsed output will be a dictionary or object
            # matching the DetectionAlert Schema.
            if response.parsed:
                 return response.parsed
                 
            # Fallback if parsed is not directly populated by the SDK wrapper
            return DetectionAlert.model_validate_json(response.text)
            
        except Exception as e:
            print(f"Error calling Gemini LLM: {e}")
            return None