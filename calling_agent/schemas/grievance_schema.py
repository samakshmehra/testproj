from typing import Optional, Literal
from pydantic import BaseModel, Field

class GrievanceRecord(BaseModel):
    is_valid: bool = Field(..., description="True if a real grievance/issue is being reported. False if it is spam or not an issue.")
    issue: Optional[str] = Field(None, description="Clear description of the reported issue. Leave null if is_valid is false.")
    location: Optional[str] = Field(None, description="Specific location mentioned by the caller. Leave null if is_valid is false.")
    priority: Optional[Literal["low", "medium", "high", "very_high"]] = None
    department: Optional[Literal["Public Works Department", "Water & Sanitation Department", "Power Department", "Waste Management Department", "Traffic Police Department", "Public Safety Department", "Environmental Department", "Health Department", "Education Department", "Telecommunication Department", "Housing & Construction Department", "Fire Department", "Municipal Corporation", "Revenue Department", "General Administration"]] = None
    category: Optional[Literal["road_infrastructure", "water_sanitation", "electricity_power", "waste_management", "traffic_transport", "public_safety", "environment_pollution", "healthcare_medical", "education_schools", "telecommunication", "housing_construction", "general_administration"]] = None
    resolution_days: Optional[int] = Field(None, description="Expected resolution time in days.")

class GrievanceChatResponse(BaseModel):
    user_transcript: str = Field(default="", description="The exact text transcription of what the user just said in the audio.")
    reply: str = Field(description="1 short sentence to say back to the user.")
    is_complete: bool = Field(default=False, description="True only when the caller has clearly provided both the problem and the location, and we are ready to end the call.")
    extracted_data: Optional[GrievanceRecord] = Field(default=None, description="Once is_complete is true, populate this field with the formalized record of the grievance.")