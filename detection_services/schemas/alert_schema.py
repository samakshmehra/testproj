from typing import Optional, Literal
from pydantic import BaseModel, Field

class DetectionAlert(BaseModel):
    is_valid: bool = Field(..., description="True if a real incident/emergency is occurring in the frame, False if it is a false positive or normal behavior.")
    issue: Optional[str] = Field(None, description='Description of what was detected (e.g., "Vehicle accident detected at intersection"). Leave null if is_valid is false.')
    message: Optional[str] = Field(None, description='What to say in the voice call to authorities. Leave null if is_valid is false.')
    priority: Optional[Literal["low", "medium", "high", "very_high"]] = None
    department: Optional[Literal["Public Works Department", "Water & Sanitation Department", "Power Department", "Waste Management Department", "Traffic Police Department", "Public Safety Department", "Environmental Department", "Health Department", "Education Department", "Telecommunication Department", "Housing & Construction Department", "Fire Department", "Municipal Corporation", "Revenue Department", "General Administration"]] = None
    category: Optional[Literal["road_infrastructure", "water_sanitation", "electricity_power", "waste_management", "traffic_transport", "public_safety", "environment_pollution", "healthcare_medical", "education_schools", "telecommunication", "housing_construction", "general_administration"]] = None
    resolution_days: Optional[int] = None
