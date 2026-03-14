from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionBroadcastRequest(BaseModel):
    number: str | None = Field(
        default=None,
        description="Optional phone number. Falls back to ALERT_PHONE_NUMBER.",
    )
    message: str = Field(..., min_length=1, description="Message that should be spoken.")


class DetectionCollectRequest(BaseModel):
    number: str | None = Field(
        default=None,
        description="Optional phone number. Falls back to ALERT_PHONE_NUMBER.",
    )
    prompt: str = Field(
        default="Hello from the grievance helpline. Please describe your issue after the beep.",
        min_length=1,
        description="Prompt that should be played before recording.",
    )
