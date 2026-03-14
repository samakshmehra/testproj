from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionBroadcastRequest(BaseModel):
    number: str = Field(..., description="Phone number to call.")
    message: str = Field(..., min_length=1, description="Message that should be spoken.")
    public_base_url: str | None = Field(
        default=None,
        description="Optional public base URL for Twilio, such as ngrok.",
    )


class DetectionCollectRequest(BaseModel):
    number: str = Field(..., description="Phone number to call.")
    prompt: str = Field(
        default="Hello from the grievance helpline. Please describe your issue after the beep.",
        min_length=1,
        description="Prompt that should be played before recording.",
    )
    public_base_url: str | None = Field(
        default=None,
        description="Optional public base URL for Twilio, such as ngrok.",
    )

