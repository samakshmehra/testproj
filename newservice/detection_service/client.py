from __future__ import annotations

import os
from typing import Any

import requests

from .schemas import DetectionBroadcastRequest, DetectionCollectRequest


class CallingServiceClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("NEW_CALLING_SERVICE_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout

    def send_broadcast(self, payload: DetectionBroadcastRequest) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/calls/broadcast",
            json=payload.model_dump(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def collect_details(self, payload: DetectionCollectRequest) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/calls/collect-details",
            json=payload.model_dump(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
