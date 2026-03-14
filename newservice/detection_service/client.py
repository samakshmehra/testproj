from __future__ import annotations

from typing import Any

import requests

from .config import DEFAULT_ALERT_PHONE_NUMBER, DEFAULT_CALLING_SERVICE_BASE_URL
from .schemas import DetectionBroadcastRequest, DetectionCollectRequest


class CallingServiceClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_CALLING_SERVICE_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.default_number = DEFAULT_ALERT_PHONE_NUMBER

    def _resolve_number(self, number: str | None) -> str:
        resolved_number = (number or self.default_number).strip()
        if not resolved_number:
            raise ValueError(
                "Phone number is required. Set ALERT_PHONE_NUMBER or pass number explicitly."
            )
        return resolved_number

    def send_broadcast(self, payload: DetectionBroadcastRequest) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/calls/broadcast",
            json={
                "number": self._resolve_number(payload.number),
                "message": payload.message,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def send_broadcast_message(
        self,
        message: str,
        number: str | None = None,
    ) -> dict[str, Any]:
        payload = DetectionBroadcastRequest(number=number, message=message)
        return self.send_broadcast(payload)

    def collect_details(self, payload: DetectionCollectRequest) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/calls/collect-details",
            json={
                "number": self._resolve_number(payload.number),
                "prompt": payload.prompt,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def request_details(
        self,
        prompt: str,
        number: str | None = None,
    ) -> dict[str, Any]:
        payload = DetectionCollectRequest(number=number, prompt=prompt)
        return self.collect_details(payload)
