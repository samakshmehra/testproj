from __future__ import annotations

from typing import Any

from detection_services.schemas.alert_schema import DetectionAlert

from .client import CallingServiceClient


class _BaseAlertService:
    fallback_message = "An emergency has been detected. Please respond immediately."

    def __init__(self, client: CallingServiceClient | None = None) -> None:
        self.client = client or CallingServiceClient()

    def _message_from_alert(self, alert: DetectionAlert) -> str:
        return (alert.message or alert.issue or self.fallback_message).strip()

    def send_alert(
        self,
        alert: DetectionAlert,
        number: str | None = None,
    ) -> dict[str, Any]:
        if not alert.is_valid:
            return {
                "status": "skipped",
                "reason": "alert_not_valid",
            }

        return self.client.send_broadcast_message(
            message=self._message_from_alert(alert),
            number=number,
        )


class AccidentSuspiciousAlertService(_BaseAlertService):
    fallback_message = (
        "A possible accident or suspicious public-safety incident has been detected."
    )


class FallFightAlertService(_BaseAlertService):
    fallback_message = (
        "A possible fall or fight has been detected. Immediate attention may be required."
    )
