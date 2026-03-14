"""Helpers for detection code to trigger the new calling service."""

from .client import CallingServiceClient
from .services import (
    AccidentSuspiciousAlertService,
    FallFightAlertService,
)

__all__ = [
    "CallingServiceClient",
    "AccidentSuspiciousAlertService",
    "FallFightAlertService",
]
