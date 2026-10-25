"""Lightweight validation for advisor responses without customer-level data."""

from __future__ import annotations

from typing import Any

from llm_advisor import validate_advice


def evaluate_advice(payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, bool]:
    try:
        validate_advice(payload)
        valid_json = True
    except (TypeError, ValueError):
        valid_json = False
    text = " ".join([str(payload.get("summary", "")), " ".join(map(str, payload.get("recommendations", [])))])
    return {
        "json_valid": valid_json,
        "consistent_segment": str(profile.get("Segment_Name", "")).lower() in str(payload.get("segment", "")).lower(),
        "no_unsupported_customer_claims": "customer id" not in text.lower() and "email" not in text.lower(),
    }
