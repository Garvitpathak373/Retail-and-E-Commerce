"""Privacy-preserving segment-level advisor with deterministic offline mock mode."""

from __future__ import annotations

import json
import os
from typing import Any


REQUIRED_KEYS = {"segment", "summary", "recommendations", "risk_note", "mode"}


def validate_advice(payload: dict[str, Any]) -> dict[str, Any]:
    if not REQUIRED_KEYS.issubset(payload) or not isinstance(payload["recommendations"], list):
        raise ValueError("Advisor response does not match the required JSON structure.")
    return payload


def mock_advice(segment_profile: dict[str, Any]) -> dict[str, Any]:
    name = str(segment_profile.get("Segment_Name", "Selected segment"))
    recency = float(segment_profile.get("Recency_Days", 0) or 0)
    frequency = float(segment_profile.get("Frequency", 0) or 0)
    action = "Prioritize retention outreach" if recency > 90 else "Use personalized cross-sell and replenishment outreach"
    return validate_advice({
        "segment": name,
        "summary": f"{name} has average frequency of {frequency:.2f} and average recency of {recency:.1f} days.",
        "recommendations": [action, "Measure response by repeat purchase and incremental order value."],
        "risk_note": "Advice is based on aggregated segment statistics only.",
        "mode": "mock",
    })


def get_segment_advice(segment_profile: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint only with aggregate segment statistics."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return mock_advice(segment_profile)
    try:
        from openai import OpenAI
        safe_profile = {key: value for key, value in segment_profile.items() if key not in {"Customer_ID", "customer_unique_id"}}
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input="Return JSON only with keys segment, summary, recommendations, risk_note, mode. Aggregated profile: " + json.dumps(safe_profile, default=str),
            max_output_tokens=350,
        )
        payload = json.loads(response.output_text)
        payload["mode"] = "api"
        return validate_advice(payload)
    except Exception:
        return mock_advice(segment_profile)
