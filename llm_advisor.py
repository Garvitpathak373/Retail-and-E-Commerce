"""Aggregate-only Gemini/OpenAI-compatible segment advisor."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OFFLINE_FALLBACK_NOTICE = (
    "AI insights are temporarily operating in offline mode. "
    "Recommendations are based on available dataset metrics."
)

try:
    from dotenv import load_dotenv
except ImportError:  # Optional convenience dependency; environment variables still work.
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False
from configs import OUTPUTS_DIR

ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
REQUIRED_KEYS = {"segment", "summary", "key_drivers", "risks", "actions", "mode"}


class GeminiCircuitBreaker:
    """Thread-safe circuit breaker preventing API hammering on quota or auth failure."""

    def __init__(self):
        self._lock = threading.RLock()
        self.state: str = "CLOSED"  # "CLOSED", "OPEN", "HALF_OPEN"
        self.last_failure_time: float = 0.0
        self.cooldown_seconds: float = 0.0
        self.last_error_type: str = ""
        self.failure_count: int = 0
        self.retry_after: float = 0.0
        self.last_api_key_hash: str = ""

    def check_key_rotation(self, current_key: str | None) -> None:
        """Reset circuit breaker automatically if API key changes."""
        with self._lock:
            key_hash = hashlib.sha256(current_key.encode("utf-8")).hexdigest() if current_key else ""
            if self.last_api_key_hash and key_hash != self.last_api_key_hash:
                self._reset_locked()
            self.last_api_key_hash = key_hash

    def should_allow_request(self) -> tuple[bool, str, float]:
        with self._lock:
            now = time.time()
            if self.state == "OPEN":
                remaining = max(0.0, (self.last_failure_time + self.cooldown_seconds) - now)
                if remaining > 0:
                    return False, self.last_error_type, remaining
                # Cooldown expired: transition to HALF_OPEN to attempt recovery
                self.state = "HALF_OPEN"
                return True, "half_open_probe", 0.0
            return True, "closed", 0.0

    def record_success(self) -> None:
        with self._lock:
            self._reset_locked()

    def record_failure(self, error_type: str, custom_cooldown: float | None = None, retry_after: float = 0.0) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.last_error_type = error_type
            self.retry_after = retry_after

            # Quota exceeded and invalid auth trip the circuit immediately (threshold=1).
            # Transient server/network errors trip after 2 consecutive failures.
            trip_threshold = 1 if error_type in ("quota_exceeded", "invalid_api_key") else 2

            if custom_cooldown is not None and custom_cooldown > 0:
                self.cooldown_seconds = custom_cooldown
                self.state = "OPEN"
            elif error_type == "quota_exceeded":
                self.cooldown_seconds = max(retry_after, 60.0)
                self.state = "OPEN"
            elif error_type == "invalid_api_key":
                self.cooldown_seconds = 300.0
                self.state = "OPEN"
            elif self.failure_count >= trip_threshold:
                if error_type == "model_not_found_404":
                    self.cooldown_seconds = 120.0
                elif error_type in ("service_unavailable_503", "deadline_exceeded_504", "network_error"):
                    self.cooldown_seconds = 20.0
                else:
                    self.cooldown_seconds = 15.0
                self.state = "OPEN"

    def _reset_locked(self) -> None:
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.cooldown_seconds = 0.0
        self.last_error_type = ""
        self.retry_after = 0.0

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def get_remaining_cooldown(self) -> float:
        with self._lock:
            if self.state != "OPEN":
                return 0.0
            now = time.time()
            return max(0.0, (self.last_failure_time + self.cooldown_seconds) - now)

    def get_recovery_hint(self) -> str:
        with self._lock:
            rem = self.get_remaining_cooldown()
            err = self.last_error_type
            if err == "invalid_api_key":
                return "External action required: Update GEMINI_API_KEY with a valid key in .env or settings."
            elif err == "quota_exceeded":
                return f"External action / wait required: Quota limit reached. Cooldown active ({rem:.0f}s remaining). Wait for quota reset."
            elif err == "model_not_found_404":
                return "Configured model reported unavailable (404) by Google. Fallback models attempted automatically. Consider setting GEMINI_MODEL=gemini-2.0-flash in .env."
            elif err in ("service_unavailable_503", "deadline_exceeded_504"):
                return f"Transient provider issue / 504 ({rem:.0f}s cooldown). Will retry automatically on next request."
            elif err == "network_error":
                return "Network connection issue. Please check internet access or proxy."
            elif err:
                return f"Provider error ({err}). Operating in verified offline fallback mode."
            return ""


circuit_breaker = GeminiCircuitBreaker()


def parse_retry_delay(exc: Exception) -> float:
    """Extract retry delay in seconds from exception headers or message."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {})
        if isinstance(headers, dict):
            for h in ("retry-after", "Retry-After", "x-ratelimit-reset"):
                if h in headers:
                    val = headers[h]
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass

    err_str = str(exc).lower()
    m = re.search(r'retry\s+(?:after|in)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?', err_str)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    m2 = re.search(r'retry_delay[\s:=]+(\d+(?:\.\d+)?)', err_str)
    if m2:
        try:
            return float(m2.group(1))
        except (ValueError, TypeError):
            pass
    return 0.0


_QUERY_CACHE: dict[str, dict[str, Any]] = {}
_QUERY_CACHE_MAX = 50


def clear_query_cache() -> None:
    """Clear in-memory query response cache."""
    _QUERY_CACHE.clear()


def get_advisor_status() -> dict[str, Any]:
    """Get current status of Gemini client, circuit breaker, and active model."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = (os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    return {
        "configured": bool(api_key and len(api_key) > 4),
        "model": model,
        "circuit_state": circuit_breaker.state,
        "cooldown_remaining": round(circuit_breaker.get_remaining_cooldown(), 1),
        "last_error_type": circuit_breaker.last_error_type,
        "recovery_hint": circuit_breaker.get_recovery_hint(),
    }


class _KeyDriverSchema(BaseModel):
    feature: str
    direction: Literal["above_overall", "below_overall"]
    value_cited: float | str | int


class _SegmentAdviceResponseSchema(BaseModel):
    segment: str
    summary: str
    key_drivers: list[_KeyDriverSchema]
    risks: list[str]
    actions: list[str]
    mode: str = "gemini_live"


def strip_json_fences(text: str) -> str:
    return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)


def _safe_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Ensure customer-level personal identifiers and order details never reach any LLM."""
    blocked_exact = {
        "customer_id", "customer_unique_id", "email", "name", "order_id",
        "invoice", "invoiceno", "phone", "address", "postal_code",
        "first_name", "last_name", "full_name", "user_name", "username",
        "zip", "zip_code", "postcode", "street", "city", "state", "country",
        "mobile", "telephone", "cell", "cellphone", "fax",
        "cpf", "ssn", "tax_id", "taxid", "national_id", "id_number",
        "credit_card", "card_number", "card", "cvv", "ip_address", "device_id",
    }
    blocked_substr = [
        "customer", "order", "invoice", "email", "phone", "mobile", "address",
        "street", "postal", "zip", "cpf", "ssn", "tax", "card", "device", "ip_",
    ]
    safe: dict[str, Any] = {}
    for key, value in profile.items():
        k_lower = str(key).strip().lower()
        if k_lower in blocked_exact:
            continue
        if k_lower not in ("segment_name", "customer_count", "total_customers", "segment_count", "segments_summary"):
            if "name" in k_lower or any(b in k_lower for b in blocked_substr):
                continue
        # Value-level safety check for strings: reject if string value matches PII patterns
        if isinstance(value, str) and k_lower not in ("segment_name", "recommendation"):
            val_clean = value.strip()
            if any(re.search(pat, val_clean, flags=re.I) for pat in [
                r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
                r"\b\d{3}-\d{2}-\d{4}\b",
                r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
                r"\b[0-9a-f]{24,36}\b",
                r"(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s])?\d{3,4}[-.\s]\d{4}\b",
            ]):
                continue
        elif isinstance(value, list):
            safe_items = []
            for item in value:
                if isinstance(item, dict):
                    safe_items.append(_safe_profile(item))
                elif isinstance(item, str):
                    if not any(re.search(pat, item.strip(), flags=re.I) for pat in [
                        r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
                        r"\b\d{3}-\d{2}-\d{4}\b",
                        r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
                        r"\b[0-9a-f]{24,36}\b",
                        r"(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s])?\d{3,4}[-.\s]\d{4}\b",
                    ]):
                        safe_items.append(item)
                else:
                    safe_items.append(item)
            value = safe_items
        elif isinstance(value, dict):
            value = _safe_profile(value)
        safe[key] = value
    return safe


def sanitize_chat_question(question: str) -> str:
    """Keep free-text chat aggregate-only before it can reach an API provider."""
    cleaned = " ".join(str(question).split())[:500]
    unsafe_patterns = (
        # Email addresses
        r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
        # Direct customer, client, user, order, invoice, account identifiers
        r"\b(customer|client|user|order|invoice|account)\s*(?:(?:id|number|no|code|#)\s*[:#-]?\s*[a-z0-9-]{2,}|[:#-]\s*[a-z0-9-]{2,}|\s+(?=[a-z0-9-]*\d)[a-z0-9-]{3,})\b",
        # Standalone UUIDs
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        # Long hex identifiers (e.g. 32-char hex customer/order IDs)
        r"\b[0-9a-f]{24,36}\b",
        # Phone numbers with formatting (e.g. +1-555-0199, (555) 012-3456, +44 20 7946 0958)
        r"(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s])?\d{3,4}[-.\s]\d{4}\b|\+\d{9,15}\b",
        # SSN / National IDs
        r"\b\d{3}-\d{2}-\d{4}\b",
        # Brazilian CPF format
        r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        # Credit cards (16 digits separated by - or space)
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    )
    if any(re.search(pattern, cleaned, flags=re.I) for pattern in unsafe_patterns):
        raise ValueError("Customer identifiers and email addresses cannot be sent to the advisor.")
    return cleaned


def validate_advice(payload: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    if not REQUIRED_KEYS.issubset(payload) or not isinstance(payload["key_drivers"], list) or not isinstance(payload["actions"], list):
        raise ValueError("invalid_schema")
    # Enforce exactly 3 actions for downstream schema and benchmark compatibility when actions are provided
    if payload["actions"]:
        if len(payload["actions"]) > 3:
            payload["actions"] = payload["actions"][:3]
        while len(payload["actions"]) < 3:
            payload["actions"].append("Review aggregate segment metrics and measure incremental outcomes.")

    if profile:
        profile_lower = {str(k).lower(): k for k in profile.keys()}
        # If profile has nested segment summaries (e.g. for All Segments Overview), include their keys
        if "Segments_Summary" in profile and isinstance(profile["Segments_Summary"], list):
            for sub_p in profile["Segments_Summary"]:
                if isinstance(sub_p, dict):
                    for k in sub_p.keys():
                        profile_lower[str(k).lower()] = k

        standard_features = {
            "recency_days": "Recency_Days",
            "frequency": "Frequency",
            "monetary": "Monetary",
            "tenure_days": "Tenure_Days",
            "unique_products": "Unique_Products",
            "average_order_value": "Average_Order_Value",
            "average_basket_size": "Average_Basket_Size",
            "return_rate": "Return_Rate",
            "customer_count": "Customer_Count",
            "total_customers": "Total_Customers",
            "segment_count": "Segment_Count",
            "cluster": "Cluster",
        }
        for sf_k, sf_v in standard_features.items():
            if sf_k not in profile_lower:
                profile_lower[sf_k] = sf_v

        fallback_feature = next((k for k in profile if k not in ("Segment_Name", "Segments_Summary", "Recommendation")), "Recency_Days")

        for driver in payload["key_drivers"]:
            if not {"feature", "direction", "value_cited"}.issubset(driver):
                raise ValueError("invalid_driver")
            feat = str(driver["feature"]).strip()
            feat_lower = feat.lower()
            if feat in profile:
                driver["feature"] = feat
            elif feat_lower in profile_lower:
                driver["feature"] = profile_lower[feat_lower]
            else:
                driver["feature"] = fallback_feature

            dir_val = str(driver.get("direction", "")).strip().lower()
            if "above" in dir_val or "high" in dir_val:
                driver["direction"] = "above_overall"
            elif "below" in dir_val or "low" in dir_val:
                driver["direction"] = "below_overall"
            elif dir_val in {"above_overall", "below_overall"}:
                driver["direction"] = dir_val
            else:
                driver["direction"] = "above_overall"
    return payload


def _load_dataset_metrics(dataset: str) -> dict[str, Any]:
    for ds in [dataset, "online_retail_ii", "olist"]:
        if ds and ds != "unknown":
            path = OUTPUTS_DIR / ds / "metrics.json"
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
    return {}


def _load_dataset_segment_profiles(dataset: str) -> list[dict[str, Any]]:
    """Load verified segment profiles CSV for a dataset."""
    for ds in [dataset, "online_retail_ii", "olist"]:
        if ds and ds != "unknown":
            path = OUTPUTS_DIR / ds / "segment_profiles.csv"
            if path.exists():
                try:
                    profiles = []
                    with path.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            item: dict[str, Any] = {}
                            for k, v in row.items():
                                try:
                                    if "." in v:
                                        item[k] = float(v)
                                    else:
                                        item[k] = int(v)
                                except (ValueError, TypeError):
                                    item[k] = v
                            profiles.append(item)
                    if profiles:
                        return profiles
                except Exception:
                    pass
    return []


SUBSTANTIVE_ANALYTICS_TRIGGERS = [
    "segment", "segments", "cohort", "cohorts", "cluster", "clusters",
    "customer", "customers", "recent engaged", "high value", "at-risk", "at risk",
    "repeat", "inactive", "spend", "spending", "spent", "monetary", "recency",
    "frequency", "order", "orders", "tenure", "aov", "basket", "return rate",
    "count", "counts", "metric", "metrics", "data", "dataset", "table",
    "statistics", "stats", "explain", "analyze", "analysis", "describe",
    "compare", "comparison", "difference", "versus", " vs ", " vs. ", "calculate",
    "break down", "breakdown", "summarize", "summary", "list", "show",
    "tell me about", "what is", "what are", "whats", "what's", "how many",
    "how much", "which segment", "which cohort", "why is", "why are",
    "retention", "retain", "churn", "loyalty", "campaign", "incentive",
    "discount", "win-back", "winback", "vip", "k-means", "kmeans", "gmm",
    "ward", "silhouette", "davies", "calinski", "ari", "stability", "algorithm", "model",
]


def classify_conversational_intent(text: str | None) -> str | None:
    """Deterministic greeting and conversational intent detection.

    Returns 'greeting', 'how_are_you', 'thanks', 'help', or None if the query
    is empty or contains substantive analytics / metric questions.
    """
    if not text or not text.strip():
        return None
    raw = text.strip().lower()

    # If query contains substantive analytics keywords or phrases, route to analytics
    for trigger in SUBSTANTIVE_ANALYTICS_TRIGGERS:
        if trigger in (" vs ", " vs. "):
            if trigger in raw:
                return None
        elif re.search(r"\b" + re.escape(trigger) + r"\b", raw):
            return None

    clean = re.sub(r"[^\w\s]", " ", raw)
    clean = re.sub(r"\s+", " ", clean).strip()

    if (
        re.match(r"^h+i+$", clean)
        or re.match(r"^h+e+y+$", clean)
        or re.match(r"^h+e+l+l+o+$", clean)
        or re.match(r"^(?:hi|hii|hello|hey|greetings|hola|yo)(?:\s+(?:there|assistant|bot|ai|loyalty\s+assistant|team|all|friend))?$", clean)
        or re.match(r"^good\s+(?:morning|afternoon|evening|day)(?:\s+(?:there|assistant|bot|ai|all))?$", clean)
    ):
        return "greeting"

    if (
        re.match(r"^how\s+(?:are\s+you|are\s+u|r\s+u|is\s+it\s+going|are\s+things|do\s+you\s+do)(?:\s+(?:doing|today|there|assistant))?$", clean)
        or clean in ("hows it going", "whats up", "what s up", "sup")
    ):
        return "how_are_you"

    if (
        re.match(r"^(?:thanks|thank\s+you|thx|ty|many\s+thanks)(?:\s+(?:a\s+lot|so\s+much|very\s+much|assistant))?$", clean)
    ):
        return "thanks"

    if clean in ("help", "what can you do", "who are you", "capabilities", "what do you do"):
        return "help"

    return None


def _build_conversational_response(intent: str, safe: dict[str, Any]) -> dict[str, Any]:
    segment_name = safe.get("Segment_Name", "Loyalty Assistant")
    if intent == "how_are_you":
        summary = (
            "I'm doing well, thank you! I'm ready to help you analyze customer segments, "
            "behavioral patterns, and loyalty strategies. What can I help you analyze today?"
        )
    elif intent == "thanks":
        summary = (
            "You're welcome! Let me know if you need any more insights on customer segments, "
            "metrics, or retention strategies."
        )
    elif intent == "help":
        summary = (
            "I'm your Loyalty AI Assistant. I can help you with:\n"
            "- Exploring customer segments (Recency, Frequency, Monetary metrics)\n"
            "- Evaluating clustering models (K-Means, GMM, Ward, Silhouette scores)\n"
            "- Analyzing churn risks and campaign retention opportunities\n"
            "- Comparing customer cohorts and spending distributions\n\n"
            "What would you like to know?"
        )
    else:  # greeting ("hi", "hii", "hello", "hey", "good morning", etc.)
        summary = (
            "Hi! I'm your Loyalty AI Assistant. "
            "I can help you explore customer segments, engagement, retention risks, "
            "and campaign opportunities. What would you like to know?"
        )

    return validate_advice({
        "segment": segment_name,
        "summary": summary,
        "key_drivers": [],
        "risks": [],
        "actions": [],
        "mode": "conversational",
        "status_detail": "conversational",
    }, safe)


def _classify_fallback_intent(question: str | None) -> str:
    if not question:
        return "general_recommendation"
    conv = classify_conversational_intent(question)
    if conv:
        return conv

    q = question.lower()

    # Case C: Clustering / model evaluation
    cluster_keywords = (
        "cluster", "clustering", "algorithm", "silhouette", "davies", "calinski",
        "ari", "adjusted rand", "stability", "k-means", "kmeans", "gmm",
        "gaussian mixture", "ward", "model evaluation", "optimal k", "elbow",
    )
    if any(k in q for k in cluster_keywords) and not any(k in q for k in ("retention", "campaign", "incentive", "win-back")):
        return "model_evaluation"

    # Case B: Segment comparison
    compare_keywords = (
        "compare", "comparison", "difference between", "versus", " vs ", " vs. ",
        "highest spending", "lowest spending", "highest monetary", "lowest monetary",
        "most active", "most inactive", "highest frequency", "lowest frequency",
        "longest tenure", "most recent", "rank", "ranking",
    )
    if any(k in q for k in compare_keywords):
        return "segment_comparison"

    # Case F: Segment explanation (e.g., "explain Recent Engaged Customers", "explain high-risk segment")
    explanation_keywords = ("explain", "describe", "tell me about", "profile", "who are", "breakdown", "what is the", "details of")
    segment_names = (
        "recent engaged", "recent", "engaged",
        "high value", "high-value", "repeat customers", "vip",
        "at-risk", "at risk", "high-risk", "high risk", "highest risk", "inactive", "inactive customers",
        "risk", "churn",
    )
    if any(k in q for k in explanation_keywords) and any(k in q for k in segment_names + ("segment", "cohort", "cluster")):
        return "segment_explanation"

    # Case A: Exact segment metrics & metric queries (e.g. "what is the average spend?")
    metric_keywords = (
        "spend", "spending", "spent", "average spend", "monetary", "monetary value",
        "recency", "frequency", "customer count", "order count", "orders", "aov",
        "average order value", "basket", "tenure", "return rate",
        "metric", "metrics", "table", "list all", "statistics", "stats", "exact average",
        "show segments", "all three", "overview", "numbers",
    )
    if any(k in q for k in metric_keywords):
        return "segment_metrics"

    # Case D: Retention / loyalty recommendations
    retention_keywords = (
        "retention", "retain", "churn", "loyalty", "recommend", "action",
        "strategy", "campaign", "incentive", "win-back", "winback", "vip",
    )
    if any(k in q for k in retention_keywords):
        return "retention_strategy"

    # Case E: Unsupported / overly broad
    unsupported_indicators = (
        "who is", "what is the capital", "weather", "poem", "joke", "code",
        "python", "president", "stock price", "forecast 203", "ceo", "song",
        "translate", "write an essay", "how old",
    )
    if any(k in q for k in unsupported_indicators):
        return "unsupported"

    return "general_recommendation"


def _format_metrics_table(segments_summary: list[dict[str, Any]]) -> str:
    lines = [
        "| Customer Segment | Customer Count | Average Recency (Days) | Purchase Frequency | Average Monetary Value |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for s in segments_summary:
        if isinstance(s, dict):
            s_name = s.get("Segment_Name", "Segment")
            cnt = f"{s['Customer_Count']:,}" if s.get("Customer_Count") is not None else "unavailable"
            rec = f"{float(s['Recency_Days']):.1f} days" if s.get("Recency_Days") is not None else "unavailable"
            freq = f"{float(s['Frequency']):.2f} orders" if s.get("Frequency") is not None else "unavailable"
            mon = f"${float(s['Monetary']):,.2f}" if s.get("Monetary") is not None else "unavailable"
            lines.append(f"| {s_name} | {cnt} | {rec} | {freq} | {mon} |")
    return "\n".join(lines)


def _build_comparison_summary(segments_summary: list[dict[str, Any]]) -> tuple[str, list[str]]:
    valid_segs = [s for s in segments_summary if isinstance(s, dict) and "Segment_Name" in s]
    if not valid_segs:
        return "No segment summary data available to perform comparison.", []

    by_monetary = sorted(valid_segs, key=lambda s: float(s.get("Monetary", 0) or 0), reverse=True)
    by_recency = sorted(valid_segs, key=lambda s: float(s.get("Recency_Days", 0) or 0))
    by_freq = sorted(valid_segs, key=lambda s: float(s.get("Frequency", 0) or 0), reverse=True)

    highest_spend = by_monetary[0]
    lowest_spend = by_monetary[-1]
    most_recent = by_recency[0]
    least_recent = by_recency[-1]
    highest_freq = by_freq[0]
    lowest_freq = by_freq[-1]

    table = _format_metrics_table(valid_segs)

    summary = (
        "### Segment Comparison (Verified Aggregate Metrics)\n\n"
        f"{table}\n\n"
        "**Key Comparative Insights:**\n"
        f"- **Monetary Spending**: **{highest_spend.get('Segment_Name')}** demonstrates the highest average monetary spending at **${float(highest_spend.get('Monetary', 0)):,.2f}**, whereas **{lowest_spend.get('Segment_Name')}** has the lowest average spending at **${float(lowest_spend.get('Monetary', 0)):,.2f}**.\n"
        f"- **Purchase Frequency**: **{highest_freq.get('Segment_Name')}** leads in order frequency with an average of **{float(highest_freq.get('Frequency', 0)):.2f} orders**, compared to **{float(lowest_freq.get('Frequency', 0)):.2f} orders** for **{lowest_freq.get('Segment_Name')}**.\n"
        f"- **Recency & Activity**: **{most_recent.get('Segment_Name')}** is the most recently active cohort (average **{float(most_recent.get('Recency_Days', 0)):.1f} days** since last purchase), while **{least_recent.get('Segment_Name')}** exhibits the longest inactivity at **{float(least_recent.get('Recency_Days', 0)):.1f} days**, indicating elevated churn risk."
    )
    actions = [
        f"Protect revenue from {highest_spend.get('Segment_Name')} with VIP retention privileges.",
        f"Nurture purchase frequency for mid-tier cohorts through category cross-sell.",
        f"Deploy targeted win-back re-engagement workflows for {least_recent.get('Segment_Name')}.",
    ]
    return summary, actions


def _build_model_evaluation_summary(metrics_data: dict[str, Any], dataset: str) -> tuple[str, list[str]]:
    if not metrics_data:
        return f"Model evaluation metrics for dataset '{dataset}' are currently unavailable.", []

    algo = metrics_data.get("selected_algorithm", "Unknown")
    k = metrics_data.get("selected_k", "N/A")
    cust_count = metrics_data.get("customer_count", "N/A")
    cust_str = f"{cust_count:,}" if isinstance(cust_count, int) else str(cust_count)
    cand = metrics_data.get("selected_candidate", {})
    sil = cand.get("silhouette")
    db = cand.get("davies_bouldin")
    ch = cand.get("calinski_harabasz")
    mean_ari = cand.get("mean_ari")
    min_ari = cand.get("min_ari")
    rule = metrics_data.get("selection_rule", "Highest silhouette across K=3..6; candidates within 0.01 use mean ARI then deterministic ordering.")
    features = metrics_data.get("model_features", [])

    sil_str = f"{float(sil):.4f}" if sil is not None else "unavailable"
    db_str = f"{float(db):.4f}" if db is not None else "unavailable"
    ch_str = f"{float(ch):.2f}" if ch is not None else "unavailable"
    ari_str = f"{float(mean_ari):.4f}" if mean_ari is not None else "unavailable"
    min_ari_str = f"{float(min_ari):.4f}" if min_ari is not None else "unavailable"

    summary = (
        f"### Clustering Model Evaluation ({dataset.replace('_', ' ').title()})\n\n"
        f"- **Deployed Algorithm**: **{algo}**\n"
        f"- **Optimal Cluster Count (K)**: **K = {k}**\n"
        f"- **Total Evaluated Customers**: **{cust_str}**\n"
        f"- **Silhouette Score**: **{sil_str}** (higher indicates denser, better-separated clusters)\n"
        f"- **Davies-Bouldin Index**: **{db_str}** (lower indicates better clustering separation)\n"
        f"- **Calinski-Harabasz Score**: **{ch_str}** (ratio of between-cluster to within-cluster dispersion)\n"
        f"- **Stability (Mean ARI)**: **{ari_str}** (minimum ARI: {min_ari_str} across stability bootstrap runs)\n"
        f"- **Model Selection Rule**: {rule}\n"
        f"- **Evaluated Features**: {', '.join(features) if features else 'unavailable'}\n\n"
        "*Note: These metrics reflect the actual reproducible pipeline evaluation executed on the training partition.*"
    )
    actions = [
        "Monitor silhouette and ARI stability on future quarterly transaction partitions.",
        "Evaluate whether adding tenure or category diversity improves cluster separation.",
        "Maintain current K=3 clustering solution for production dashboard reporting.",
    ]
    return summary, actions


def _build_retention_summary(safe: dict[str, Any], segments_summary: list[dict[str, Any]] | None) -> tuple[str, list[str]]:
    name = str(safe.get("Segment_Name", "Selected segment"))
    recency = float(safe.get("Recency_Days", 0) or 0)
    monetary = float(safe.get("Monetary", 0) or 0)
    freq = float(safe.get("Frequency", 0) or 0)

    is_overview = "overview" in name.lower() or "all segments" in name.lower()

    if is_overview and segments_summary:
        cohort_bullets = []
        for seg in segments_summary:
            s_name = seg.get("Segment_Name", "Unknown Cohort")
            s_mon = float(seg.get("Monetary", 0) or 0)
            s_rec = float(seg.get("Recency_Days", 0) or 0)
            s_freq = float(seg.get("Frequency", 0) or 0)
            s_cnt = seg.get("Customer_Count")
            cnt_str = f"{s_cnt:,} customers" if isinstance(s_cnt, int) else ""
            stats_str = f"average spend **${s_mon:,.2f}**, frequency **{s_freq:.2f}**, recency **{s_rec:.1f} days**"
            if cnt_str:
                stats_str += f" ({cnt_str})"

            hypo = (
                "Deploy automated win-back workflows, time-limited reactivation discounts, and churn exit surveys."
                if s_rec > 180
                else "Trigger replenishment prompts, category discovery recommendations, and point multipliers to cultivate habit."
                if s_rec > 90
                else "Establish exclusive VIP loyalty tier perks, early product drops, and priority support to protect high-margin revenue."
            )
            cohort_bullets.append(
                f"- **{s_name}**:\n"
                f"  - *Verified Metrics*: {stats_str}.\n"
                f"  - *Retention Hypothesis*: {hypo}"
            )

        cohort_text = "\n\n".join(cohort_bullets)
        summary = (
            "### Customer Retention Strategy by Segment\n\n"
            "#### Verified Dataset Metrics\n"
            "Based on aggregate clustering data from the active dataset, segments exhibit distinct behavioral profiles:\n\n"
            f"{cohort_text}\n\n"
            "> **Disclaimer**: *Verified metrics reflect aggregate clustering data. Retention actions are general strategic suggestions to validate experimentally via randomized testing.*"
        )
        actions = [
            "Prioritize VIP retention incentives for High Value Repeat Customers.",
            "Deploy replenishment and habit-forming triggers for Recent Engaged Customers.",
            "Test low-cost automated win-back workflows for At-Risk Inactive Customers.",
        ]
    else:
        extra_metrics = []
        for k in ["Average_Order_Value", "Tenure_Days", "Return_Rate"]:
            if k in safe and safe[k] is not None:
                val = safe[k]
                lbl = k.replace("_", " ")
                if isinstance(val, (int, float)):
                    extra_metrics.append(f"- **{lbl}**: {val:,.2f}" if isinstance(val, float) else f"- **{lbl}**: {val:,}")

        extra_str = ("\n" + "\n".join(extra_metrics)) if extra_metrics else ""
        campaign_focus = (
            "Re-engagement and reactivation campaign"
            if recency > 180
            else "Habit building and cross-sell campaign"
            if recency > 90
            else "VIP retention campaign and lifetime value optimization"
        )

        summary = (
            f"### Customer Retention Strategy for {name}\n\n"
            f"#### Verified Dataset Metrics\n"
            f"- **Average Recency**: **{recency:.1f} days**\n"
            f"- **Order Frequency**: **{freq:.2f} orders**\n"
            f"- **Average Spend (Monetary)**: **${monetary:,.2f}**"
            f"{extra_str}\n\n"
            f"#### Strategic Hypotheses & Suggestions\n"
            f"- **Suggested Campaign Focus**: {campaign_focus}\n\n"
            f"> **Disclaimer**: *Metrics above reflect verified cluster averages from the dataset. Recommendations are general strategic hypotheses to validate experimentally.*"
        )
        if recency > 180:
            actions = [
                "Implement automated win-back discount sequences within 14 days.",
                "Send brief exit feedback surveys to identify churn drivers.",
                "Cap retention spending to avoid negative ROI on low-probability reactivations.",
            ]
        elif recency > 90:
            actions = [
                "Trigger personalized next-order reminders based on average inter-purchase interval.",
                "Offer category expansion incentives for complementary product purchases.",
                "Measure incremental conversion lift against an uncontacted holdout group.",
            ]
        else:
            actions = [
                "Provide dedicated account perks and premium loyalty tier benefits.",
                "Offer sneak peeks and early access to new product collections.",
                "Gather product feedback directly from high-value repeat shoppers.",
            ]
    return summary, actions


def _build_unsupported_summary(question: str) -> tuple[str, list[str]]:
    summary = (
        "### Local Fallback Notice: Query Outside Offline Scope\n\n"
        f"The question *\"{question}\"* cannot be answered in deterministic local fallback mode without live AI.\n\n"
        "**Available Fallback Capabilities:**\n"
        "- Exact customer segment metrics (Recency, Frequency, Monetary, Customer Counts)\n"
        "- Cross-segment comparisons and spending rankings\n"
        "- Clustering model evaluation (K-Means, GMM, Ward, Silhouette, Davies-Bouldin, ARI)\n"
        "- Data-driven customer loyalty and retention strategy hypotheses\n\n"
        "*To answer open-domain or generative questions, please verify internet connectivity and ensure your Gemini API quota is active.*"
    )
    actions = [
        "Ask for exact segment metrics: 'List all segments with their recency, frequency, and monetary value in a table.'",
        "Ask for segment comparisons: 'Compare customer segments by average spending.'",
        "Ask for model evaluation: 'What algorithm was selected and what is the silhouette score?'",
    ]
    return summary, actions


def _find_risk_segment(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the high-risk / at-risk customer segment from a list of segment profiles."""
    if not segments:
        return None
    # 1. Direct name matches for at-risk, inactive, churn, dormant, or risk
    for s in segments:
        s_name = str(s.get("Segment_Name", "")).lower()
        if any(term in s_name for term in ("at-risk", "at risk", "inactive", "churn", "dormant", "risk")):
            return s
    # 2. Behavioral identification: segment with highest recency (> 90 days indicates longest dormancy / churn risk)
    valid_with_recency = [s for s in segments if s.get("Recency_Days") is not None]
    if valid_with_recency:
        highest_recency_seg = max(valid_with_recency, key=lambda s: float(s.get("Recency_Days", 0) or 0))
        if float(highest_recency_seg.get("Recency_Days", 0) or 0) > 90:
            return highest_recency_seg
    return None


def _extract_target_segment(
    question: str | None,
    safe: dict[str, Any],
    dataset: str = "unknown",
) -> tuple[dict[str, Any] | None, str]:
    """Find target segment data from safe profile or dataset profiles based on user question.

    Returns:
        tuple[matched_segment_dict or None, status_string]
        status_string can be "matched", "overview", "high_risk_not_found", or "segment_not_found:<name>"
    """
    seg_summary_list = safe.get("Segments_Summary")
    if not isinstance(seg_summary_list, list) or not seg_summary_list:
        seg_summary_list = _load_dataset_segment_profiles(dataset)

    if not question or not question.strip():
        if safe.get("Segment_Name") and safe.get("Segment_Name") != "All Segments (Overview)":
            return safe, "matched"
        return safe, "overview"

    q_low = question.lower().strip()

    # 1. High-risk / churn / inactive segment intent detection
    risk_patterns = [
        r"\bhigh[- ]?risk\b",
        r"\bhighest[- ]?risk\b",
        r"\bat[- ]?risk\b",
        r"\bchurn[- ]?risk\b",
        r"\belevated[- ]?risk\b",
        r"\bchurn\b",
        r"\binactive\b",
        r"\bdormant\b",
        r"\briskiest\b",
    ]
    if any(re.search(pat, q_low) for pat in risk_patterns):
        risk_seg = _find_risk_segment(seg_summary_list)
        if risk_seg:
            return risk_seg, "matched"
        return None, "high_risk_not_found"

    # 2. High-value / VIP / top-spending segment intent detection
    high_val_patterns = [
        r"\bhigh[- ]?value\b",
        r"\bvip\b",
        r"\btop[- ]?spenders?\b",
        r"\bhighest[- ]?spend(?:ing)?\b",
        r"\brepeat[- ]?customers?\b",
        r"\bloyal[- ]?customers?\b",
        r"\bpremium\b",
    ]
    if any(re.search(pat, q_low) for pat in high_val_patterns):
        if seg_summary_list:
            for s in seg_summary_list:
                s_name = str(s.get("Segment_Name", "")).lower()
                if any(k in s_name for k in ("high value", "repeat", "loyal", "vip")):
                    return s, "matched"
            # Fallback to segment with highest monetary spend
            valid_mon = [s for s in seg_summary_list if s.get("Monetary") is not None]
            if valid_mon:
                return max(valid_mon, key=lambda s: float(s.get("Monetary", 0) or 0)), "matched"

    # 3. Recent / engaged customer segment intent detection
    recent_patterns = [
        r"\brecent[- ]?engaged\b",
        r"\bnewly[- ]?engaged\b",
        r"\brecent[- ]?customers?\b",
        r"\bactive[- ]?shoppers?\b",
        r"\bengaged\b",
        r"\bmost[- ]?recent\b",
    ]
    if any(re.search(pat, q_low) for pat in recent_patterns):
        if seg_summary_list:
            for s in seg_summary_list:
                s_name = str(s.get("Segment_Name", "")).lower()
                if "recent" in s_name or "engaged" in s_name:
                    return s, "matched"
            valid_rec = [s for s in seg_summary_list if s.get("Recency_Days") is not None]
            if valid_rec:
                return min(valid_rec, key=lambda s: float(s.get("Recency_Days", 0) or 0)), "matched"

    # 4. Direct segment name matching from available profiles
    if seg_summary_list:
        for s in seg_summary_list:
            s_name = str(s.get("Segment_Name", "")).lower()
            if s_name and (s_name in q_low or all(w in q_low for w in s_name.split() if len(w) > 3)):
                return s, "matched"

    # 5. Check if question explicitly requests a specific non-existent segment name
    explicit_seg_match = re.search(
        r'\b(?:explain|describe|tell me about|profile|breakdown)\s+(?:the\s+)?([a-z0-9_-]+(?:\s+[a-z0-9_-]+)?)\s+(?:segment|cohort|group|tier)\b',
        q_low,
    )
    if explicit_seg_match:
        requested = explicit_seg_match.group(1).strip()
        if requested not in ("all", "each", "every", "the", "current", "customer", "our", "this"):
            return None, f"segment_not_found:{requested}"

    # 6. Overall / default behavior
    if safe.get("Segment_Name") and safe.get("Segment_Name") != "All Segments (Overview)":
        return safe, "matched"
    return safe, "overview"


def _build_segment_explanation(
    target_seg: dict[str, Any] | None,
    match_status: str,
    question: str | None,
    dataset: str,
    segments_summary: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    ds_title = dataset.replace('_', ' ').title() if dataset != 'unknown' else 'Current Dataset'
    all_segs = segments_summary or _load_dataset_segment_profiles(dataset)

    # Case 1: High-risk segment requested, but not defined in dataset
    if match_status == "high_risk_not_found":
        available_names = [str(s.get("Segment_Name", "Unknown")) for s in all_segs if s.get("Segment_Name")]
        summary = (
            f"### High-Risk Segment Notice ({ds_title})\n\n"
            f"The active dataset does not contain a defined high-risk customer segment. "
            f"The verified customer segments identified in this dataset are:\n\n"
            + "\n".join(f"- **{n}**" for n in available_names)
            + "\n\n*Note: High-risk cohorts are typically characterized by extended inactivity (high recency) and elevated churn probability.*"
        )
        actions = [
            "Review customer recency distributions to identify potential churn thresholds.",
            "Inspect engagement patterns across existing defined segments.",
            "Evaluate whether retraining with K=4 isolates a distinct high-churn risk cohort.",
        ]
        return summary, actions

    # Case 2: Specific requested segment name not found in dataset
    if match_status.startswith("segment_not_found:"):
        requested_name = match_status.split(":", 1)[1].strip()
        available_names = [str(s.get("Segment_Name", "Unknown")) for s in all_segs if s.get("Segment_Name")]
        summary = (
            f"### Segment Not Found ({ds_title})\n\n"
            f"The active dataset does not contain a customer segment named **\"{requested_name.title()}\"**.\n\n"
            f"The verified customer segments available in this dataset are:\n\n"
            + "\n".join(f"- **{n}**" for n in available_names)
            + "\n\n*Please select or ask about one of the available segments listed above.*"
        )
        actions = [
            "Select one of the verified dataset segments for detailed analysis.",
            "Review cluster profiles to explore behavioral differences across segments.",
            "Compare aggregate segment metrics in the Segments tab.",
        ]
        return summary, actions

    # Case 3: Overview requested (or fallback to overview)
    if target_seg is None or match_status == "overview" or target_seg.get("Segment_Name") == "All Segments (Overview)":
        target = target_seg or {}
        recency = float(target.get("Recency_Days", 0) or 0)
        monetary = float(target.get("Monetary", 0) or 0)
        freq = float(target.get("Frequency", 0) or 0)
        total_cust = target.get("Total_Customers") or target.get("Customer_Count")

        # If overview averages are not precomputed in target, calculate from all_segs
        if (recency == 0.0 and monetary == 0.0 and freq == 0.0) and all_segs:
            total_c = sum(int(s.get("Customer_Count", 0) or 0) for s in all_segs)
            if total_c > 0:
                recency = sum(float(s.get("Recency_Days", 0) or 0) * int(s.get("Customer_Count", 0) or 0) for s in all_segs) / total_c
                freq = sum(float(s.get("Frequency", 0) or 0) * int(s.get("Customer_Count", 0) or 0) for s in all_segs) / total_c
                monetary = sum(float(s.get("Monetary", 0) or 0) * int(s.get("Customer_Count", 0) or 0) for s in all_segs) / total_c
                total_cust = total_c

        cnt_str = f"**{total_cust:,} total customers**" if isinstance(total_cust, (int, float)) and total_cust > 0 else "Verified population"

        available_names = [str(s.get("Segment_Name", "Unknown")) for s in all_segs if s.get("Segment_Name")]
        seg_list_text = ", ".join(f"**{n}**" for n in available_names) if available_names else "behavioral clusters"

        summary = (
            f"### Segment Profile: All Segments (Overall Dataset Weighted Averages)\n\n"
            f"> **Important Notice**: *The metrics below reflect overall weighted averages across all customer segments in the active dataset ({ds_title}), not a single individual segment.*\n\n"
            f"#### Overall Population Metrics\n"
            f"- **Customer Population**: {cnt_str}\n"
            f"- **Weighted Average Recency**: **{recency:.1f} days**\n"
            f"- **Weighted Purchase Frequency**: **{freq:.2f} orders**\n"
            f"- **Weighted Average Spend**: **${monetary:,.2f}**\n\n"
            f"#### Identified Customer Segments\n"
            f"This dataset contains {len(available_names)} distinct behavioral cohorts: {seg_list_text}.\n\n"
            f"> *To view segment-specific metrics, specify an individual segment such as the high-risk, recent engaged, or high value segment.*"
        )
        actions = [
            "Ask about a specific segment: 'Explain the high-risk customer segment in the current dataset.'",
            "Ask for segment comparisons: 'Compare customer segments by average spending.'",
            "Review individual segment KPIs in the dashboard Segments view.",
        ]
        return summary, actions

    # Case 4: Target segment matched (e.g. At-Risk Inactive Customers)
    name = str(target_seg.get("Segment_Name", "Selected segment"))
    recency = float(target_seg.get("Recency_Days", 0) or 0)
    monetary = float(target_seg.get("Monetary", 0) or 0)
    freq = float(target_seg.get("Frequency", 0) or 0)
    cnt = target_seg.get("Customer_Count")
    cnt_str = f"**{cnt:,} customers**" if isinstance(cnt, int) else "Verified population"

    if any(k in name.lower() for k in ("at-risk", "inactive", "risk", "churn")):
        behavioral_desc = (
            "Customers with prolonged inactivity since their last transaction. "
            "They exhibit low purchase frequency and elevated churn probability, "
            "representing customers at risk of permanently churning."
        )
        opp = (
            "Deploy low-cost automated win-back workflows, service recovery prompts, "
            "and exit surveys to reactivate salvageable customers without excessive margin erosion."
        )
        actions = [
            "Implement automated win-back discount sequences within 14 days.",
            "Send brief exit feedback surveys to identify churn drivers.",
            "Cap retention spending to avoid negative ROI on low-probability reactivations.",
        ]
    elif "recent" in name.lower() or "engaged" in name.lower():
        behavioral_desc = (
            "Active repeat shoppers with strong engagement and short intervals between purchases. "
            "They demonstrate brand affinity and consistent purchase momentum."
        )
        opp = (
            "Nurture into High Value VIPs through replenishment prompts, loyalty rewards, "
            "and category discovery cross-sells."
        )
        actions = [
            "Trigger automated replenishment reminders based on average repurchase cycles.",
            "Introduce category discovery cross-sells to expand basket variety.",
            "Offer loyalty milestone incentives to accelerate progression to high-value tier.",
        ]
    elif "high value" in name.lower() or "repeat" in name.lower() or "vip" in name.lower():
        behavioral_desc = (
            "Top-tier premium customers driving substantial total monetary revenue. "
            "They purchase frequently and exhibit high basket spend."
        )
        opp = (
            "Protect high-margin revenue through exclusive VIP perks, dedicated support, "
            "and early access to new product drops."
        )
        actions = [
            "Provide dedicated VIP concierge perks and priority fulfillment.",
            "Offer exclusive early access to premium product collections.",
            "Conduct quarterly customer advisory feedback check-ins.",
        ]
    else:
        behavioral_desc = "Cluster cohort identified through reproducible unsupervised behavioral segmentation."
        opp = "Monitor purchase patterns and test tailored promotional triggers."
        actions = [
            "Review cohort spend and frequency metrics.",
            "Test tailored promotional messaging.",
            "Track incremental cohort revenue lift against holdout groups.",
        ]

    summary = (
        f"### Segment Profile: {name}\n\n"
        f"#### Verified Dataset Metrics\n"
        f"- **Customer Count**: {cnt_str}\n"
        f"- **Average Recency**: **{recency:.1f} days**\n"
        f"- **Purchase Frequency**: **{freq:.2f} orders**\n"
        f"- **Average Spend (Monetary)**: **${monetary:,.2f}**\n\n"
        f"#### Behavioral Characteristics & Strategic Opportunity\n"
        f"- **Profile**: {behavioral_desc}\n"
        f"- **Strategic Opportunity**: {opp}\n\n"
        f"> **Disclaimer**: *Metrics reflect verified cluster averages from the dataset ({ds_title}). Recommendations are strategic hypotheses for experimentation.*"
    )
    return summary, actions


def _build_metric_query_summary(
    safe: dict[str, Any],
    question: str | None,
    dataset: str,
) -> tuple[str, list[str]]:
    q_low = (question or "").lower()
    seg_summary_list = safe.get("Segments_Summary")
    name = str(safe.get("Segment_Name", "Selected segment"))

    is_all_table = (
        "table" in q_low
        or "list all" in q_low
        or "all customer segments" in q_low
        or "all segments" in q_low
        or sum(1 for m in ("recency", "frequency", "monetary") if m in q_low) >= 2
    )
    is_spend_query = any(k in q_low for k in ("spend", "spending", "monetary", "revenue", "dollar", "amount"))
    is_recency_query = any(k in q_low for k in ("recency", "days since", "last purchase", "recent"))
    is_freq_query = any(k in q_low for k in ("frequency", "orders", "purchases", "order count"))

    ds_label = dataset.replace('_', ' ').title() if dataset != 'unknown' else 'Dataset'

    if seg_summary_list and isinstance(seg_summary_list, list):
        if is_all_table:
            table = _format_metrics_table(seg_summary_list)
            summary = (
                f"### Aggregate Metrics for All Segments ({ds_label})\n\n"
                f"{table}"
            )
            actions = [
                "Prioritize high-monetary cohorts with exclusive VIP loyalty incentives.",
                "Re-engage moderate-recency customers using habit-building next-purchase prompts.",
                "Implement win-back automation for inactive segments exceeding baseline recency.",
            ]
            return summary, actions
        elif is_spend_query:
            bullets = []
            for s in seg_summary_list:
                s_name = s.get("Segment_Name", "Segment")
                s_mon = float(s.get("Monetary", 0) or 0)
                bullets.append(f"- **{s_name}**: average monetary spend of **${s_mon:,.2f}**")
            bullets_text = "\n".join(bullets)
            summary = (
                f"### Average Monetary Spend by Segment ({ds_label})\n\n"
                f"Based on verified dataset clustering metrics, here is the average spend for each customer segment:\n\n"
                f"{bullets_text}\n\n"
                f"*Note: Values represent verified cluster averages from the dataset.*"
            )
            actions = [
                "Target highest-spending cohorts with VIP loyalty privileges.",
                "Implement cross-sell triggers to raise average basket sizes for mid-tier cohorts.",
                "Review lifetime value contribution across segment spending tiers.",
            ]
            return summary, actions
        elif is_recency_query:
            bullets = [f"- **{s.get('Segment_Name')}**: average recency of **{float(s.get('Recency_Days', 0)):.1f} days**" for s in seg_summary_list]
            summary = f"### Average Recency by Segment ({ds_label})\n\n" + "\n".join(bullets)
            return summary, ["Automate win-back sequences for segments with recency > 180 days."]
        elif is_freq_query:
            bullets = [f"- **{s.get('Segment_Name')}**: average purchase frequency of **{float(s.get('Frequency', 0)):.2f} orders**" for s in seg_summary_list]
            summary = f"### Average Purchase Frequency by Segment ({ds_label})\n\n" + "\n".join(bullets)
            return summary, ["Design repurchase nudges to increase repeat order frequency."]
        else:
            table = _format_metrics_table(seg_summary_list)
            summary = (
                f"### Aggregate Metrics for All Segments ({ds_label})\n\n"
                f"{table}"
            )
            return summary, [
                "Prioritize high-monetary cohorts with exclusive VIP loyalty incentives.",
                "Re-engage moderate-recency customers using habit-building next-purchase prompts.",
                "Implement win-back automation for inactive segments exceeding baseline recency.",
            ]
    else:
        monetary = float(safe.get("Monetary", 0) or 0)
        recency = float(safe.get("Recency_Days", 0) or 0)
        freq = float(safe.get("Frequency", 0) or 0)
        if is_spend_query:
            summary = (
                f"### Average Monetary Spend for {name}\n\n"
                f"- **Average Spend (Monetary)**: **${monetary:,.2f}**\n"
                f"- **Order Frequency**: **{freq:.2f} orders**\n"
                f"- **Average Recency**: **{recency:.1f} days**\n\n"
                f"*Note: Reflects verified aggregate cluster metrics from the dataset.*"
            )
            actions = ["Review aggregate spending behavior.", "Test personalized cross-sell recommendations."]
            return summary, actions
        else:
            metric_items = [
                f"- **Average Spend (Monetary)**: **${monetary:,.2f}**",
                f"- **Order Frequency**: **{freq:.2f} orders**",
                f"- **Average Recency**: **{recency:.1f} days**",
            ]
            summary = f"### Verified Metrics for {name}\n\n" + "\n".join(metric_items)
            return summary, ["Review aggregate metrics and evaluate retention opportunities."]


def mock_advice(
    segment_profile: dict[str, Any],
    question: str | None = None,
    dataset: str = "unknown",
) -> dict[str, Any]:
    safe = _safe_profile(segment_profile)
    conversational_intent = classify_conversational_intent(question)
    if conversational_intent:
        return _build_conversational_response(conversational_intent, safe)

    name = str(safe.get("Segment_Name", "Selected segment"))
    feature = "Recency_Days" if "Recency_Days" in safe else next(iter(safe), "Segment_Name")
    direction = "above_overall" if float(safe.get("Recency_Days", 0) or 0) > 90 else "below_overall"

    wants_metrics_only = False
    if question:
        q_clean = question.lower()
        if (
            "metrics only" in q_clean
            or "only metrics" in q_clean
            or "only aggregate metrics" in q_clean
            or "no recommendation" in q_clean
            or "without recommendation" in q_clean
            or "no marketing" in q_clean
            or "omit marketing" in q_clean
            or ("omit" in q_clean and "recommendation" in q_clean)
            or ("do not" in q_clean and "recommendation" in q_clean)
            or ("don't" in q_clean and "recommendation" in q_clean)
            or ("never" in q_clean and "recommendation" in q_clean)
        ):
            wants_metrics_only = True

    intent = _classify_fallback_intent(question)
    seg_summary_list = safe.get("Segments_Summary") if isinstance(safe.get("Segments_Summary"), list) else None
    if not seg_summary_list:
        seg_summary_list = _load_dataset_segment_profiles(dataset)

    target_seg, match_status = _extract_target_segment(question, safe, dataset)
    if match_status == "matched" and target_seg:
        name = str(target_seg.get("Segment_Name", name))
        feature = "Recency_Days" if "Recency_Days" in target_seg else next(iter(target_seg), "Segment_Name")
        direction = "above_overall" if float(target_seg.get("Recency_Days", 0) or 0) > 90 else "below_overall"

    if intent == "unsupported" and question:
        summary, actions = _build_unsupported_summary(question)
    elif intent == "model_evaluation":
        metrics_data = _load_dataset_metrics(dataset)
        summary, actions = _build_model_evaluation_summary(metrics_data, dataset)
    elif intent == "segment_comparison" and seg_summary_list:
        summary, actions = _build_comparison_summary(seg_summary_list)
    elif intent == "segment_explanation":
        summary, actions = _build_segment_explanation(target_seg, match_status, question, dataset, seg_summary_list)
    elif intent == "segment_metrics":
        summary, actions = _build_metric_query_summary(target_seg or safe, question, dataset)
    elif intent in ("retention_strategy", "general_recommendation") and not wants_metrics_only:
        summary, actions = _build_retention_summary(target_seg or safe, seg_summary_list)
    elif seg_summary_list:
        # Case A: Exact segment metrics
        table = _format_metrics_table(seg_summary_list)
        summary = (
            f"### Aggregate Metrics for All Segments ({dataset.replace('_', ' ').title() if dataset != 'unknown' else 'Current Dataset'})\n\n"
            f"{table}"
        )
        actions = [
            "Prioritize high-monetary cohorts with exclusive VIP loyalty incentives.",
            "Re-engage moderate-recency customers using habit-building next-purchase prompts.",
            "Implement win-back automation for inactive segments exceeding baseline recency.",
        ]
    else:
        # Single segment metrics or general profile advice
        if wants_metrics_only:
            metric_items = []
            for k, v in safe.items():
                if k not in ("Segment_Name", "Segments_Summary", "Recommendation"):
                    if isinstance(v, float):
                        metric_items.append(f"- **{k}**: {v:,.2f}")
                    elif isinstance(v, int):
                        metric_items.append(f"- **{k}**: {v:,}")
                    else:
                        metric_items.append(f"- **{k}**: {v}")
            summary = (
                f"Available aggregate metrics for {name}:\n\n"
                + ("\n".join(metric_items) if metric_items else "No additional metrics available.")
            )
            actions = []
        else:
            summary = (
                f"Verified aggregate metrics for {name} regarding '{question}':"
                if question
                else f"Aggregate analysis for {name} based on verified cluster statistics."
            )
            actions = ["Review aggregate behavior.", "Test one targeted campaign.", "Measure incremental outcomes."]

    if wants_metrics_only:
        actions = []

    risks = [] if wants_metrics_only else ["Local fallback operates deterministically on verified project data."]

    driver_val = target_seg.get(feature, safe.get(feature, "not available")) if target_seg else safe.get(feature, "not available")
    return validate_advice({
        "segment": name,
        "summary": summary,
        "key_drivers": [{"feature": feature, "direction": direction, "value_cited": driver_val}],
        "risks": risks,
        "actions": actions,
        "mode": "mock",
        "status_detail": "simulation",
    }, target_seg or safe)


def _log_run(dataset: str, model: str, temperature: float, prompt: str, raw: str, parsed: bool, mode: str) -> None:
    path = OUTPUTS_DIR / "llm_runs.csv"
    path.parent.mkdir(exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "dataset", "model", "temperature", "prompt_hash", "run_id", "raw_output", "parse_success", "mode"])
        if not exists:
            writer.writeheader()
        clean_raw = raw
        for secret in [os.getenv("GEMINI_API_KEY"), os.getenv("LLM_API_KEY"), os.getenv("OPENAI_API_KEY")]:
            if secret and len(secret) > 4:
                clean_raw = clean_raw.replace(secret, "[REDACTED_KEY]")
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "model": model,
            "temperature": temperature,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "run_id": uuid.uuid4().hex,
            "raw_output": clean_raw,
            "parse_success": parsed,
            "mode": mode,
        })


def _call_gemini_api(
    safe_profile: dict[str, Any],
    question: str | None,
    api_key: str,
    model: str,
    temperature: float,
) -> tuple[dict[str, Any], str, float]:
    """Call Google Gemini API using the official google-genai SDK with structured output and candidate fallback."""
    from google import genai
    from google.genai import types

    http_options = types.HttpOptions(
        timeout=10000,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    client = genai.Client(api_key=api_key, http_options=http_options)
    features_list = [k for k in safe_profile.keys() if k not in ("Segment_Name", "Segments_Summary")]
    sys_instruction = (
        "You are an expert e-commerce customer loyalty strategist. "
        "Analyze aggregate segment statistics and answer the user question. "
        "Strictly adhere to the aggregate statistics provided. "
        "Never invent, assume, or request customer IDs, order IDs, email addresses, or personal customer data. "
        "Output ONLY a valid JSON object matching the required schema."
    )
    prompt_text = (
        "Analyze the following aggregate segment statistics and provide actionable loyalty strategy:\n"
        f"Available aggregate features to cite in key_drivers: {features_list}\n"
        f"Aggregate segment statistics:\n{json.dumps(safe_profile, default=str)}\n"
    )
    if question:
        prompt_text += f"\nUser Question:\n{question}\n"

    config = types.GenerateContentConfig(
        system_instruction=sys_instruction,
        response_mime_type="application/json",
        response_schema=_SegmentAdviceResponseSchema,
        temperature=temperature,
        max_output_tokens=2048,
    )

    candidate_models = [model]
    # Verified available models from Google AI Studio catalog for this account
    for alt in ["gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.8-flash"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    last_exc: Exception | None = None
    for candidate in candidate_models:
        t_call_start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt_text,
                config=config,
            )
            raw = response.text or ""
            t_api_ms = (time.perf_counter() - t_call_start) * 1000
            parsed = json.loads(strip_json_fences(raw))
            parsed["mode"] = "gemini_live"
            parsed["status_detail"] = "live"
            parsed["model_used"] = candidate
            validated = validate_advice(parsed, safe_profile)
            return validated, raw, t_api_ms
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()

            # For model-not-found (404 / no longer available), attempt the next candidate model
            if (
                "404" in err_str
                or "not_found" in err_str
                or "not found" in err_str
                or "no longer available" in err_str
            ) and candidate != candidate_models[-1]:
                logger.info("Candidate model %s unavailable (%s). Trying fallback candidate.", candidate, str(exc)[:80])
                continue

            # For schema rejection on a supported model, retry once without response_schema
            if (
                "response_schema" in err_str
                or "schema not supported" in err_str
            ) and not any(k in err_str for k in ("503", "504", "429", "quota", "unavailable", "deadline")):
                try:
                    plain_config = types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        response_mime_type="application/json",
                        temperature=temperature,
                        max_output_tokens=2048,
                    )
                    t_plain_start = time.perf_counter()
                    response = client.models.generate_content(
                        model=candidate,
                        contents=prompt_text,
                        config=plain_config,
                    )
                    raw = response.text or ""
                    t_api_ms = (time.perf_counter() - t_plain_start) * 1000
                    parsed = json.loads(strip_json_fences(raw))
                    parsed["mode"] = "gemini_live"
                    parsed["status_detail"] = "live"
                    parsed["model_used"] = candidate
                    return validate_advice(parsed, safe_profile), raw, t_api_ms
                except Exception as inner_exc:
                    last_exc = inner_exc

            # For 503, 504, 429, timeout, network error: do not chain retries across models on same endpoint
            raise last_exc or exc

    if last_exc:
        raise last_exc
    raise RuntimeError("No candidate model succeeded.")


def get_segment_advice(
    segment_profile: dict[str, Any],
    dataset: str = "unknown",
    allow_mock_fallback: bool = False,
    question: str | None = None,
) -> dict[str, Any]:
    t_total_start = time.perf_counter()

    # 1. Data prep and sanitization stage
    t_prep_start = time.perf_counter()
    safe = _safe_profile(segment_profile)
    question = sanitize_chat_question(question) if question else None
    t_prep_ms = (time.perf_counter() - t_prep_start) * 1000

    # 2. Deterministic greeting & conversational intent detection BEFORE Gemini or fallback templates
    conversational_intent = classify_conversational_intent(question)
    if conversational_intent:
        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        result = _build_conversational_response(conversational_intent, safe)
        result["timing_ms"] = {
            "prep_ms": round(t_prep_ms, 2),
            "api_ms": 0.0,
            "parse_ms": 0.0,
            "total_ms": round(t_total_ms, 2),
        }
        _log_run(dataset, "conversational_router", 0.0, question or "", json.dumps(result), True, result["mode"])
        return result

    # 3. Route to target segment if question specifically targets one (e.g. high-risk)
    target_seg, match_status = _extract_target_segment(question, safe, dataset)
    if target_seg is not None and match_status == "matched":
        seg_summary_backup = safe.get("Segments_Summary")
        safe = _safe_profile(target_seg)
        if seg_summary_backup:
            safe["Segments_Summary"] = seg_summary_backup

    # Fetch environment variables securely
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = (os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Reset circuit breaker if API key changed
    circuit_breaker.check_key_rotation(api_key)

    if not api_key:
        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        result = mock_advice(safe, question=question, dataset=dataset) if allow_mock_fallback else {
            "segment": safe.get("Segment_Name", "Selected segment"),
            "summary": "Advisor credentials are not configured. Set GEMINI_API_KEY in .env.",
            "key_drivers": [],
            "risks": [],
            "actions": ["Configure GEMINI_API_KEY.", "Enable mock mode explicitly.", "Review aggregate profiles."],
            "mode": "error",
            "status_detail": "missing_api_key",
        }
        if result.get("mode") == "mock":
            result["status_detail"] = "missing_api_key"
            result["fallback_notice"] = OFFLINE_FALLBACK_NOTICE
            result["recovery_hint"] = "Configure GEMINI_API_KEY in .env to enable live Gemini insights."
        result["timing_ms"] = {
            "prep_ms": round(t_prep_ms, 2),
            "api_ms": 0.0,
            "parse_ms": 0.0,
            "total_ms": round(t_total_ms, 2),
        }
        _log_run(dataset, model or "unconfigured", temperature, json.dumps(safe, default=str), json.dumps(result), True, result["mode"])
        return result

    # Check in-memory query cache for identical prompt on same profile
    cache_key = ""
    if question:
        cache_key = f"{dataset}:{safe.get('Segment_Name', '')}:{question.lower().strip()}:{model}"
        if cache_key in _QUERY_CACHE:
            cached_res = copy.deepcopy(_QUERY_CACHE[cache_key])
            cached_res["cached"] = True
            t_total_ms = (time.perf_counter() - t_total_start) * 1000
            cached_res["timing_ms"] = {
                "prep_ms": round(t_prep_ms, 2),
                "api_ms": 0.0,
                "parse_ms": 0.0,
                "total_ms": round(t_total_ms, 2),
            }
            return cached_res

    # Check circuit breaker before making external network call
    allowed, cb_reason, remaining_cooldown = circuit_breaker.should_allow_request()
    if not allowed:
        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        if allow_mock_fallback:
            result = mock_advice(safe, question=question, dataset=dataset)
            result["mode"] = "mock_fallback"
            result["status_detail"] = cb_reason
            result["circuit_open"] = True
            result["cooldown_remaining"] = round(remaining_cooldown, 1)
            result["recovery_hint"] = circuit_breaker.get_recovery_hint()
            result["fallback_notice"] = OFFLINE_FALLBACK_NOTICE
        else:
            result = {
                "segment": safe.get("Segment_Name", "Selected segment"),
                "summary": f"AI insights are temporarily in cooldown ({remaining_cooldown:.0f}s remaining) due to recent {cb_reason}.",
                "key_drivers": [],
                "risks": ["Service in cooldown."],
                "actions": ["Wait for cooldown to elapse.", "Review aggregate profiles."],
                "mode": "error",
                "status_detail": cb_reason,
                "circuit_open": True,
                "cooldown_remaining": round(remaining_cooldown, 1),
                "recovery_hint": circuit_breaker.get_recovery_hint(),
            }
        result["timing_ms"] = {
            "prep_ms": round(t_prep_ms, 2),
            "api_ms": 0.0,
            "parse_ms": 0.0,
            "total_ms": round(t_total_ms, 2),
        }
        _log_run(dataset, model, temperature, json.dumps(safe, default=str), json.dumps(result), True, result["mode"])
        return result

    raw = ""
    parsed = False
    result: dict[str, Any] | None = None
    status_detail = "api_error"
    t_api_ms = 0.0
    t_parse_ms = 0.0

    # 3. Primary path: Official Google Gen AI SDK
    try:
        t_api_entry = time.perf_counter()
        result, raw, t_api_call_ms = _call_gemini_api(safe, question, api_key, model, temperature)
        t_api_ms = t_api_call_ms
        t_parse_ms = max(0.0, (time.perf_counter() - t_api_entry) * 1000 - t_api_call_ms)
        parsed = True
        # Record success to close circuit breaker and reset failures
        circuit_breaker.record_success()
        if cache_key and result.get("mode") in ("gemini_live", "mock"):
            if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
                _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)), None)
            _QUERY_CACHE[cache_key] = copy.deepcopy(result)
    except Exception as exc:
        err_msg = str(exc)
        exc_type = type(exc).__name__
        for secret in [api_key, os.getenv("GEMINI_API_KEY"), os.getenv("LLM_API_KEY"), os.getenv("OPENAI_API_KEY")]:
            if secret and len(secret) > 4:
                err_msg = err_msg.replace(secret, "[REDACTED_KEY]")

        err_lower = err_msg.lower()
        if "429" in err_lower or "resource_exhausted" in err_lower or "quota" in err_lower:
            status_detail = "quota_exceeded"
        elif "401" in err_lower or "403" in err_lower or "unauthenticated" in err_lower or "api_key_invalid" in err_lower or "permission_denied" in err_lower:
            status_detail = "invalid_api_key"
        elif "503" in err_lower or "unavailable" in err_lower:
            status_detail = "service_unavailable_503"
        elif "504" in err_lower or "deadline" in err_lower:
            status_detail = "deadline_exceeded_504"
        elif "404" in err_lower or "not_found" in err_lower or "not found" in err_lower or "no longer available" in err_lower:
            status_detail = "model_not_found_404"
        elif "timeout" in err_lower or "timed out" in err_lower or "connect" in err_lower or "getaddrinfo" in err_lower or "connection reset" in err_lower:
            status_detail = "network_error"
        else:
            status_detail = f"api_error_{exc_type}"

        # Record failure and set circuit breaker cooldown
        retry_delay = parse_retry_delay(exc)
        circuit_breaker.record_failure(status_detail, retry_after=retry_delay)

        clean_err = err_msg[:180]

        # 4. Secondary fallback: OpenAI compatible endpoint only for non-Google third-party proxies
        is_google_base = "googleapis.com" in os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).lower()
        if os.getenv("LLM_BASE_URL") and not is_google_base:
            try:
                from openai import OpenAI
                t_openai_start = time.perf_counter()
                client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
                prompt = (
                    "Return strict JSON with segment, summary, key_drivers, risks, actions (exactly 3), mode. "
                    f"Use only this aggregate profile: {json.dumps(safe, default=str)}"
                )
                if question:
                    prompt += f" Analyst question: {question}."
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Use aggregate retail statistics only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                )
                t_api_ms = (time.perf_counter() - t_openai_start) * 1000
                raw = response.choices[0].message.content or ""
                candidate = validate_advice(json.loads(strip_json_fences(raw)), safe)
                candidate["mode"] = "gemini_live"
                candidate["status_detail"] = "live"
                result = candidate
                parsed = True
                circuit_breaker.record_success()
            except Exception:
                pass

        if result is None:
            logger.warning(
                "Gemini advisor API failure (%s): %s. Operating in offline fallback mode.",
                status_detail,
                clean_err,
            )
            if allow_mock_fallback:
                result = mock_advice(safe, question=question, dataset=dataset)
                result["mode"] = "mock_fallback"
                result["status_detail"] = status_detail
                result["api_error"] = f"[{exc_type}] {clean_err}"
                result["fallback_notice"] = OFFLINE_FALLBACK_NOTICE
                result["recovery_hint"] = circuit_breaker.get_recovery_hint()
                result["cooldown_remaining"] = round(circuit_breaker.get_remaining_cooldown(), 1)
            else:
                result = {
                    "segment": safe.get("Segment_Name", "Selected segment"),
                    "summary": "AI insights are temporarily unavailable. Please try again later.",
                    "key_drivers": [],
                    "risks": ["Service temporarily offline."],
                    "actions": ["Retry request.", "Verify credentials in settings.", "Review aggregate profiles."],
                    "mode": "error",
                    "status_detail": status_detail,
                    "api_error": f"[{exc_type}] {clean_err}",
                    "recovery_hint": circuit_breaker.get_recovery_hint(),
                    "cooldown_remaining": round(circuit_breaker.get_remaining_cooldown(), 1),
                }

    t_total_ms = (time.perf_counter() - t_total_start) * 1000
    result["timing_ms"] = {
        "prep_ms": round(t_prep_ms, 2),
        "api_ms": round(t_api_ms, 2),
        "parse_ms": round(t_parse_ms, 2),
        "total_ms": round(t_total_ms, 2),
    }

    _log_run(dataset, model, temperature, json.dumps(safe, default=str), raw or json.dumps(result), parsed, result["mode"])
    return result
