from pathlib import Path
import logging
import os
import base64
import math
import re

from dotenv import load_dotenv


# Load .env from AI_model folder
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
logger = logging.getLogger(__name__)
FREE_TIER_MODELS = {"gemini-3.8-flash", "gemini-3.5-flash-lite"}


class AIQuotaExhausted(RuntimeError):
    def __init__(self, message="AI provider usage limit reached", *, retry_after=None, reason="unknown"):
        super().__init__(message)
        self.retry_after = retry_after
        self.reason = reason


def quota_failure(error):
    """Read only structured quota metadata; never expose prompts, keys or project IDs."""
    payload = getattr(error, "details", {})
    if not isinstance(payload, dict):
        payload = {}
    body = payload.get("error", payload)
    details = body.get("details", []) if isinstance(body, dict) else []
    retry_after, reason = None, "unknown"
    for item in details if isinstance(details, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
            value = item.get("retryDelay")
            if isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?s", value):
                seconds = float(value[:-1])
                if 0 < seconds <= 604800:
                    retry_after = math.ceil(seconds)
        if item.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
            violations = item.get("violations", [])
            for violation in violations if isinstance(violations, list) else []:
                if not isinstance(violation, dict):
                    continue
                quota_id = str(violation.get("quotaId", ""))
                if str(violation.get("quotaValue")) == "0":
                    reason = "quota_unavailable"
                elif "PerDay" in quota_id and reason != "quota_unavailable":
                    reason = "daily_quota"
                elif "PerMinute" in quota_id and reason == "unknown":
                    reason = "rate_limit"
    return AIQuotaExhausted(retry_after=retry_after, reason=reason)


def quota_http_error(error):
    from fastapi import HTTPException
    messages = {
        "daily_quota": "The AI provider's daily usage limit has been reached. Try again after the quota resets.",
        "quota_unavailable": "AI is unavailable because the provider has not allocated quota. Please contact Social9 support.",
        "rate_limit": "The AI provider is receiving too many requests. Please try again shortly.",
        "unknown": "The AI provider's usage limit has been reached. Please try again later.",
    }
    reason = error.reason if error.reason in messages else "unknown"
    headers = {"X-AI-Limit-Reason": reason}
    if error.retry_after:
        headers["Retry-After"] = str(error.retry_after)
    return HTTPException(429, messages[reason], headers=headers)


def generate_ai_text(prompt: str, *, json_mode: bool = False, images: list[str] | None = None, max_output_tokens: int = 2048) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    if os.getenv("GEMINI_FREE_TIER_ONLY", "true").lower() == "true" and model not in FREE_TIER_MODELS:
        raise RuntimeError("This model is disabled in free-tier-only mode")
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=170_000))
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json" if json_mode else "text/plain",
            max_output_tokens=max(256, min(max_output_tokens, 32768, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768")))),
            thinking_config=types.ThinkingConfig(thinking_level="medium" if max_output_tokens > 4096 else "low") if model == "gemini-3.8-flash" else None,
        )
        contents = [prompt]
        for image in images or []:
            header, encoded = image.split(",", 1)
            contents.append(types.Part.from_bytes(data=base64.b64decode(encoded, validate=True), mime_type=header[5:].split(";")[0]))
        response = client.models.generate_content(model=model, contents=contents if images else prompt, config=config)
    except Exception as error:
        if getattr(error, "code", None) == 429:
            failure = quota_failure(error)
            logger.warning("Gemini usage limit: model=%s reason=%s retry_after=%s", model, failure.reason, failure.retry_after)
            raise failure from error
        # Provider messages can contain request details. Log only safe diagnostics.
        logger.error(
            "Gemini generation failed: exception=%s status=%s model=%s",
            type(error).__name__,
            getattr(error, "code", None),
            model,
        )
        raise

    return response.text or ""
