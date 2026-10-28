from pathlib import Path
import logging
import os
import base64

from dotenv import load_dotenv


# Load .env from AI_model folder
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
logger = logging.getLogger(__name__)
FREE_TIER_MODELS = {"gemini-3.8-flash", "gemini-3.5-flash-lite"}


class AIQuotaExhausted(RuntimeError):
    pass

def generate_ai_text(prompt: str, *, json_mode: bool = False, images: list[str] | None = None) -> str:
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
            max_output_tokens=max(4096, min(32768, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768")))),
            thinking_config=types.ThinkingConfig(thinking_level="medium") if model == "gemini-3.8-flash" else None,
        )
        contents = [prompt]
        for image in images or []:
            header, encoded = image.split(",", 1)
            contents.append(types.Part.from_bytes(data=base64.b64decode(encoded, validate=True), mime_type=header[5:].split(";")[0]))
        response = client.models.generate_content(model=model, contents=contents if images else prompt, config=config)
    except Exception as error:
        if getattr(error, "code", None) == 429:
            raise AIQuotaExhausted("Free AI quota exhausted; try again later") from error
        # Provider messages can contain request details. Log only safe diagnostics.
        logger.error(
            "Gemini generation failed: exception=%s status=%s model=%s",
            type(error).__name__,
            getattr(error, "code", None),
            model,
        )
        raise

    return response.text or ""
