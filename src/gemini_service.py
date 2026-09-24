from pathlib import Path
import logging
import os

from dotenv import load_dotenv


# Load .env from AI_model folder
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
logger = logging.getLogger(__name__)

def generate_ai_text(prompt: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception as error:
        # Provider messages can contain request details. Log only safe diagnostics.
        logger.error(
            "Gemini generation failed: exception=%s status=%s model=%s",
            type(error).__name__,
            getattr(error, "code", None),
            model,
        )
        raise

    return response.text or ""
