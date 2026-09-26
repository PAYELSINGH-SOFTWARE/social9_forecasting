from pathlib import Path
import logging
import os
import base64

from dotenv import load_dotenv


# Load .env from AI_model folder
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
logger = logging.getLogger(__name__)

def generate_ai_text(prompt: str, *, json_mode: bool = False, images: list[str] | None = None) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    try:
        config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
        contents = [prompt]
        for image in images or []:
            header, encoded = image.split(",", 1)
            contents.append(types.Part.from_bytes(data=base64.b64decode(encoded, validate=True), mime_type=header[5:].split(";")[0]))
        response = client.models.generate_content(model=model, contents=contents if images else prompt, config=config)
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
