from pathlib import Path
import os

from dotenv import load_dotenv


# Load .env from AI_model folder
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

def generate_ai_text(prompt: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
    )

    return response.text or ""
