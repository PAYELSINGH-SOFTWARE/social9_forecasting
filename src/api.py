import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .predict import forecast
from .gemini_service import generate_ai_text
from .ai_calendar import generate_content_calendar


app = FastAPI(
    title="Social9 Forecasting API",
    description="ML forecasting and AI content generation service",
    version="1.0.0",
)


# =========================
# REQUEST MODELS
# =========================

class DailyMetric(BaseModel):
    date: str
    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    reach: int = Field(ge=0)
    impressions: int = Field(ge=0)
    followers: int = Field(ge=0)
    posts_count: int = Field(default=0, ge=0)


class ForecastRequest(BaseModel):
    forecast_days: int = Field(
        default=7,
        ge=1,
        le=90
    )
    history: list[DailyMetric] | None = Field(default=None, min_length=8, max_length=365)


def require_service_key(x_social9_forecasting_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("FORECASTING_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Forecasting service authentication is not configured")
    if not x_social9_forecasting_key or not hmac.compare_digest(
        x_social9_forecasting_key, expected
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


class CaptionRequest(BaseModel):
    topic: str = Field(
        default="AI forecasting",
        min_length=1,
        max_length=200
    )

    platform: str = Field(
        default="Instagram",
        min_length=1,
        max_length=50
    )

    tone: str = Field(
        default="professional",
        min_length=1,
        max_length=50
    )


class CalendarRequest(BaseModel):
    platform: str = Field(
        default="Instagram",
        min_length=1,
        max_length=50
    )

    duration_days: int = Field(
        default=7,
        ge=1,
        le=30
    )

    topic: str = Field(
        default="AI forecasting",
        min_length=1,
        max_length=200
    )


class AITextRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=2000
    )


# =========================
# BASIC ENDPOINTS
# =========================

@app.get("/")
def root():
    return {
        "service": "Social9 Forecasting",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================
# FORECASTING API
# =========================

@app.post("/forecast")
def create_forecast(
    request: ForecastRequest,
    x_social9_forecasting_key: str | None = Header(default=None),
):
    try:
        require_service_key(x_social9_forecasting_key)
        predictions = forecast(
            request.forecast_days,
            [item.model_dump() for item in request.history] if request.history else None,
        )

        return {
            "model": "XGBoost",
            "forecast_days": request.forecast_days,
            "predictions": predictions
        }

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Forecasting is temporarily unavailable"
        ) from error


# =========================
# AI CAPTION API
# =========================

@app.post("/ai-caption", dependencies=[Depends(require_service_key)])
def create_ai_caption(request: CaptionRequest):
    try:
        prompt = f"""
You are a social media content expert.

Create an engaging social media caption.

Topic: {request.topic}
Platform: {request.platform}
Tone: {request.tone}

Requirements:
- Write one engaging caption.
- Include relevant hashtags.
- Do not include unnecessary explanations.
"""

        caption = generate_ai_text(prompt)

        return {
            "platform": request.platform,
            "topic": request.topic,
            "tone": request.tone,
            "caption": caption
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="AI content generation is temporarily unavailable"
        )


# =========================
# AI TEXT API
# =========================

@app.post("/ai-text", dependencies=[Depends(require_service_key)])
def create_ai_text(request: AITextRequest):
    try:
        result = generate_ai_text(
            request.prompt
        )

        return {
            "prompt": request.prompt,
            "response": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="AI content generation is temporarily unavailable"
        )


# =========================
# AI CONTENT CALENDAR API
# =========================

@app.post("/ai-calendar", dependencies=[Depends(require_service_key)])
def create_ai_calendar(request: CalendarRequest):
    try:
        calendar = generate_content_calendar(
            platform=request.platform,
            duration_days=request.duration_days,
            topic=request.topic
        )

        return {
            "platform": request.platform,
            "duration_days": request.duration_days,
            "topic": request.topic,
            "calendar": calendar
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="AI content generation is temporarily unavailable"
        )
