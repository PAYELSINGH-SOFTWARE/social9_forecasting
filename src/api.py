from fastapi import FastAPI, HTTPException
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

class ForecastRequest(BaseModel):
    forecast_days: int = Field(
        default=7,
        ge=1,
        le=90
    )


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
def create_forecast(request: ForecastRequest):
    try:
        predictions = forecast(
            request.forecast_days
        )

        return {
            "model": "XGBoost",
            "forecast_days": request.forecast_days,
            "predictions": predictions
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================
# AI CAPTION API
# =========================

@app.post("/ai-caption")
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
            detail=str(error)
        )


# =========================
# AI TEXT API
# =========================

@app.post("/ai-text")
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
            detail=str(error)
        )


# =========================
# AI CONTENT CALENDAR API
# =========================

@app.post("/ai-calendar")
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
            detail=str(error)
        )