from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import forecast


app = FastAPI(
    title="Social9 Forecasting API",
    description="Standalone ML forecasting service",
    version="1.0.0",
)


class ForecastRequest(BaseModel):

    forecast_days: int = Field(
        default=7,
        ge=1,
        le=90
    )


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


@app.post("/forecast")
def create_forecast(
    request: ForecastRequest
):

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