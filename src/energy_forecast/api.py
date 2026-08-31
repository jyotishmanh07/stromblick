"""FastAPI application."""

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from .service import ForecastService

app = FastAPI(title="Stromblick", version="0.1.0")
service = ForecastService()


class ForecastRequest(BaseModel):
    as_of: datetime
    horizon_hours: int = 24

    @field_validator("as_of")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone, for example Z")
        return value

    @field_validator("horizon_hours")
    @classmethod
    def valid_horizon(cls, value: int) -> int:
        if value < 1 or value > 24:
            raise ValueError("horizon_hours must be between 1 and 24")
        return value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "data_source": service.data_source}


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict[str, object]:
    return service.forecast(request.as_of, request.horizon_hours)
