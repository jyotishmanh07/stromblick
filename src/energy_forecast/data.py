"""SMARD ingestion and canonical hourly demand data handling."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

DEFAULT_SMARD_MODULE_ID = 410
DEFAULT_SMARD_REGION = "DE"
DEFAULT_SMARD_RESOLUTION = "hour"
DEFAULT_SMARD_INDEX_URL = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
DEFAULT_SMARD_URL = DEFAULT_SMARD_INDEX_URL


def smard_api_chunk_url(module_id: int, region: str, resolution: str, timestamp: int) -> str:
    """Build the official SMARD chart-data URL for one timestamp chunk."""
    return (
        f"https://www.smard.de/app/chart_data/{module_id}/{region}/"
        f"{module_id}_{region}_{resolution}_{timestamp}.json"
    )


class DataValidationError(ValueError):
    """Raised when a source file cannot be converted to the project schema."""


@dataclass
class IngestionResult:
    data: pd.DataFrame
    source_url: str
    collected_at: str
    raw_path: Path | None = None
    metadata_path: Path | None = None


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])
    for col in columns:
        text = str(col).lower()
        if any(candidate in text for candidate in candidates):
            return str(col)
    return None


def _json_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if "series" in payload and isinstance(payload["series"], list):
            series = payload["series"]
            if series and isinstance(series[0], (list, tuple)) and len(series[0]) >= 2:
                return pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            [row[0] for row in series], unit="ms", utc=True
                        ),
                        "demand_mw": [row[1] for row in series],
                    }
                )
        for key in ("data", "values", "series", "result"):
            if key in payload and isinstance(payload[key], (list, dict)):
                return _json_to_frame(payload[key])
    raise DataValidationError("JSON response does not contain a tabular data array")


def canonicalize_demand(frame: pd.DataFrame, timezone_name: str = "Europe/Berlin") -> pd.DataFrame:
    """Return sorted, deduplicated hourly data with UTC timestamps and demand_mw.

    Naive source timestamps are interpreted as German local time. Converting to UTC
    before resampling makes daylight-saving transitions explicit and reproducible.
    """
    if frame.empty:
        raise DataValidationError("demand source is empty")
    timestamp_col = _find_column(
        list(frame.columns), ("timestamp", "time", "datetime", "date", "utc_timestamp")
    )
    demand_col = _find_column(
        list(frame.columns),
        ("demand_mw", "demand", "consumption", "load", "electricity_consumption"),
    )
    if timestamp_col is None or demand_col is None:
        raise DataValidationError("source must contain timestamp and demand/consumption columns")

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if timestamps.isna().any():
        raise DataValidationError("source contains invalid timestamps")
    if timestamps.dt.tz is None:
        try:
            timestamps = timestamps.dt.tz_localize(
                timezone_name, ambiguous="infer", nonexistent="shift_forward"
            )
        except ValueError as exc:
            raise DataValidationError(
                "source contains ambiguous daylight-saving timestamps; "
                "provide timezone-aware values"
            ) from exc
        if timestamps.isna().any():
            raise DataValidationError("source contains ambiguous daylight-saving timestamps")
    timestamps = timestamps.dt.tz_convert("UTC")
    demand = pd.to_numeric(
        frame[demand_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    result = pd.DataFrame({"timestamp": timestamps, "demand_mw": demand}).dropna()
    if result.empty:
        raise DataValidationError("source contains no numeric demand observations")
    result = result.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return (
        result.set_index("timestamp")
        .resample("1h")
        .agg(demand_mw=("demand_mw", "mean"))
        .reset_index()
    )


def missing_hourly_timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Identify gaps; gaps are reported instead of silently imputed."""
    if frame.empty:
        return pd.DatetimeIndex([], tz="UTC")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).sort_values()
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="1h", tz="UTC")
    return expected.difference(timestamps)


class SMARDClient:
    """Small, testable client for a downloaded SMARD export or compatible endpoint."""

    def __init__(self, url: str = DEFAULT_SMARD_URL, timeout_seconds: float = 30.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def fetch(self) -> tuple[bytes, str]:
        try:
            response = httpx.get(self.url, timeout=self.timeout_seconds, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ConnectionError(f"could not download SMARD data: {exc}") from exc
        return response.content, response.headers.get("content-type", "")

    def parse(self, content: bytes, content_type: str = "") -> pd.DataFrame:
        text = content.decode("utf-8-sig")
        if "json" in content_type or text.lstrip().startswith(("{", "[")):
            return canonicalize_demand(_json_to_frame(json.loads(text)))
        return canonicalize_demand(pd.read_csv(io.StringIO(text), sep=None, engine="python"))

    def download(
        self, raw_dir: str | Path = "data/raw", clean_dir: str | Path = "data/clean"
    ) -> IngestionResult:
        raw_dir, clean_dir = Path(raw_dir), Path(clean_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        content, content_type = self.fetch()
        collected_at = datetime.now(timezone.utc).isoformat()
        raw_path = raw_dir / "smard_demand_download"
        raw_path.write_bytes(content)
        data = self.parse(content, content_type)
        clean_path = clean_dir / "demand_hourly.csv"
        data.to_csv(clean_path, index=False)
        metadata = {
            "source_url": self.url,
            "collected_at": collected_at,
            "rows": len(data),
            "first_timestamp": data.timestamp.min().isoformat(),
            "last_timestamp": data.timestamp.max().isoformat(),
            "missing_hourly_observations": len(missing_hourly_timestamps(data)),
        }
        metadata_path = clean_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return IngestionResult(data, self.url, collected_at, raw_path, metadata_path)


def load_clean_demand(path: str | Path = "data/clean/demand_hourly.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"clean demand file not found: {path}")
    return canonicalize_demand(pd.read_csv(path))
