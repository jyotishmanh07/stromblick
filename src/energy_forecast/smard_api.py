"""Client for SMARD's official chart-data index and weekly JSON chunks.

The endpoint is used by the SMARD market-data application. Module 410 is
"Gesamt (Netzlast)" for Germany; values are average interval demand in MWh.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from .data import (
    DEFAULT_SMARD_MODULE_ID,
    DEFAULT_SMARD_REGION,
    DEFAULT_SMARD_RESOLUTION,
    DataValidationError,
    canonicalize_demand,
    smard_api_chunk_url,
)


class SMARDAPIClient:
    def __init__(
        self,
        module_id: int = DEFAULT_SMARD_MODULE_ID,
        region: str = DEFAULT_SMARD_REGION,
        resolution: str = DEFAULT_SMARD_RESOLUTION,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.module_id = module_id
        self.region = region
        self.resolution = resolution
        self.timeout_seconds = timeout_seconds
        self.index_url = (
            f"https://www.smard.de/app/chart_data/{module_id}/{region}/index_{resolution}.json"
        )

    def _get_json(self, url: str) -> dict:
        try:
            response = httpx.get(url, timeout=self.timeout_seconds, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ConnectionError(f"could not download SMARD data: {exc}") from exc
        try:
            return json.loads(response.content.decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"SMARD returned invalid JSON: {url}") from exc

    def fetch_latest(self, weeks: int = 52) -> tuple[pd.DataFrame, list[str]]:
        """Fetch and combine the latest `weeks` chunks listed by SMARD."""
        index = self._get_json(self.index_url)
        self.raw_payloads = {self.index_url: index}
        timestamps = index.get("timestamps")
        if not timestamps:
            raise DataValidationError("SMARD index response has no timestamps")
        selected = [int(value) for value in timestamps[-max(1, weeks):]]
        frames, urls = [], []
        for timestamp in selected:
            url = smard_api_chunk_url(self.module_id, self.region, self.resolution, timestamp)
            payload = self._get_json(url)
            self.raw_payloads[url] = payload
            series = payload.get("series")
            if not isinstance(series, list):
                raise DataValidationError(f"SMARD chunk has no series: {url}")
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime([row[0] for row in series], unit="ms", utc=True),
                    "demand_mw": [row[1] for row in series],
                }
            )
            frames.append(frame)
            urls.append(url)
        return canonicalize_demand(pd.concat(frames, ignore_index=True)), urls

    def download(
        self,
        weeks: int = 52,
        raw_dir: str | Path = "data/raw",
        clean_dir: str | Path = "data/clean",
    ) -> pd.DataFrame:
        raw_dir, clean_dir = Path(raw_dir), Path(clean_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        data, urls = self.fetch_latest(weeks)
        for position, (url, payload) in enumerate(self.raw_payloads.items()):
            (raw_dir / f"smard_{position:03d}.json").write_text(
                json.dumps({"source_url": url, "payload": payload}, indent=2), encoding="utf-8"
            )
        data.to_csv(clean_dir / "demand_hourly.csv", index=False)
        metadata = {
            "source": "Bundesnetzagentur | SMARD.de",
            "source_url": self.index_url,
            "chunk_urls": urls,
            "module_id": self.module_id,
            "region": self.region,
            "resolution": self.resolution,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(data),
            "first_timestamp": data.timestamp.min().isoformat(),
            "last_timestamp": data.timestamp.max().isoformat(),
        }
        (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return data
