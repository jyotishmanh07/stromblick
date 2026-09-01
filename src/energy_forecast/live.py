"""Live SMARD fetch for the deployed dashboard.

The committed snapshot under `data/clean/` is a fixed historical export, so an app
reading only that file forecasts forward from whenever it was last collected. This
module bridges the gap: it pulls the most recent weekly chunks from SMARD, merges
them onto the committed history, and hands back a frame that ends at the last
*published* hour.

Two properties are load-bearing and both come from `canonicalize_demand`:

* **Concat order decides who wins.** `.drop_duplicates(keep="last")` follows
  concatenation order, so `pd.concat([committed, live])` lets live values overwrite
  the committed snapshot on overlapping hours. Reversing the arguments silently
  inverts that.
* **A live NaN can never clobber an observed value.** `canonicalize_demand` drops
  null rows *before* the dedup, so an hour SMARD has indexed but not yet published
  simply is not there to win.

Everything here is best-effort. A hosted demo must not break because an upstream
API changed shape, so every failure path returns the committed frame with a
human-readable warning rather than raising.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .data import DataValidationError, canonicalize_demand
from .quality import validate_demand
from .smard_api import SMARDAPIClient

COMMITTED_LABEL = "SMARD clean export"
LIVE_LABEL = "SMARD live API"


@dataclass
class LiveResult:
    """Outcome of a live-fetch attempt. `history` is always usable."""

    history: pd.DataFrame
    is_live: bool
    source_label: str
    provenance: dict | None = None
    warning: str | None = None


def bridge_weeks(
    committed_max: pd.Timestamp, now: pd.Timestamp, min_weeks: int = 2, max_weeks: int = 8
) -> int | None:
    """How many weekly chunks are needed to reach `now` from the committed snapshot.

    A fixed fetch size only works while the snapshot is fresh. Left alone for two
    months, a two-week fetch would leave a two-month hole that `canonicalize_demand`
    resamples into NaN rows, breaking lag lookups either side of the seam. So the
    count is derived from the actual gap.

    Returns None when the gap is wider than `max_weeks` — at that point the snapshot
    needs a real re-ingest, and serving a series with a hole in it would be worse
    than admitting the data is stale. The cap also bounds latency: the SMARD client
    fetches sequentially with no connection pooling.
    """
    gap_days = (now - committed_max).total_seconds() / 86_400
    if gap_days < 0:  # committed data ahead of "now" (clock skew, or a test fixture)
        return min_weeks
    needed = math.ceil(gap_days / 7) + 1
    if needed > max_weeks:
        return None
    return max(min_weeks, needed)


def fetch_live_history(
    committed: pd.DataFrame, now: pd.Timestamp | None = None, timeout_seconds: float = 5.0,
    max_weeks: int = 8, deadline_seconds: float = 12.0,
) -> LiveResult:
    """Merge the latest SMARD chunks onto the committed history, within a hard deadline.

    Never raises: any failure returns the committed frame with `is_live=False`.

    `timeout_seconds` bounds a single HTTP request; `deadline_seconds` bounds the whole
    attempt. Both are needed — the client fetches sequentially, so per-request timeouts
    alone would let a slow upstream stall a page load for their sum. Past the deadline
    the committed snapshot is served instead; a visitor waiting on a spinner is worse
    than a visitor seeing slightly older data.
    """
    box: dict[str, LiveResult] = {}

    def run() -> None:
        box["result"] = _fetch_live_history(committed, now, timeout_seconds, max_weeks)

    # A daemon thread, not ThreadPoolExecutor: the executor joins its workers both on
    # context exit and at interpreter shutdown, so a hung request would still block --
    # which is the whole thing this deadline exists to prevent.
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=deadline_seconds)
    if worker.is_alive() or "result" not in box:
        return LiveResult(
            committed, False, COMMITTED_LABEL,
            warning=(
                f"Live SMARD fetch exceeded {deadline_seconds:.0f}s and was abandoned. "
                "Showing the committed snapshot."
            ),
        )
    return box["result"]


def _fetch_live_history(
    committed: pd.DataFrame, now: pd.Timestamp | None, timeout_seconds: float, max_weeks: int
) -> LiveResult:
    fallback = LiveResult(committed, False, COMMITTED_LABEL)
    if committed.empty:
        return LiveResult(committed, False, COMMITTED_LABEL, warning="No committed history.")

    reference = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    committed_max = pd.to_datetime(committed.timestamp, utc=True).max()
    weeks = bridge_weeks(committed_max, reference, max_weeks=max_weeks)
    if weeks is None:
        fallback.warning = (
            f"The committed snapshot ends {committed_max:%Y-%m-%d} — too far behind to bridge "
            f"with a live fetch (limit {max_weeks} weeks). Re-run "
            "`scripts/ingest_smard_api.py` to refresh it."
        )
        return fallback

    try:
        client = SMARDAPIClient(timeout_seconds=timeout_seconds)
        live, urls = client.fetch_latest(weeks=weeks)
        # Order matters: live values win on overlapping hours. See module docstring.
        merged = canonicalize_demand(pd.concat([committed, live], ignore_index=True))
        report = validate_demand(merged)
        if not report.passed:
            fallback.warning = (
                "Live SMARD data failed the quality gate ("
                + "; ".join(f"{c.name}: {c.detail}" for c in report.errors)
                + "). Showing the committed snapshot instead."
            )
            return fallback
        observed = merged.dropna(subset=["demand_mw"])
        provenance = {
            "source": "Bundesnetzagentur | SMARD.de",
            "source_url": client.index_url,
            "chunk_urls": urls,
            "module_id": client.module_id,
            "region": client.region,
            "resolution": client.resolution,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(merged)),
            "first_timestamp": merged.timestamp.min().isoformat(),
            "last_timestamp": merged.timestamp.max().isoformat(),
            "last_observed_timestamp": (
                observed.timestamp.max().isoformat() if not observed.empty else None
            ),
            "weeks_fetched": weeks,
            "quality": report.to_dict(),
        }
        return LiveResult(merged, True, LIVE_LABEL, provenance=provenance)
    except (ConnectionError, DataValidationError) as error:
        fallback.warning = f"Live SMARD fetch failed ({error}). Showing the committed snapshot."
        return fallback
    except Exception as error:  # noqa: BLE001 - a hosted demo must survive upstream changes
        fallback.warning = (
            f"Live SMARD fetch failed unexpectedly ({type(error).__name__}: {error}). "
            "Showing the committed snapshot."
        )
        return fallback


def last_observed(history: pd.DataFrame) -> pd.Timestamp | None:
    """Timestamp of the most recent hour with a published value.

    SMARD indexes hours before publishing them, so the last *row* and the last
    *observation* are not always the same instant. Forecasts and freshness figures
    must key off this, not off `timestamp.max()`.
    """
    observed = history.dropna(subset=["demand_mw"])
    return None if observed.empty else pd.to_datetime(observed.timestamp, utc=True).max()
