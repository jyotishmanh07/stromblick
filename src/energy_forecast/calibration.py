"""Split-conformal calibration pool for the forecast's prediction band.

The band is the point forecast plus or minus the conformal quantile of the absolute
24-hour errors the shipped model made at the previous `CALIBRATION_ORIGINS` daily
origins. `scripts/interval_study.py` chose this recipe under a rule fixed before the run
(reports/interval_study.md): it covers 95% where the old one-week validation band
over-covered at 96.8% while being about 35% wider.

Computing the pool needs one model fit per origin, ~25 s for 28, which is too slow for
`api.py` (it builds the service at import) and for the dashboard's cold start. So the
daily forecast workflow builds it once (`build_pool`) and commits it to `POOL_PATH`;
everything else only reads it (`load_pool`, `pool_spread`). A missing, stale or
other-generation pool returns None, and callers fall back to the old recipe and say so.

`POOL_ORIGINS` is larger than `CALIBRATION_ORIGINS` so the track-record replay can size
its band from origins strictly before the days it scores (`verification.hindcast`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .inference import conformal_quantile
from .models import MODEL_VERSION, HistGradientBoostingForecast

POOL_PATH = Path("data/forecasts/calibration_pool.csv")
CALIBRATION_ORIGINS = 28
POOL_ORIGINS = CALIBRATION_ORIGINS + 7  # + the replayed week of the Track record tab
HORIZON_HOURS = 24
# A pool older than this no longer describes the model's recent errors. The daily
# workflow refreshes it; a week of missed runs is when the fallback should take over.
MAX_STALENESS = pd.Timedelta(days=8)
POOL_COLUMNS = ["origin", "timestamp", "demand_mw", "prediction", "abs_error", "model_version"]


def build_pool(
    history: pd.DataFrame, end: pd.Timestamp, origins: int = POOL_ORIGINS,
    horizon_hours: int = HORIZON_HOURS, model_factory=HistGradientBoostingForecast,
) -> pd.DataFrame:
    """The shipped model's 24-hour errors at the `origins` daily origins before `end`.

    Each origin is `end - k * 24h`, so its whole horizon is observed by `end`. Training
    slices are time based (`timestamp <= origin`), never positional.
    """
    data = history.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    actuals = data.set_index("timestamp").demand_mw
    end = pd.Timestamp(end)
    parts = []
    for k in range(origins, 0, -1):
        origin = end - pd.Timedelta(hours=k * horizon_hours)
        target = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h", tz="UTC"
        )
        prediction = model_factory().fit(data[data.timestamp <= origin]).predict(target)
        actual = actuals.reindex(target).to_numpy(dtype=float)
        parts.append(pd.DataFrame({
            "origin": origin, "timestamp": target, "demand_mw": actual,
            "prediction": np.round(prediction, 1),
            "abs_error": np.round(np.abs(actual - prediction), 1),
            "model_version": MODEL_VERSION,
        }))
    if not parts:
        return pd.DataFrame(columns=POOL_COLUMNS)
    return pd.concat(parts, ignore_index=True)[POOL_COLUMNS]


def write_pool(pool: pd.DataFrame, path: str | Path = POOL_PATH) -> bool:
    """Write the pool; False (and no write) when the bytes would not change."""
    path = Path(path)
    text = pool.to_csv(index=False)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def load_pool(path: str | Path = POOL_PATH) -> pd.DataFrame:
    """The committed pool, or an empty frame when there is none or it cannot be read."""
    path = Path(path)
    try:
        pool = pd.read_csv(path, dtype={"model_version": str})
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=POOL_COLUMNS)
    if not set(POOL_COLUMNS) <= set(pool.columns):
        return pd.DataFrame(columns=POOL_COLUMNS)
    pool["origin"] = pd.to_datetime(pool.origin, utc=True)
    pool["timestamp"] = pd.to_datetime(pool.timestamp, utc=True)
    return pool


def pool_spread(
    pool: pd.DataFrame | None, cutoff: pd.Timestamp, level: float = 0.95,
    n_origins: int = CALIBRATION_ORIGINS, horizon_hours: int = HORIZON_HOURS,
) -> float | None:
    """Half-width of the band for a forecast issued at `cutoff`, or None to fall back.

    Uses only origins whose whole horizon ends at or before `cutoff`, so the band never
    sees hours that were not yet observed when the forecast was issued, and only rows
    from the shipped model generation. None when fewer than `n_origins` qualify, when the
    newest is more than `MAX_STALENESS` older than `cutoff`, or when the errors cannot
    support `level`.
    """
    if pool is None or len(pool) == 0:
        return None
    cutoff = pd.Timestamp(cutoff)
    rows = pool[pool.model_version.astype(str) == MODEL_VERSION]
    ends = rows.origin + pd.Timedelta(hours=horizon_hours)
    rows = rows[ends <= cutoff]
    chosen = sorted(rows.origin.unique())[-n_origins:]
    if len(chosen) < n_origins or cutoff - chosen[-1] > MAX_STALENESS:
        return None
    spread = conformal_quantile(rows[rows.origin.isin(chosen)].abs_error, level)
    return spread if np.isfinite(spread) else None
