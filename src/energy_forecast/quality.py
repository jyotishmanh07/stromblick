"""Explicit data-quality gate for the canonical demand frame.

`canonicalize_demand()` already rejects structurally broken input -- missing columns,
unparseable or ambiguous timestamps, no numeric values. This module handles the class
of problem that parses cleanly but is still wrong: a plausible-looking frame carrying
a decimal-shifted value, a stale snapshot, or a week of holes.

Checks are graded. A `severity="error"` failure means the frame should not be written
or trained on and raises `DataValidationError`; a `severity="warning"` failure is
recorded and surfaced but does not stop the pipeline, because gaps are a normal
property of a SMARD snapshot -- the project reports them rather than imputing them.

At production scale this role belongs to a dedicated framework (Great Expectations,
dbt tests) with a scheduler. At this size an explicit, tested function is the honest
equivalent, and it keeps the dependency surface small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data import DataValidationError, missing_hourly_timestamps

# Germany's grid load has sat roughly between 30 and 85 GW for years. The band is
# deliberately wide: it is a decimal-point and unit-swap trap, not a forecast.
MIN_PLAUSIBLE_MW = 20_000.0
MAX_PLAUSIBLE_MW = 100_000.0


@dataclass
class Check:
    name: str
    passed: bool
    severity: str
    detail: str
    value: Any = None


@dataclass
class QualityReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> QualityReport:
        if self.errors:
            joined = "; ".join(f"{c.name}: {c.detail}" for c in self.errors)
            raise DataValidationError(f"data-quality gate failed -- {joined}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name, "passed": c.passed, "severity": c.severity,
                    "detail": c.detail, "value": c.value,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        failed = len(self.errors) + len(self.warnings)
        head = "PASS" if self.passed else "FAIL"
        return (
            f"{head}: {len(self.checks) - failed}/{len(self.checks)} checks clean, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


def validate_demand(
    frame: pd.DataFrame, max_gap_fraction: float = 0.02, max_nan_fraction: float = 0.05,
    max_staleness_hours: float | None = None, now: pd.Timestamp | None = None,
) -> QualityReport:
    """Run the gate over a canonicalized demand frame.

    `max_staleness_hours` is opt-in: the committed snapshot is deliberately historical,
    so freshness is only meaningful when checking a just-collected ingest.
    """
    checks: list[Check] = []

    def add(name, passed, severity, detail, value=None):
        checks.append(Check(name, bool(passed), severity, detail, value))

    required = {"timestamp", "demand_mw"}
    if not required.issubset(frame.columns):
        add("schema", False, "error", f"missing columns {sorted(required - set(frame.columns))}")
        return QualityReport(checks)
    add("schema", True, "error", "timestamp and demand_mw present")

    if frame.empty:
        add("non_empty", False, "error", "frame contains no rows", 0)
        return QualityReport(checks)
    add("non_empty", True, "error", f"{len(frame):,} rows", int(len(frame)))

    stamps = pd.to_datetime(frame.timestamp, utc=True)
    add(
        "timezone_aware_utc", frame.timestamp.dt.tz is not None, "error",
        "timestamps must be tz-aware UTC before they cross a module boundary",
    )
    add(
        "chronological", stamps.is_monotonic_increasing, "error",
        "timestamps must be sorted ascending",
    )
    duplicates = int(stamps.duplicated().sum())
    add(
        "unique_timestamps", duplicates == 0, "error",
        f"{duplicates} duplicate timestamp(s)" if duplicates else "no duplicates", duplicates,
    )

    values = pd.to_numeric(frame.demand_mw, errors="coerce")
    observed = values.dropna()
    if observed.empty:
        add("has_observations", False, "error", "every demand value is null", 0)
        return QualityReport(checks)
    add("has_observations", True, "error", f"{len(observed):,} observed hours", int(len(observed)))

    out_of_band = int(
        ((observed < MIN_PLAUSIBLE_MW) | (observed > MAX_PLAUSIBLE_MW)).sum()
    )
    add(
        "plausible_range", out_of_band == 0, "error",
        f"{out_of_band} value(s) outside {MIN_PLAUSIBLE_MW:,.0f}-{MAX_PLAUSIBLE_MW:,.0f} MW "
        f"(observed {observed.min():,.0f}-{observed.max():,.0f})",
        out_of_band,
    )
    non_positive = int((observed <= 0).sum())
    add(
        "positive_demand", non_positive == 0, "error",
        f"{non_positive} non-positive value(s)", non_positive,
    )

    nan_fraction = float(values.isna().mean())
    add(
        "nan_fraction", nan_fraction <= max_nan_fraction, "warning",
        f"{nan_fraction:.2%} of hours indexed but unpublished (limit {max_nan_fraction:.0%})",
        round(nan_fraction, 5),
    )

    gaps = missing_hourly_timestamps(frame)
    expected = len(frame) + len(gaps)
    gap_fraction = len(gaps) / expected if expected else 0.0
    add(
        "hourly_continuity", gap_fraction <= max_gap_fraction, "warning",
        f"{len(gaps)} missing hourly timestamp(s), {gap_fraction:.2%} of the expected index "
        f"(limit {max_gap_fraction:.0%}); gaps are reported, never imputed",
        len(gaps),
    )

    if max_staleness_hours is not None:
        reference = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        age = (reference - stamps.max()).total_seconds() / 3600
        add(
            "freshness", age <= max_staleness_hours, "error",
            f"latest observation is {age:,.1f}h old (limit {max_staleness_hours:,.0f}h)",
            round(age, 2),
        )

    return QualityReport(checks)
