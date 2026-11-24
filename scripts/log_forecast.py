#!/usr/bin/env python3
"""Issue today's 24-hour forecast and append it to the committed forecast log.

The log is the record of what the model actually said at the time, which a
retrospective replay cannot manufacture after the outcome is known.
"""

import argparse
import os

from energy_forecast.calibration import POOL_PATH, build_pool, write_pool
from energy_forecast.service import ForecastService
from energy_forecast.verification import DEFAULT_LOG_PATH, append_forecast_log, forecast_record

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=str(DEFAULT_LOG_PATH), help="forecast log CSV")
parser.add_argument("--pool-path", default=str(POOL_PATH), help="calibration pool CSV")
parser.add_argument("--dry-run", action="store_true", help="print the rows, write nothing")
parser.add_argument("--allow-stale", action="store_true", help="log even without live data")
parser.add_argument(
    "--timeout-seconds", type=float, default=10.0, help="live SMARD fetch deadline"
)
args = parser.parse_args()

service = ForecastService(live=True, timeout_seconds=args.timeout_seconds)

# A forecast issued from a stale snapshot's origin is not "what we published today" —
# it is a hindcast wearing today's date. Fail loudly rather than log a fiction; the log
# is only evidence if every row was issued before its hours were observable.
if not service.is_live and not args.allow_stale:
    print(f"Live data unavailable: {service.live_warning}")
    print("Refusing to log a forecast from a stale origin (pass --allow-stale to override).")
    raise SystemExit(1)

# Refresh the split-conformal calibration pool from the same live history first, so the
# band logged today comes from errors up to today's origin rather than yesterday's file.
# ~35 model fits (~30 s), which is why it is built here and committed, not at app start.
pool = build_pool(service.history, end=service.last_observed)
service.use_calibration_pool(pool)
print(f"Band: {service.interval_method}")

rows = forecast_record(service)
origin = rows.origin.iloc[0]

if args.dry_run:
    print(rows.to_string(index=False))
    print(f"Dry run: nothing written to {args.path} or {args.pool_path}")
    changed = False
else:
    log, changed = append_forecast_log(args.path, rows)
    pool_changed = write_pool(pool, args.pool_path)
    changed = changed or pool_changed
    print(f"Calibration pool {'updated' if pool_changed else 'unchanged'} at {args.pool_path}")
    origins = int(log.origin.nunique())

print(f"Origin: {origin.isoformat()}")
print(f"Issued {len(rows)} rows; file changed: {changed}")
if not args.dry_run:
    print(f"Log now holds {origins} distinct origins at {args.path}")

github_output = os.environ.get("GITHUB_OUTPUT")
if github_output:
    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"origin={origin.isoformat()}\n")
        handle.write(f"changed={'true' if changed else 'false'}\n")
