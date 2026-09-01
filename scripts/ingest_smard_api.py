#!/usr/bin/env python3
"""Fetch the latest German grid-load chunks from SMARD's official API."""

import argparse

from energy_forecast.quality import validate_demand
from energy_forecast.smard_api import SMARDAPIClient

parser = argparse.ArgumentParser()
parser.add_argument("--weeks", type=int, default=52, help="number of latest weekly chunks")
args = parser.parse_args()
if args.weeks < 1:
    parser.error("--weeks must be positive")

# download() runs the quality gate itself and refuses to write on a hard failure;
# re-running it here only to print the per-check detail for the operator.
data = SMARDAPIClient().download(weeks=args.weeks)
report = validate_demand(data)
print(f"Saved {len(data):,} hourly observations to data/clean/demand_hourly.csv")
print(f"Quality gate — {report.summary()}")
for check in report.checks:
    if not check.passed:
        print(f"  [{check.severity.upper()}] {check.name}: {check.detail}")
