#!/usr/bin/env python3
"""Fetch the latest German grid-load chunks from SMARD's official API."""

import argparse

from energy_forecast.smard_api import SMARDAPIClient

parser = argparse.ArgumentParser()
parser.add_argument("--weeks", type=int, default=52, help="number of latest weekly chunks")
args = parser.parse_args()
if args.weeks < 1:
    parser.error("--weeks must be positive")
data = SMARDAPIClient().download(weeks=args.weeks)
print(f"Saved {len(data):,} hourly observations to data/clean/demand_hourly.csv")
