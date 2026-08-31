#!/usr/bin/env python3
"""Download and validate a SMARD export."""

import argparse

from energy_forecast.data import DEFAULT_SMARD_URL, SMARDClient

parser = argparse.ArgumentParser()
parser.add_argument("--url", default=DEFAULT_SMARD_URL)
args = parser.parse_args()
result = SMARDClient(args.url).download()
print(f"Saved {len(result.data):,} hourly observations to {result.metadata_path}")
