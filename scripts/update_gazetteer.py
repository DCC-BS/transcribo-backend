"""Refresh the street-name asset from Basel open data.

Run via ``mise run gazetteer:update``. The object-name list next to it is curated by
hand and is never touched here.
"""

import csv
import io
import sys
from pathlib import Path

import httpx

DATASET_URL = (
    "https://data.bs.ch/api/explore/v2.1/catalog/datasets/100189/exports/csv?select=strassenname&delimiter=%3B"
)
TARGET = Path(__file__).parent.parent / "src" / "transcribo_backend" / "assets" / "basel_streets.txt"
MIN_EXPECTED = 1_000

HEADER = (
    "# Basler Strassennamen, Open-Data-Datensatz 100189 (data.bs.ch).\n"
    "# Generiert - nicht von Hand bearbeiten: 'mise run gazetteer:update'.\n"
)


def main() -> None:
    response = httpx.get(DATASET_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()

    rows = csv.DictReader(io.StringIO(response.text.lstrip("﻿")), delimiter=";")
    names = sorted({name for row in rows if (name := (row.get("strassenname") or "").strip())})

    # A partial response would silently shrink the gazetteer and degrade correction
    # quality without any error, so refuse to write an implausibly short list.
    if len(names) < MIN_EXPECTED:
        sys.exit(f"only {len(names)} street names received, expected at least {MIN_EXPECTED} - not writing")

    TARGET.write_text(HEADER + "\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(names)} street names -> {TARGET.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
