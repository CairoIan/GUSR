#!/usr/bin/env python3
"""Download the final GeoGCG image requests from licensed image endpoints."""

import argparse
import csv
import io
import json
import math
import os
import time
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ROOT / "download" / "image_requests.csv"
CONFIG = ROOT / "download" / "image_source.json"
STATUS = ROOT / "download" / "download_status.csv"
EXPECTED = {
    "zhengzhou": {"sat": 6613, "street": 8990},
    "beijing": {"sat": 2668, "street": 5301},
    "changsha": {"sat": 3111, "street": 4006},
    "shanghai": {"sat": 5768, "street": 11442},
    "wuhan": {"sat": 5582, "street": 10882},
    "xian": {"sat": 4506, "street": 8787},
}


def tile_xy(lon, lat, zoom=18):
    lat_radians = math.radians(float(lat))
    n = 2**zoom
    x = int((float(lon) + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_radians) + 1 / math.cos(lat_radians)) / math.pi) / 2 * n)
    return x, y


def fetch_image(session, url):
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB"), response.headers
        except (requests.RequestException, OSError) as error:
            last_error = error
            time.sleep(attempt + 1)
    raise RuntimeError(str(last_error))


def download_satellite(session, row, config):
    x, y = tile_xy(row["lon"], row["lat"])
    mosaic = None
    dates = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            fields = dict(row, x=x + dx, y=y + dy, z=18,
                          api_key=os.environ.get("SATELLITE_API_KEY", ""))
            image, headers = fetch_image(session, config["satellite_tile_url"].format(**fields))
            if mosaic is None:
                tile_size = image.width
                mosaic = Image.new("RGB", (tile_size * 3, tile_size * 3))
            mosaic.paste(image, ((dx + 1) * tile_size, (dy + 1) * tile_size))
            date = headers.get(config.get("capture_date_header", "X-Capture-Date"))
            if date:
                dates.add(date)
    left = (mosaic.width - 700) // 2
    top = (mosaic.height - 700) // 2
    return mosaic.crop((left, top, left + 700, top + 700)), ";".join(sorted(dates))


def download_street(session, row, config):
    fields = dict(row, api_key=os.environ.get("STREET_API_KEY", ""))
    image, headers = fetch_image(session, config["street_view_url"].format(**fields))
    if image.size != (720, 720):
        image = image.resize((720, 720), Image.Resampling.LANCZOS)
    return image, headers.get(config.get("capture_date_header", "X-Capture-Date"), "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--requests", type=Path, default=REQUESTS)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--city", help="Optional one-city download")
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Configure an authorized image source in {args.config}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config.get("satellite_tile_url") or not config.get("street_view_url"):
        parser.error("Both satellite_tile_url and street_view_url are required")

    args.status.parent.mkdir(parents=True, exist_ok=True)
    status_exists = args.status.exists()
    counts = {city: {"sat": 0, "street": 0, "missing": 0} for city in EXPECTED}
    with args.requests.open(newline="", encoding="utf-8") as source, args.status.open("a", newline="", encoding="utf-8") as target, requests.Session() as session:
        writer = csv.DictWriter(target, fieldnames=["image_path", "city", "modality", "status", "actual_capture_date", "error"])
        if not status_exists:
            writer.writeheader()
        for row in csv.DictReader(source):
            if args.city and row["city"] != args.city:
                continue
            path = ROOT / row["image_path"]
            city_counts = counts[row["city"]]
            if path.is_file() and path.stat().st_size > 0:
                city_counts[row["modality"]] += 1
                writer.writerow(dict(image_path=row["image_path"], city=row["city"], modality=row["modality"], status="existing", actual_capture_date="", error=""))
                continue
            try:
                image, actual_date = (download_satellite(session, row, config) if row["modality"] == "sat" else download_street(session, row, config))
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="JPEG", quality=95)
                writer.writerow(dict(image_path=row["image_path"], city=row["city"], modality=row["modality"], status="downloaded", actual_capture_date=actual_date, error=""))
                city_counts[row["modality"]] += 1
            except (RuntimeError, ValueError, KeyError) as error:
                writer.writerow(dict(image_path=row["image_path"], city=row["city"], modality=row["modality"], status="missing", actual_capture_date="", error=str(error)))
                city_counts["missing"] += 1
            target.flush()
            time.sleep(float(config.get("delay_seconds", 0.2)))
    incomplete = False
    for city in ([args.city] if args.city else EXPECTED):
        actual, expected = counts[city], EXPECTED[city]
        print(f"{city}: satellite {actual['sat']}/{expected['sat']}, street {actual['street']}/{expected['street']}, missing {actual['missing']}")
        incomplete |= any(actual[k] != expected[k] for k in ("sat", "street"))
    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
