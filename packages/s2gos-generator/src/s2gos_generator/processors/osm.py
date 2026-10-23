"""OpenStreetMap access shared by the OSM-based processors: Overpass fetching, file loading, tag parsing."""

import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .._version import get_version

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MAX_RETRIES = 5
OVERPASS_RETRY_DELAY_S = 15
OVERPASS_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class OverpassFetchError(RuntimeError):
    """Raised when the Overpass API fails on every retry attempt."""


def _is_transient_urlerror(exc: urllib.error.URLError) -> bool:
    """Return True if a URLError wraps a known transient transport failure."""
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (ConnectionResetError, socket.timeout, ssl.SSLEOFError))


def fetch_overpass(query: str) -> dict:
    """Run an Overpass QL query, retrying on transient failures."""
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": f"s2gos-generator/{get_version()}",
    }
    for attempt in range(1, OVERPASS_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                OVERPASS_URL, data=data, method="POST", headers=headers
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            if isinstance(exc, urllib.error.HTTPError):
                retryable, cause = exc.code in OVERPASS_RETRYABLE_STATUS, exc.code
            else:
                retryable, cause = _is_transient_urlerror(exc), repr(exc.reason)
            if not retryable or attempt == OVERPASS_MAX_RETRIES:
                raise OverpassFetchError(
                    f"Overpass API request failed ({type(exc).__name__}): {exc}"
                )
            logging.info(
                "Overpass error %s, retrying in %ds (attempt %d/%d)",
                cause,
                OVERPASS_RETRY_DELAY_S,
                attempt,
                OVERPASS_MAX_RETRIES,
            )
            time.sleep(OVERPASS_RETRY_DELAY_S)


def load_osm_data(cfg, query: str, label: str) -> Optional[dict]:
    """Fetch OSM data from Overpass or load it from ``cfg.file_path``, per ``cfg.source``."""
    if cfg.source == "overpass":
        logging.info("Fetching %s from Overpass API", label)
        return fetch_overpass(query)

    if cfg.source == "file":
        logging.info("Loading %s from file: %s", label, cfg.file_path)
        try:
            with open(cfg.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
            logging.warning("Failed to load %s file: %s", label, exc)
            return None

    logging.warning("Unknown %s data source: %s", label, cfg.source)
    return None


def parse_osm_width(width_str: str) -> Optional[float]:
    """Parse an OSM width tag value. Handles '5', '5.5', '5 m', '5.5m' formats."""
    s = width_str.strip().lower()
    if s.endswith("m"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None
