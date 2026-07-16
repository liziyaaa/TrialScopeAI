"""Application-wide configuration with safe defaults."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

DEMO_NCT_ID = "NCT02347774"
DEMO_SOURCE_URL = f"https://clinicaltrials.gov/study/{DEMO_NCT_ID}"

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200
MIN_EXTRACTED_TEXT_CHARS = 200
MAX_LLM_CHUNK_CHARS = 30_000
MAX_LLM_TOTAL_CHARS = 60_000
MAX_LIVE_CALLS_PER_SESSION = 3
MAX_LIVE_CALLS_PER_HOUR = 30

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_TIMEOUT_SECONDS = 90.0
DEEPSEEK_MAX_RETRIES = 2


def env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean flag from the environment."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
