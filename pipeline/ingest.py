"""
pipeline/ingest.py
Fetches job postings from Adzuna API and saves them as a parquet file.
Target: 2000-5000 postings across multiple job categories.
"""

import requests
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.append(".")
from config.settings import ADZUNA_APP_ID, ADZUNA_API_KEY, ADZUNA_BASE_URL, RAW_DATA_PATH

# Job categories to pull — diversify so the ML model sees variety
CATEGORIES = ["it-jobs", "engineering-jobs", "data-science-jobs", "product-management-jobs"]
COUNTRY    = "us"
RESULTS_PER_PAGE = 50   # Adzuna max per request
PAGES_PER_CATEGORY = 10  # 10 pages × 50 results × 4 categories = 2000 postings


def fetch_page(category: str, page: int) -> list[dict]:
    """Fetch one page of results for a given category."""
    url = f"{ADZUNA_BASE_URL}/{COUNTRY}/search/{page}"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "category": category,
        "results_per_page": RESULTS_PER_PAGE,
        "content-type": "application/json",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", [])


def parse_posting(raw: dict) -> dict:
    """Extract only the fields we need from a raw Adzuna result."""
    return {
        "id":          raw.get("id", ""),
        "title":       raw.get("title", ""),
        "company":     raw.get("company", {}).get("display_name", ""),
        "location":    raw.get("location", {}).get("display_name", ""),
        "description": raw.get("description", ""),
        "salary_min":  raw.get("salary_min"),       # may be None
        "salary_max":  raw.get("salary_max"),       # may be None
        "category":    raw.get("category", {}).get("label", ""),
        "created":     raw.get("created", ""),
    }


def ingest(pages_per_category: int = PAGES_PER_CATEGORY) -> pd.DataFrame:
    """Fetch all categories, parse, deduplicate, and return a DataFrame."""
    all_postings = []

    for category in CATEGORIES:
        logger.info(f"Fetching category: {category}")
        for page in range(1, pages_per_category + 1):
            try:
                results = fetch_page(category, page)
                all_postings.extend([parse_posting(r) for r in results])
                logger.debug(f"  Page {page}: {len(results)} results")
            except Exception as e:
                # Skip failed pages — don't let one bad page kill the run
                logger.warning(f"  Page {page} failed: {e}")

    df = pd.DataFrame(all_postings)
    df = df.drop_duplicates(subset="id")   # deduplicate by Adzuna ID
    logger.info(f"Total postings collected: {len(df)}")
    return df


def save(df: pd.DataFrame, path: str = RAW_DATA_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    df = ingest()
    save(df)