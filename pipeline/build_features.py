"""
pipeline/build_features.py
Reads raw postings parquet, engineers features, creates seniority label, saves processed parquet.

Label logic (genuinely senior = 1):
  - Compensation above category median AND
  - 3+ hard technical skills in description AND
  - Title does NOT match inflated patterns (ninja, rockstar, guru, wizard)
"""

import re
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.append(".")
from config.settings import RAW_DATA_PATH, FEATURES_DATA_PATH

# --- Keyword lists ---

SENIORITY_KEYWORDS = ["senior", "sr.", "lead", "principal", "staff", "architect", "manager", "director", "vp", "head of"]

INFLATED_TITLE_PATTERNS = ["ninja", "rockstar", "guru", "wizard", "superhero", "evangelist", "hacker"]

HARD_SKILLS = [
    "python", "java", "scala", "c++", "rust", "go", "kubernetes", "docker",
    "spark", "kafka", "airflow", "terraform", "aws", "gcp", "azure",
    "pytorch", "tensorflow", "sql", "mongodb", "redis", "elasticsearch"
]

YOE_PATTERN = re.compile(r"(\d+)\+?\s*years?\s*(of\s*)?experience", re.IGNORECASE)


# --- Feature functions (one job at a time for clarity) ---

def has_seniority_keyword(title: str) -> int:
    title_lower = title.lower()
    return int(any(kw in title_lower for kw in SENIORITY_KEYWORDS))


def is_inflated_title(title: str) -> int:
    title_lower = title.lower()
    return int(any(p in title_lower for p in INFLATED_TITLE_PATTERNS))


def count_hard_skills(description: str) -> int:
    desc_lower = description.lower()
    return sum(1 for skill in HARD_SKILLS if skill in desc_lower)


def extract_min_yoe(description: str) -> float:
    matches = YOE_PATTERN.findall(description)
    if not matches:
        return 0.0
    # Take the minimum mentioned — most conservative requirement
    return float(min(int(m[0]) for m in matches))


def description_length(description: str) -> int:
    return len(description.split())


def compensation_mentioned(description: str) -> int:
    return int(bool(re.search(r"\$[\d,]+|salary|compensation|pay range", description, re.IGNORECASE)))


def remote_flag(description: str) -> int:
    return int("remote" in description.lower())


def salary_midpoint(row: pd.Series) -> float:
    """Average of min/max salary. Falls back to 0 if both are missing."""
    if pd.notna(row["salary_min"]) and pd.notna(row["salary_max"]):
        return (row["salary_min"] + row["salary_max"]) / 2
    elif pd.notna(row["salary_max"]):
        return row["salary_max"]
    elif pd.notna(row["salary_min"]):
        return row["salary_min"]
    return 0.0


# --- Label creation ---

def create_label(df: pd.DataFrame) -> pd.Series:
    """
    Genuinely senior (1) if ALL three conditions hold:
      1. salary_midpoint > category median salary
      2. hard_skill_count >= 3
      3. is_inflated_title == 0
    """
    category_median = df.groupby("category")["salary_midpoint"].transform("median")

    above_median   = df["salary_midpoint"] > category_median
    enough_skills  = df["hard_skill_count"] >= 3
    not_inflated   = df["is_inflated_title"] == 0

    return (above_median & enough_skills & not_inflated).astype(int)


# --- Main ---

def build_features(raw_path: str = RAW_DATA_PATH, out_path: str = FEATURES_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_parquet(raw_path)

    logger.info("Engineering features...")

    df["seniority_in_title"] = df["title"].apply(has_seniority_keyword)
    df["is_inflated_title"]  = df["title"].apply(is_inflated_title)
    df["hard_skill_count"]   = df["description"].apply(count_hard_skills)
    df["min_yoe"]            = df["description"].apply(extract_min_yoe)
    df["desc_word_count"]    = df["description"].apply(description_length)
    df["comp_mentioned"]     = df["description"].apply(compensation_mentioned)
    df["remote_flag"]        = df["description"].apply(remote_flag)
    df["salary_midpoint"]    = df.apply(salary_midpoint, axis=1)

    logger.info("Creating seniority label...")
    df["is_genuinely_senior"] = create_label(df)

    label_dist = df["is_genuinely_senior"].value_counts(normalize=True).round(3)
    logger.info(f"Label distribution:\n{label_dist}")

    # Keep only model-ready columns + identifiers
    feature_cols = [
        "id", "title", "company", "category", "description",
        "seniority_in_title", "is_inflated_title", "hard_skill_count",
        "min_yoe", "desc_word_count", "comp_mentioned", "remote_flag",
        "salary_midpoint", "is_genuinely_senior"
    ]
    df = df[feature_cols]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Features saved to {out_path} — shape: {df.shape}")

    return df


if __name__ == "__main__":
    build_features()