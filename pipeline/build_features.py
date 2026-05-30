"""
pipeline/build_features.py
Reads raw postings parquet, engineers features, creates seniority label, saves processed parquet.

Label logic (genuinely senior = 1):
  - (seniority keyword in title  OR  2+ hard skills)  AND  not inflated title
  Relaxed to OR because Adzuna truncates descriptions — skill count is naturally low.
"""

import re
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.append(".")
from config.settings import RAW_DATA_PATH, FEATURES_DATA_PATH

# --- Keyword lists ---
SENIORITY_KEYWORDS      = ["senior", "sr.", "lead", "principal", "staff", "architect",
                            "manager", "director", "vp", "head of"]
INFLATED_TITLE_PATTERNS = ["ninja", "rockstar", "guru", "wizard", "superhero", "evangelist", "hacker"]
HARD_SKILLS             = [
    "python", "java", "scala", "c++", "rust", "go", "kubernetes", "docker",
    "spark", "kafka", "airflow", "terraform", "aws", "gcp", "azure",
    "pytorch", "tensorflow", "sql", "mongodb", "redis", "elasticsearch"
]
YOE_PATTERN = re.compile(r"(\d+)\+?\s*years?\s*(of\s*)?experience", re.IGNORECASE)

FEATURE_COLS = [
    "min_yoe", "desc_word_count", "comp_mentioned", "remote_flag",
    "salary_midpoint", "is_genuinely_senior",
]


# --- Feature functions ---
def has_seniority_keyword(title: str) -> int:
    return int(any(kw in title.lower() for kw in SENIORITY_KEYWORDS))

def is_inflated_title(title: str) -> int:
    return int(any(p in title.lower() for p in INFLATED_TITLE_PATTERNS))

def count_hard_skills(description: str) -> int:
    desc_lower = description.lower()
    return sum(1 for skill in HARD_SKILLS if skill in desc_lower)

def extract_min_yoe(description: str) -> float:
    matches = YOE_PATTERN.findall(description)
    return float(min(int(m[0]) for m in matches)) if matches else 0.0

def description_length(description: str) -> int:
    return len(description.split())

def compensation_mentioned(description: str) -> int:
    return int(bool(re.search(r"\$[\d,]+|salary|compensation|pay range", description, re.IGNORECASE)))

def remote_flag(description: str) -> int:
    return int("remote" in description.lower())

def salary_midpoint(row: pd.Series) -> float:
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
    Genuinely senior (1) if:
      - (seniority keyword in title  OR  2+ hard skills in description)
      AND title is NOT inflated
    """
    skilled_or_titled = (df["seniority_in_title"] == 1) | (df["hard_skill_count"] >= 2)
    return (skilled_or_titled & (df["is_inflated_title"] == 0)).astype(int)


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

    if df["is_genuinely_senior"].mean() < 0.05:
        logger.warning("⚠  Positive rate still below 5% — check raw data quality")

    # Drop text/ID columns — XGBoost only accepts numeric
    df_model = df[FEATURE_COLS].copy()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_model.to_parquet(out_path, index=False)
    logger.info(f"Features saved to {out_path} — shape: {df_model.shape}")

    return df_model


if __name__ == "__main__":
    build_features()