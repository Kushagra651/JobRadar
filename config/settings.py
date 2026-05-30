import os
from dotenv import load_dotenv
 
load_dotenv()
 
# --- Adzuna API ---
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
 
# --- HuggingFace ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
 
# --- Qdrant ---
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "job_postings"
 
# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
 
# --- ML ---
MODEL_PATH       = "ml/artifacts/xgb_model.json"
ROC_AUC_TARGET   = 0.78
TOP_K_RETRIEVAL  = 5
 
# --- Paths ---
RAW_DATA_PATH      = "data/raw/postings.parquet"
FEATURES_DATA_PATH = "data/processed/features.parquet"
 