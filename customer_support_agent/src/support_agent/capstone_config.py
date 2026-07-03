from __future__ import annotations

from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
MAX_SEQ_LENGTH = 512
TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

SYSTEM_PROMPT = (
    "You are an intent router for customer support. Return only valid JSON "
    "with keys intent and category. Do not include explanation text."
)
