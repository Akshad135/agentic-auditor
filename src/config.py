import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# File Paths
VECTOR_DB_PATH = DATA_DIR / "qdrant_db"
RAW_DATA_CSV = DATA_DIR / "real_legal_clauses.csv"
DERIVED_PLAYBOOK_JSON = DATA_DIR / "derived_playbook.json"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
REPORT_DIR = DATA_DIR / "audit_reports"

# Input Settings
INPUT_FILENAME = "test_contract.pdf"
INPUT_PDF_PATH = RAW_PDF_DIR / INPUT_FILENAME

# Model Settings
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"

# Vector Store Settings
COLLECTION_NAME = "legal_playbook"
VECTOR_DIMENSION = 768

# Groq Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- OPERATIONAL SETTINGS ---
MAX_RETRIES = 3 