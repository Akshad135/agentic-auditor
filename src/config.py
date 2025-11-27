import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_PATH = DATA_DIR / "qdrant_db"

# Model Settings
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"

# Vector Store Settings
COLLECTION_NAME = "legal_playbook"
VECTOR_DIMENSION = 768

# Groq Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")