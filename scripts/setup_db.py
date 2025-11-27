import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import from our new config file
from src.config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_DIMENSION

def initialize_memory():
    print(f"--- 1. Loading Embedding Model ({EMBEDDING_MODEL}) ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True} 
    )

    print(f"\n--- 2. Initializing Qdrant at {VECTOR_DB_PATH} ---")
    
    # Create the directory if it doesn't exist
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    client = QdrantClient(path=str(VECTOR_DB_PATH))

    # Reset Collection
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE)
    )

    # --- Dummy Data for Phase 3 Testing ---
    rules_text = [
        "The company typically allows non-compete clauses, but they must not exceed 2 years in duration.",
        "Confidentiality obligations must be mutual. We do not accept one-way NDAs.",
        "Jurisdiction for all legal disputes must be the State of Delaware.",
        "Payment terms are standard Net-30. Net-60 is acceptable only with VP approval."
    ]
    docs = [Document(page_content=rule, metadata={"source": "manual_playbook"}) for rule in rules_text]

    qdrant = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings
    )

    qdrant.add_documents(docs)
    print(f"✅ Indexed {len(docs)} rules.")
    return qdrant

if __name__ == "__main__":
    db = initialize_memory()
    db.client.close()