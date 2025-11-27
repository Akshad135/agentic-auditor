import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_DIMENSION,
)


def initialize_memory():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Initializing Qdrant at {VECTOR_DB_PATH}")
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    client = QdrantClient(path=str(VECTOR_DB_PATH))

    # Reset collection if it exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

    # Test rules for indexing
    rules_text = [
        "Non-compete clauses must not exceed 2 years.",
        "Confidentiality obligations must be mutual.",
        "Legal jurisdiction must be the State of Delaware.",
        "Payment terms are Net-30. Net-60 requires VP approval.",
    ]

    docs = [
        Document(page_content=rule, metadata={"source": "manual_playbook"})
        for rule in rules_text
    ]

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    qdrant.add_documents(docs)
    print(f"Indexed {len(docs)} rules")

    return qdrant


if __name__ == "__main__":
    db = initialize_memory()
    db.client.close()
