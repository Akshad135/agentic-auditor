import sys
import json
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
    DERIVED_PLAYBOOK_JSON,
)


def initialize_memory():
    print("Initializing knowledge injection")
    print(f"Target DB: {VECTOR_DB_PATH}")

    if not DERIVED_PLAYBOOK_JSON.exists():
        print(f"Playbook not found: {DERIVED_PLAYBOOK_JSON}")
        print("Run derive_playbook.py first")
        return

    with open(DERIVED_PLAYBOOK_JSON, "r", encoding="utf-8") as f:
        policies = json.load(f)

    print(f"Loaded policies: {len(policies)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Initializing Qdrant")
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    client = QdrantClient(path=str(VECTOR_DB_PATH))

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' reset")

    docs = []
    for policy in policies:
        content = (
            f"POLICY: {policy.get('policy_name')}\n"
            f"CATEGORY: {policy.get('category')}\n"
            f"RULE: {policy.get('text')}"
        )
        metadata = {
            "source": "derived_playbook",
            "category": policy.get("category"),
            "policy_name": policy.get("policy_name"),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    print(f"Indexing {len(docs)} documents")
    qdrant = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    qdrant.add_documents(docs)

    print("Knowledge base initialized")
    return qdrant


if __name__ == "__main__":
    db = initialize_memory()
    if db:
        db.client.close()
