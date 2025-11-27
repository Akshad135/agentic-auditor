import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

_embeddings_instance = None


def get_embeddings():
    """Load embeddings once and reuse (GPU-heavy)."""
    global _embeddings_instance

    if _embeddings_instance is None:
        print(f"Loading embeddings: {EMBEDDING_MODEL}")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings_instance


def get_vector_store():
    """Connect to the existing Qdrant collection."""
    client = QdrantClient(path=str(VECTOR_DB_PATH))
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def retrieve_relevant_rules(query: str, k: int = 2) -> list[str]:
    """Return top-k similar rules for a query."""
    try:
        db = get_vector_store()
        results = db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
