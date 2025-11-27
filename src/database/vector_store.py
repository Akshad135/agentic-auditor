import sys
from pathlib import Path

# Ensure we can find the config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# Global cache to prevent reloading the 500MB model on every call
_embeddings_instance = None

def get_embeddings():
    """
    Singleton pattern to load the embedding model onto GPU only once.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"🔌 Loading {EMBEDDING_MODEL} for retrieval...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings_instance

def get_vector_store():
    """
    Connects to the existing Qdrant database on disk.
    """
    client = QdrantClient(path=str(VECTOR_DB_PATH))
    embeddings = get_embeddings()
    
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

def retrieve_relevant_rules(query: str, k: int = 2) -> list[str]:
    """
    The main function Agents will call to check the playbook.
    """
    try:
        db = get_vector_store()
        results = db.similarity_search(query, k=k)

        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        return []