import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
# UPDATED: Use the official connector instead of community
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIGURATION ---
VECTOR_DB_PATH = "./local_qdrant_db"
COLLECTION_NAME = "legal_playbook"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def initialize_memory():
    print(f"--- 1. Loading Embedding Model ({EMBEDDING_MODEL_NAME}) on GPU ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True} 
    )
    print("✅ Model Loaded on CUDA.")

    print(f"\n--- 2. initializing Qdrant at {VECTOR_DB_PATH} ---")
    
    # Initialize the client explicitly
    client = QdrantClient(path=VECTOR_DB_PATH)

    # Check if collection exists and recreate it manually
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        print(f"   Existing collection '{COLLECTION_NAME}' found. Deleting...")
        client.delete_collection(COLLECTION_NAME)
    
    # Create collection with specific vector size (Nomic v1.5 is 768 dimensions)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    print(f"   Created new collection '{COLLECTION_NAME}'.")

    # --- Data Ingestion ---
    rules_text = [
        "The company typically allows non-compete clauses, but they must not exceed 2 years in duration.",
        "Confidentiality obligations must be mutual. We do not accept one-way NDAs.",
        "Jurisdiction for all legal disputes must be the State of Delaware.",
        "Payment terms are standard Net-30. Net-60 is acceptable only with VP approval."
    ]

    docs = [Document(page_content=rule, metadata={"source": "manual_playbook"}) for rule in rules_text]

    # UPDATED: Initialize LangChain wrapper using the new QdrantVectorStore class
    qdrant = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings
    )

    # Add documents
    qdrant.add_documents(docs)
    print(f"✅ Indexed {len(docs)} rules into Qdrant.")

    return qdrant

def test_retrieval(qdrant_instance):
    print("\n--- 3. Testing Retrieval ---")
    query = "What is the maximum duration for a non-compete?"
    print(f"❓ Query: {query}")

    # Search
    results = qdrant_instance.similarity_search(query, k=1)

    if results:
        print(f"💡 Found Rule: \"{results[0].page_content}\"")
        print("✅ System is working!")
    else:
        print("❌ No results found.")

if __name__ == "__main__":
    db = initialize_memory()
    test_retrieval(db)
    # Explicitly close the client to prevent "sys.meta_path is None" errors on exit
    print("--- Closing Database Connection ---")
    db.client.close()