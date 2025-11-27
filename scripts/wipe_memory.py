import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from src.config import VECTOR_DB_PATH, COLLECTION_NAME


def wipe_brain():
    print(f"Wiping vector DB at: {VECTOR_DB_PATH}")

    if not Path(VECTOR_DB_PATH).exists():
        print("Database path does not exist. Nothing to wipe.")
        return

    client = QdrantClient(path=str(VECTOR_DB_PATH))

    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        print(f"Deleting collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        print("Collection deleted")
    else:
        print(f"Collection not found: {COLLECTION_NAME}")

    client.close()


if __name__ == "__main__":
    wipe_brain()
