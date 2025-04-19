import os


EMBEDDING_MODEL = "thenlper/gte-small"

QDRANT_COLLECTION_NAME = "polars-documentation"
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
CHAT_API_KEY = os.environ.get("CHAT_API_KEY", "")


def get_qdrant_config():
    from qdrant_client import models
    QDRANT_COLLECTION_CONFIG = {
        "collection_name": QDRANT_COLLECTION_NAME,
        "vectors_config": models.VectorParams(size=384, distance=models.Distance.COSINE),   # on_disk=True),
        # "hnsw_config": models.HnswConfigDiff(on_disk=True)
    }
    return QDRANT_COLLECTION_CONFIG
