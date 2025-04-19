import os


EMBEDDING_MODEL = "thenlper/gte-small"

QDRANT_COLLECTION_NAME = "polars-documentation"
url = "https://4b15c9c8-a0b5-4e8e-8315-6292867ca0e0.europe-west3-0.gcp.cloud.qdrant.io"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOlt7ImNvbGxlY3Rpb24iOiJwb2xhcnMtZG9jdW1lbnRhdGlvbiIsImFjY2VzcyI6InJ3In1dfQ.H_YSNobSMvK2bXufOQsLYNa71XsDmFR1zWZHKkCKbh4"
QDRANT_URL = os.environ.get("QDRANT_URL", url)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", api_key)
CHAT_API_KEY = os.environ.get("CHAT_API_KEY", "")


def get_qdrant_config():
    from qdrant_client import models
    QDRANT_COLLECTION_CONFIG = {
        "collection_name": QDRANT_COLLECTION_NAME,
        "vectors_config": models.VectorParams(size=384, distance=models.Distance.COSINE),   # on_disk=True),
        # "hnsw_config": models.HnswConfigDiff(on_disk=True)
    }
    return QDRANT_COLLECTION_CONFIG
