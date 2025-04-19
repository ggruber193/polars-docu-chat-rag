from typing import Any

from qdrant_client import QdrantClient, models
from uuid import uuid4
from transformers import PreTrainedModel

from src.config import QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
from src.embeddings import TextEmbedder


class QdrantStore:
    def __init__(self, client: QdrantClient, collection_config=None):
        self.client = client
        self.collection_names = set([i.name for i in client.get_collections().collections])

        if collection_config is not None:
            self.create_collection(collection_config)

    def create_collection(self, collection_config: dict):
        collection_name = collection_config["collection_name"]
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(**collection_config)
            self.collection_names.add(collection_name)

    def _check_collection_name(self, collection_name):
        if collection_name not in self.collection_names:
            raise ValueError(f"Collection: {collection_name} does not exist.")

    def upsert_points(self,
                      vectors: Any | list[Any],
                      payloads: dict | list[dict],
                      collection_name: str):
        self._check_collection_name(collection_name)

        ids = [str(uuid4()) for _ in payloads]

        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                payloads=payloads,
                vectors=vectors
            )
        )

    def delete_points(self,
                      filters: dict[str, list[models.FieldCondition]],
                      collection_name: str):
        self._check_collection_name(collection_name)

        self.client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(**filters)
        )

    def delete_points_by_match(self,
                               key_value: tuple[str, list[str] | str],
                               collection_name: str):
        key, values = key_value
        if isinstance(values, str):
            values = [values]
        filter = {"must": [models.FieldCondition(key=key, match=models.MatchAny(any=values))]}
        self.delete_points(filter, collection_name)

    def get_topk_points_single(self,
                               query: Any | str,
                               collection_name: str,
                               k=5):
        responses = self.client.query_points(collection_name=collection_name,
                                             query=query,
                                             limit=k)

        return [i.payload["text"] for i in responses.points]


if __name__ == '__main__':
    client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    qdrant_store = QdrantStore(client)
    embedding_model = TextEmbedder(modelname=EMBEDDING_MODEL)
    query = "How to filter a dataframe"
    query_emb = embedding_model.embed_text(query)
    responses = qdrant_store.get_topk_points_single(query_emb[0], collection_name=QDRANT_COLLECTION_NAME)
    print(responses)
