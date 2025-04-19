from pathlib import Path

from qdrant_client import QdrantClient
from tqdm import tqdm
import argparse

from src.database.qdrant_store import QdrantStore
from src.embeddings import TextEmbedder
from src.data_processing.process_markdown import process_markdown_files
from src.config import get_qdrant_config, EMBEDDING_MODEL, QDRANT_COLLECTION_NAME


def embedding_pipeline(qdrant_store: QdrantStore,
                       embedder: TextEmbedder, files: list[str | Path],
                       document_batch_size=4,
                       embedding_batch_size=128):
    with tqdm(total=len(files)) as pbar:
        for doc in process_markdown_files(files, batch_size=document_batch_size):
            text, metadata = doc
            embeddings = embedder.embed_text(text, batch_size=embedding_batch_size)
            payload = [{"text": i, **j} for i, j in zip(text, metadata)]
            qdrant_store.upsert_points(embeddings, payload, QDRANT_COLLECTION_NAME)

            pbar.update(document_batch_size)


if __name__ == "__main__":
    import os
    polars_dir = Path("../data/polars-docu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=polars_dir)
    parser.add_argument('--changed-files', type=Path, nargs="*", default=list(polars_dir.rglob("*")))
    parser.add_argument('--document-batch-size', type=int, default=4)
    parser.add_argument('--embedding-batch-size', type=int, default=128)
    parser.add_argument('--reset-collection', action="store_true", default=False)

    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    if not qdrant_url:
        raise(ValueError("Api Key or Url not found in environment"))

    args = parser.parse_args()
    changed_files: set[Path] = set(args.changed_files)
    data_dir: Path = args.directory
    document_batch_size = args.document_batch_size
    embedding_batch_size = args.embedding_batch_size
    files_present = [i for i in data_dir.rglob("*.md") if i in changed_files]

    reset_collection = args.reset_collection

    client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
    qdrant_store = QdrantStore(client, collection_config=get_qdrant_config())
    embedder = TextEmbedder(EMBEDDING_MODEL)

    if reset_collection:
        client.delete_collection(QDRANT_COLLECTION_NAME)
        qdrant_store.create_collection(get_qdrant_config())

    if changed_files:
        qdrant_store.delete_points_by_match(("path", [str(i) for i in changed_files]),
                                            collection_name=QDRANT_COLLECTION_NAME)
        embedding_pipeline(qdrant_store,
                           embedder,
                           files_present,
                           document_batch_size=document_batch_size,
                           embedding_batch_size=embedding_batch_size)
