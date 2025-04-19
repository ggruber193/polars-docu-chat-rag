from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from markdown import markdown
from pathlib import Path
from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter, TextSplitter

from src.utils import batched


def read_markdown_file(path: str | Path) -> [str, str]:
    path = Path(path)
    with open(path, 'r', encoding="utf8") as f_r:
        text = f_r.read()

    # text = markdown(text)
    # text = ''.join(BeautifulSoup(text).findAll(text=True))
    return text, str(path)


def split_markdown(md: str | list[str],
                   metadata=dict[str, Any] | list[dict[str, Any]],
                   chunk_size=512,
                   overlap=64,
                   splitter: TextSplitter = None) -> list[Document]:
    if isinstance(md, str):
        md = [md]
        if isinstance(metadata, list):
            raise ValueError("metadata should be a single dict")
        metadata = [metadata]
    if splitter is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md = [MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False).split_text(i) for i in md]
        metadata = [{**metadata[i], **text.metadata} for i, text_split in enumerate(md) for text in text_split]
        md = [j.page_content for i in md for j in i]
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    docs = splitter.create_documents(md, metadata)
    return docs


def process_markdown_files(paths: list[str | Path], batch_size=1, chunk_size=512, overlap=64):
    for files in batched(paths, batch_size):
        mds_w_paths = [read_markdown_file(i) for i in files]
        metadata = [{"path": md_path} for _, md_path in mds_w_paths]
        md = [md for md, _ in mds_w_paths]
        docs = split_markdown(md, metadata, chunk_size=chunk_size, overlap=overlap)
        yield [i.page_content for i in docs], [i.metadata for i in docs]


if __name__ == '__main__':
    docs_files = Path("D:\PycharmProjects\polargs-docu-chat-rag\data\polars-docu").rglob("*.md")
    for doc in process_markdown_files(docs_files):
        print(doc)
        break
