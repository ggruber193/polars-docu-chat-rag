from bs4 import BeautifulSoup
from markdown import markdown
from langchain.text_splitter import MarkdownTextSplitter


path = "D:\PycharmProjects\polargs-docu-chat-rag\data\polars-docu\concepts\data-types-and-structures.md"

with open(path, 'r', encoding="utf8") as f_r:
    test_md = f_r.read()

html = markdown(test_md)
text = ''.join(BeautifulSoup(html).findAll(text=True))

print(text[:10])

splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=64)

docs = splitter.create_documents([text])
print(docs)
