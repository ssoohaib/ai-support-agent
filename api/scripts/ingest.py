# scripts/ingest.py   →   uv run --env-file ../.env python -m scripts.ingest
from pathlib import Path
from langchain_core.documents import Document
from langchain_postgres import Column
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag import TABLE, engine, get_store

engine.init_vectorstore_table(
    table_name=TABLE,
    vector_size=1536,                      # must match the embedding model
    metadata_columns=[Column("source", "TEXT"), Column("audience", "TEXT")],
)
store = get_store()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
docs = []
for path in sorted(Path("../data/policies").glob("*.md")):
    audience = "internal" if path.stem.startswith("internal-") else "public"
    for chunk in splitter.split_text(path.read_text()):
        docs.append(Document(page_content=chunk,
                             metadata={"source": path.name, "audience": audience}))

store.add_documents(docs)
print(f"Indexed {len(docs)} chunks")