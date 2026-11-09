# app/rag.py
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from app.config import settings

TABLE = "policy_chunks"
engine = PGEngine.from_connection_string(url=settings.database_url)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   # 1,536 numbers per chunk

def get_store() -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE,
        embedding_service=embeddings,
        metadata_columns=["source", "audience"],   # real columns → fast filters
    )

def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)