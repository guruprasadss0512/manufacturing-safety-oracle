"""
Multi-Query retriever — generates query variants and fetches candidate chunks.
"""
import os

# ── Suppress ChromaDB telemetry warnings ──────────────────
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"
try:
    import chromadb.telemetry.product.posthog as _posthog
    _posthog.Posthog.capture = lambda self, *a, **kw: None
except Exception:
    pass
# ──────────────────────────────────────────────────────────
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH",  "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "factory_manuals")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL       = os.getenv("LLM_MODEL",       "llama-3.1-8b-instant")
TOP_K_RETRIEVE  = int(os.getenv("TOP_K_RETRIEVE", "12"))


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local MiniLM embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load the existing ChromaDB vector store from disk."""
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def get_llm() -> ChatGroq:
    """Load the Groq LLM for query generation."""
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def get_multiquery_retriever(vectorstore: Chroma, llm: ChatGroq) -> MultiQueryRetriever:
    """
    Build a MultiQueryRetriever.
    It uses the LLM to generate 3 alternative phrasings of the user question,
    runs all 3 against ChromaDB, then deduplicates results.
    """
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},   # 4 results per query variant = up to 12 total
    )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )


def retrieve_candidates(question: str,
                        retriever: MultiQueryRetriever) -> list:
    """
    Run multi-query retrieval.
    Returns deduplicated list of candidate Document objects.
    """
    print(f"[Retriever] Running multi-query retrieval for: '{question}'")
    docs = retriever.invoke(question)
    print(f"[Retriever] Retrieved {len(docs)} unique candidate chunks.")
    return docs


def run_retriever_pipeline(question: str) -> list:
    """
    Convenience function — loads all components and runs retrieval end-to-end.
    Used by the chat engine.
    """
    embeddings  = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm         = get_llm()
    retriever   = get_multiquery_retriever(vectorstore, llm)
    return retrieve_candidates(question, retriever)


if __name__ == "__main__":
    test_q = "What is the torque specification for M12 bolts on the CNC lathe?"
    print("=" * 55)
    print("  RETRIEVER TEST")
    print("=" * 55)
    docs = run_retriever_pipeline(test_q)
    print(f"\nTop {len(docs)} candidate chunks retrieved:")
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source_file", "unknown")
        pg  = doc.metadata.get("page", "?")
        print(f"\n--- Chunk {i+1} | Source: {src} | Page: {pg} ---")
        print(doc.page_content[:200], "...")
