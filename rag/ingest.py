"""
Ingestion pipeline — loads documents, semantic chunks, embeds, stores in ChromaDB.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH",   "./data/chroma_db")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME",  "factory_manuals")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")


def get_embeddings():
    """Load the local MiniLM embedding model (downloads once, ~80 MB)."""
    print("[Ingest] Loading embedding model:", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(folder_path: str) -> list[Document]:
    """Load all PDF, DOCX, and TXT files from a folder."""
    docs = []
    supported = {".pdf": PyMuPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}

    if not os.path.exists(folder_path):
        print(f"[Ingest] ERROR: Folder not found: {folder_path}")
        return docs

    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in supported]

    if not files:
        print(f"[Ingest] No supported files found in: {folder_path}")
        return docs

    for filename in files:
        ext      = os.path.splitext(filename)[1].lower()
        filepath = os.path.join(folder_path, filename)
        loader   = supported[ext](filepath)
        try:
            loaded = loader.load()
            # Tag every page with the source filename
            for doc in loaded:
                doc.metadata["source_file"] = filename
                doc.metadata["file_type"]   = ext.lstrip(".")
            docs.extend(loaded)
            print(f"[Ingest]   Loaded: {filename} ({len(loaded)} pages)")
        except Exception as e:
            print(f"[Ingest]   ERROR loading {filename}: {e}")

    print(f"[Ingest] Total pages loaded: {len(docs)}")
    return docs


def chunk_documents(documents: list[Document],
                    embeddings: HuggingFaceEmbeddings) -> list[Document]:
    """
    Semantically chunk documents.
    SemanticChunker splits on meaning boundaries — not fixed token windows.
    This preserves complete specifications within a single chunk.
    """
    print("[Ingest] Semantic chunking...")
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,   # Split when similarity drops below 85th percentile
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    print(f"[Ingest] Total chunks created: {len(chunks)}")

    # Show a sample chunk so you can verify quality
    if chunks:
        print("\n[Ingest] Sample chunk preview (chunk 0):")
        print("-" * 50)
        print(chunks[0].page_content[:300], "...")
        print("-" * 50)

    return chunks


def embed_and_store(chunks: list[Document],
                    embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed all chunks and store in ChromaDB.
    Uses upsert so re-running won't duplicate documents.
    """
    print(f"[Ingest] Storing {len(chunks)} chunks in ChromaDB...")
    print(f"[Ingest] DB path: {CHROMA_DB_PATH}")

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"[Ingest] ChromaDB now contains {count} chunks.")
    return vectorstore


def verify_store():
    """Quick sanity check — query the store with a test question."""
    print("\n[Ingest] Verifying vector store with test query...")
    embeddings  = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    results = vectorstore.similarity_search("torque specification M12 bolt", k=2)
    print(f"[Ingest] Test query returned {len(results)} chunks.")
    if results:
        print("[Ingest] Top result preview:")
        print(results[0].page_content[:200])
    print("[Ingest] Verification complete.")


def run_ingestion(folder_path: str = None):
    """Main entry point — run the full ingestion pipeline."""
    folder = folder_path or "./data/manuals"
    print("=" * 55)
    print("  MANUFACTURING SAFETY ORACLE — INGESTION PIPELINE")
    print("=" * 55)

    embeddings = get_embeddings()
    docs       = load_documents(folder)

    if not docs:
        print("[Ingest] Nothing to ingest. Add PDF/DOCX/TXT files to:", folder)
        return

    chunks = chunk_documents(docs, embeddings)
    embed_and_store(chunks, embeddings)
    verify_store()

    print("\n[Ingest] Pipeline complete!")
    print(f"[Ingest] {len(chunks)} chunks ready for retrieval.")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data/manuals"
    run_ingestion(folder)
