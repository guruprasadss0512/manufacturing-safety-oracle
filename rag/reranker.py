"""
Cross-encoder re-ranker — scores candidate chunks and returns top-K.
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
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import CrossEncoder
from langchain.schema import Document

RERANKER_MODEL = os.getenv("RERANKER_MODEL",
                           "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K_RERANK   = int(os.getenv("TOP_K_RERANK", "3"))

# Cache the model at module level so it loads only once per session
_reranker_model = None


def get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model (downloads ~90 MB on first run).
    Cached after first load — no repeated disk reads.
    """
    global _reranker_model
    if _reranker_model is None:
        print(f"[Reranker] Loading cross-encoder: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512)
        print("[Reranker] Model loaded.")
    return _reranker_model


def rerank(question: str,
           candidate_docs: list[Document],
           top_k: int = None) -> list[Document]:
    """
    Score each candidate doc against the original question using the
    cross-encoder. Returns top_k docs sorted by relevance score (highest first).

    Why this beats cosine similarity alone:
    - Cosine similarity compares vectors independently
    - Cross-encoder reads question + chunk TOGETHER — much more accurate
    """
    k       = top_k or TOP_K_RERANK
    model   = get_reranker()

    if not candidate_docs:
        print("[Reranker] No candidates to re-rank.")
        return []

    # Build (question, chunk_text) pairs for the cross-encoder
    pairs  = [(question, doc.page_content) for doc in candidate_docs]
    scores = model.predict(pairs)

    # Zip scores with docs, sort descending, take top-k
    scored = sorted(zip(scores, candidate_docs),
                    key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in scored[:k]]

    print(f"[Reranker] Scored {len(candidate_docs)} chunks → kept top {k}.")
    for i, (score, doc) in enumerate(scored[:k]):
        src = doc.metadata.get("source_file", "unknown")
        print(f"[Reranker]   #{i+1} score={score:.4f} | {src}")

    return top_docs


def rerank_with_scores(question: str,
                       candidate_docs: list[Document],
                       top_k: int = None) -> list[tuple]:
    """
    Same as rerank() but returns (score, doc) tuples.
    Used by the chat engine to compute confidence badges.
    """
    k      = top_k or TOP_K_RERANK
    model  = get_reranker()

    if not candidate_docs:
        return []

    pairs  = [(question, doc.page_content) for doc in candidate_docs]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, candidate_docs),
                    key=lambda x: x[0], reverse=True)
    return scored[:k]


def score_to_confidence(score: float) -> str:
    """
    Convert a raw cross-encoder score to a human-readable confidence label.
    Cross-encoder scores are log-odds — not bounded 0-1.
    Thresholds calibrated for ms-marco-MiniLM on technical docs.
    """
    if score >= 5.0:
        return "High"
    elif score >= 1.0:
        return "Medium"
    else:
        return "Low"


if __name__ == "__main__":
    from rag.retriever import run_retriever_pipeline

    test_q = "What is the torque specification for M12 bolts on the CNC lathe?"
    print("=" * 55)
    print("  RE-RANKER TEST")
    print("=" * 55)

    print("\nStep 1 — Retrieving candidates...")
    candidates = run_retriever_pipeline(test_q)

    print("\nStep 2 — Re-ranking...")
    scored_docs = rerank_with_scores(test_q, candidates)

    print(f"\nTop {len(scored_docs)} chunks after re-ranking:")
    for i, (score, doc) in enumerate(scored_docs):
        src        = doc.metadata.get("source_file", "unknown")
        confidence = score_to_confidence(score)
        print(f"\n--- Rank #{i+1} | Score: {score:.4f} | "
              f"Confidence: {confidence} | Source: {src} ---")
        print(doc.page_content[:300], "...")
