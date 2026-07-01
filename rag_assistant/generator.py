"""
generator.py
------------
Generates a clear, direct answer to the user's question from retrieved
chunks using google/flan-t5-large, run entirely locally.

Strategy:
1. From each retrieved chunk, extract the most relevant sentences using
   keyword overlap with the query (so the best content fills the model's
   context window).
2. Combine the top sentences into a tight context block, numbered by source.
3. Prompt flan-t5-large with an explicit, direct Q&A instruction.
4. Return the full chunk text for every source card in the UI.
"""

import re
from typing import List, Dict, Any

from transformers import pipeline

# flan-t5-base (~250M params) is a good balance of speed and quality for CPU.
MODEL_NAME = "google/flan-t5-base"

_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            # Keep model on CPU; works without a GPU
            device=-1,
        )
    return _generator


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _tokenish_words(text: str) -> set:
    """Lower-cased alpha tokens — used for keyword relevance scoring."""
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _score_sentence(sentence: str, query_words: set) -> int:
    """Count how many query keywords appear in the sentence."""
    return len(query_words & _tokenish_words(sentence))


def _extract_best_sentences(text: str, query_words: set, max_chars: int = 600) -> str:
    """
    Split `text` into sentences, rank them by keyword overlap with the query,
    and return the top sentences up to `max_chars` characters (in their
    original order so the text flows naturally).
    """
    # Simple sentence split on ". ", "! ", "? " boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text[:max_chars]

    scored = sorted(
        enumerate(sentences),
        key=lambda t: _score_sentence(t[1], query_words),
        reverse=True,
    )

    # Pick top sentences by score, then sort back to original order
    selected_indices = set()
    total_chars = 0
    for idx, sent in scored:
        if total_chars + len(sent) > max_chars:
            break
        selected_indices.add(idx)
        total_chars += len(sent) + 1

    # If nothing was selected (very short text), take everything up to limit
    if not selected_indices:
        return text[:max_chars]

    ordered = [sentences[i] for i in sorted(selected_indices)]
    return " ".join(ordered)


def _build_context(query: str, chunks_by_source: Dict[str, List[str]],
                   ordered_sources: List[str]) -> str:
    """
    Build a numbered context block where each source contributes its most
    query-relevant sentences. Total context is kept under ~1 800 chars so it
    fits comfortably within flan-t5-large's 512-token encoder.
    """
    query_words = _tokenish_words(query)
    n = max(1, len(ordered_sources))
    per_source_budget = 1800 // n  # distribute budget evenly

    parts = []
    for i, src in enumerate(ordered_sources, start=1):
        full_text = " ".join(chunks_by_source[src])
        best = _extract_best_sentences(full_text, query_words,
                                       max_chars=per_source_budget)
        parts.append(f"[Source {i} – {src}]:\n{best}")

    return "\n\n".join(parts)


# -----------------------------------------------------------------------
# Main generation entry-point
# -----------------------------------------------------------------------

def generate_related_work(query: str,
                           retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    retrieved: list of chunk dicts from VectorStore.search() with keys
               "text", "source", "page", "score".

    Returns:
        {
          "text":    "<clear, direct answer with [Source N] citations>",
          "sources": [{"index", "source", "page", "excerpt"}, ...]
        }
    """
    if not retrieved:
        return {
            "text": (
                "No relevant content was found. Please upload a PDF first, "
                "then ask your question."
            ),
            "sources": [],
        }

    # Collect all chunks grouped by source (paper)
    all_chunks_by_source: Dict[str, List[str]] = {}
    best_per_source: Dict[str, Dict[str, Any]] = {}

    for chunk in retrieved:
        src = chunk["source"]
        all_chunks_by_source.setdefault(src, []).append(chunk["text"])
        if src not in best_per_source or chunk["score"] > best_per_source[src]["score"]:
            best_per_source[src] = chunk

    ordered_sources = sorted(
        best_per_source.keys(),
        key=lambda s: -best_per_source[s]["score"],
    )

    # Build the focused context
    context = _build_context(query, all_chunks_by_source, ordered_sources)

    # Craft an explicit, unambiguous Q&A prompt for flan-t5-large
    prompt = (
        "You are a research assistant. Use ONLY the context below to answer "
        "the question. Give a clear, complete, and specific answer. "
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    gen = _get_generator()
    raw = gen(
        prompt,
        max_new_tokens=300,   # enough for a full paragraph
        do_sample=False,      # deterministic (greedy), more factual
        num_beams=4,          # beam search for more coherent output
    )[0]["generated_text"].strip()

    # Append clear inline citations
    citations = " | ".join(
        f"[{i}] {s}" for i, s in enumerate(ordered_sources, 1)
    )
    full_answer = f"{raw}\n\n─── Sources ───\n{citations}"

    # Sources list — expose the complete retrieved text for the UI
    sources_out = []
    for i, src in enumerate(ordered_sources, start=1):
        full_text = " ".join(all_chunks_by_source[src])
        sources_out.append({
            "index": i,
            "source": src,
            "page": best_per_source[src].get("page"),
            "excerpt": full_text,
        })

    return {"text": full_answer, "sources": sources_out}
