# ResearchRAG

A fully local RAG research-citation assistant. Upload PDFs, ask about a
research topic, get a "Related Work" paragraph with citations back to the
exact papers and chunks used. No paid APIs, no API keys.

## Stack
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local)
- **Vector store:** FAISS (`IndexFlatIP`, cosine similarity), persisted to
  disk as `.faiss` + `.json` (metadata) + `.npy` (raw vectors, for fast
  per-paper deletion without re-embedding)
- **Generation:** `google/flan-t5-base` (local, via `transformers`)
- **Backend:** FastAPI
- **Frontend:** plain HTML/CSS/JS (no build step, no framework)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first run will download `all-MiniLM-L6-v2` (~90MB) and `flan-t5-base`
(~950MB) from Hugging Face — this needs an internet connection once. After
that everything runs fully offline.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the frontend is served automatically.

## How it works

1. **Ingest** (`POST /api/ingest`): PDF → `pdf_processor.py` extracts text
   per page → splits into ~180-word overlapping chunks → embedded and added
   to the FAISS index via `vector_store.py`.
2. **Library** (`GET /api/library`, `DELETE /api/library`): lists indexed
   papers with chunk counts; deleting a paper rebuilds the FAISS index from
   the retained vectors (no re-embedding needed).
3. **Query** (`POST /api/query`): your research topic is embedded, FAISS
   returns the top-K most similar chunks, then `generator.py` asks
   flan-t5-base to summarize each source's most relevant chunk in one
   sentence, and stitches them into a paragraph with `[Source Title]`
   citation tags plus a `sources` list (title, page, excerpt) for display.

## File structure

```
main.py            FastAPI app + routes
vector_store.py     FAISS index + metadata persistence
pdf_processor.py     PDF extraction + chunking
generator.py         flan-t5-base grounded generation
requirements.txt
static/index.html    Frontend (matches the ResearchRAG UI)
vector_data/          created at runtime — your FAISS index + metadata live here
```

## Notes / things to tune

- **Chunk size** (`DEFAULT_CHUNK_WORDS` in `pdf_processor.py`, default 180
  words with 30-word overlap) — smaller chunks give more precise citations,
  larger chunks give more context per citation.
- **flan-t5-base** is small and fast on CPU but not a strong long-form
  writer. If you have a GPU and want noticeably better prose, swap
  `MODEL_NAME` in `generator.py` for `google/flan-t5-large` or
  `google/flan-t5-xl` — same code, just a bigger download.
- Scanned/image-only PDFs won't extract text (no OCR is included). Add
  `pytesseract` + `pdf2image` if you need OCR support.
