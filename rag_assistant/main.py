"""
main.py
-------
FastAPI backend for ResearchRAG: a fully local, free/open-source RAG
citation assistant.

Pipeline:
  Upload PDF -> extract + chunk (pdf_processor) -> embed + store (vector_store, FAISS)
  Query text -> embed -> FAISS retrieve -> flan-t5-base generate (generator) -> cited output

No paid APIs or external keys are used anywhere in this file.
"""

from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vector_store import get_vector_store
from pdf_processor import process_pdf
from generator import generate_related_work

app = FastAPI(title="ResearchRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class DeleteRequest(BaseModel):
    source: str


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
@app.post("/api/ingest")
async def ingest(title: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    chunks = process_pdf(pdf_bytes, source_title=title.strip() or file.filename)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. The PDF may be scanned/image-only.",
        )

    store = get_vector_store()
    added = store.add_chunks(chunks)

    return {
        "title": title.strip() or file.filename,
        "chunks_indexed": added,
        "total_chunks": store.count(),
    }


# ----------------------------------------------------------------------
# Library management
# ----------------------------------------------------------------------
@app.get("/api/library")
def library():
    store = get_vector_store()
    sources = store.list_sources()
    return {
        "papers": sources,
        "paper_count": len(sources),
        "total_chunks": store.count(),
    }


@app.delete("/api/library")
def remove_paper(req: DeleteRequest):
    store = get_vector_store()
    removed = store.remove_by_source(req.source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{req.source}'.")
    return {"removed": removed, "total_chunks": store.count()}


# ----------------------------------------------------------------------
# Query / generation
# ----------------------------------------------------------------------
@app.post("/api/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    store = get_vector_store()
    if store.count() == 0:
        raise HTTPException(status_code=422, detail="No papers indexed yet. Upload a PDF first.")

    retrieved = store.search(req.query, top_k=max(1, min(req.top_k, 20)))
    result = generate_related_work(req.query, retrieved)

    return {
        "text": result["text"],
        "sources": result["sources"],
        "pipeline": {
            "embedding_dim": store.EMBEDDING_DIM,
            "chunks_retrieved": len(retrieved),
            "papers_used": len({c["source"] for c in retrieved}),
        },
    }
