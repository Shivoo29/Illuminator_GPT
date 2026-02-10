"""
Query API - Endpoints for RAG queries.
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from app.services.rag_engine import RAGEngine
from app.services.llm_manager import GenerationConfig


router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str
    n_results: int = 5
    filter_document_id: Optional[str] = None
    stream: bool = False
    max_tokens: int = 2048
    temperature: float = 0.7


class SummarizeRequest(BaseModel):
    """Request model for document summarization."""
    document_id: str
    max_length: int = 500


class CompareRequest(BaseModel):
    """Request model for document comparison."""
    document_id_1: str
    document_id_2: str


@router.post("")
async def query(request: Request, body: QueryRequest):
    """
    Query documents using RAG.
    Returns answer with sources.
    """
    app_state = request.app.state.app_state

    if not app_state.llm_ready:
        raise HTTPException(status_code=503, detail="LLM not initialized")

    if not app_state.vectorstore_ready:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    # Build filter
    filter_metadata = None
    if body.filter_document_id:
        filter_metadata = {"document_id": body.filter_document_id}

    # Create RAG engine
    rag = RAGEngine(app_state.llm_manager, app_state.vector_store)

    # Configure generation
    config = GenerationConfig(
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )

    if body.stream:
        # Streaming response
        async def stream_response():
            async for chunk in rag.query_stream(
                body.question,
                n_results=body.n_results,
                filter_metadata=filter_metadata,
                generation_config=config,
            ):
                yield json.dumps({"chunk": chunk}) + "\n"

        return StreamingResponse(
            stream_response(),
            media_type="application/x-ndjson",
        )

    # Non-streaming response
    result = await rag.query(
        body.question,
        n_results=body.n_results,
        filter_metadata=filter_metadata,
        generation_config=config,
    )

    return {
        "answer": result.answer,
        "sources": result.sources,
        "query": result.query,
    }


@router.post("/with_citations")
async def query_with_citations(request: Request, body: QueryRequest):
    """
    Query with inline citations.
    Returns structured response with cited passages.
    """
    app_state = request.app.state.app_state

    if not app_state.llm_ready or not app_state.vectorstore_ready:
        raise HTTPException(status_code=503, detail="Services not initialized")

    rag = RAGEngine(app_state.llm_manager, app_state.vector_store)

    result = await rag.answer_with_citations(
        body.question,
        n_results=body.n_results,
    )

    return result


@router.post("/summarize")
async def summarize_document(request: Request, body: SummarizeRequest):
    """Generate a summary of a document."""
    app_state = request.app.state.app_state

    if not app_state.llm_ready or not app_state.vectorstore_ready:
        raise HTTPException(status_code=503, detail="Services not initialized")

    rag = RAGEngine(app_state.llm_manager, app_state.vector_store)

    summary = await rag.summarize_document(
        body.document_id,
        max_length=body.max_length,
    )

    return {
        "document_id": body.document_id,
        "summary": summary,
    }


@router.post("/compare")
async def compare_documents(request: Request, body: CompareRequest):
    """Compare two documents."""
    app_state = request.app.state.app_state

    if not app_state.llm_ready or not app_state.vectorstore_ready:
        raise HTTPException(status_code=503, detail="Services not initialized")

    rag = RAGEngine(app_state.llm_manager, app_state.vector_store)

    comparison = await rag.compare_documents(
        body.document_id_1,
        body.document_id_2,
    )

    return {
        "document_id_1": body.document_id_1,
        "document_id_2": body.document_id_2,
        "comparison": comparison,
    }


@router.post("/key_points")
async def extract_key_points(
    request: Request,
    document_id: str,
    num_points: int = 5,
):
    """Extract key points from a document."""
    app_state = request.app.state.app_state

    if not app_state.llm_ready or not app_state.vectorstore_ready:
        raise HTTPException(status_code=503, detail="Services not initialized")

    rag = RAGEngine(app_state.llm_manager, app_state.vector_store)

    points = await rag.extract_key_points(document_id, num_points)

    return {
        "document_id": document_id,
        "key_points": points,
    }


@router.post("/search")
async def search_documents(
    request: Request,
    query: str,
    n_results: int = 10,
    document_id: Optional[str] = None,
):
    """Search documents without generating an answer."""
    app_state = request.app.state.app_state

    if not app_state.vectorstore_ready:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    filter_metadata = {"document_id": document_id} if document_id else None

    results = await app_state.vector_store.search(
        query=query,
        n_results=n_results,
        filter_metadata=filter_metadata,
    )

    return {
        "query": query,
        "results": [
            {
                "id": r.id,
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in results
        ],
        "total": len(results),
    }