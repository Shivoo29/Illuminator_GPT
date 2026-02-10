"""
RAG Engine - Retrieval Augmented Generation for document Q&A.
"""
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass

from app.services.llm_manager import LLMManager, GenerationConfig
from app.services.vector_store import VectorStoreManager, SearchResult


@dataclass
class RAGResult:
    """RAG query result."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]


class RAGEngine:
    """
    Retrieval Augmented Generation engine.
    Combines vector search with LLM generation for document Q&A.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        vector_store: VectorStoreManager,
    ):
        self.llm = llm_manager
        self.vector_store = vector_store

    async def query(
        self,
        question: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> RAGResult:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            n_results: Number of documents to retrieve
            filter_metadata: Optional metadata filter
            generation_config: LLM generation configuration
        
        Returns:
            RAGResult with answer and sources
        """
        # Retrieve relevant documents
        search_results = await self.vector_store.search(
            query=question,
            n_results=n_results,
            filter_metadata=filter_metadata,
        )

        # Build context from retrieved documents
        context = self._build_context(search_results)

        # Generate answer
        prompt = self._build_prompt(question, context)
        answer = await self.llm.generate(prompt, generation_config)

        # Format sources
        sources = [
            {
                "id": r.id,
                "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in search_results
        ]

        return RAGResult(
            query=question,
            answer=answer.strip(),
            sources=sources,
        )

    async def query_stream(
        self,
        question: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the RAG response."""
        # Retrieve relevant documents
        search_results = await self.vector_store.search(
            query=question,
            n_results=n_results,
            filter_metadata=filter_metadata,
        )

        # Build context and prompt
        context = self._build_context(search_results)
        prompt = self._build_prompt(question, context)

        # Stream answer
        async for chunk in self.llm.generate_stream(prompt, generation_config):
            yield chunk

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Build context string from search results."""
        if not search_results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            source_info = result.metadata.get("filename", f"Source {i}")
            context_parts.append(f"[{source_info}]\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the RAG prompt."""
        return f"""Answer the question based on the context provided. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    async def answer_with_citations(
        self,
        question: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Answer with inline citations."""
        search_results = await self.vector_store.search(
            query=question,
            n_results=n_results,
        )

        context = self._build_context(search_results)

        prompt = f"""Answer the question based on the context. Include citations in [1], [2] format referencing the sources.

Context:
{context}

Question: {question}

Provide a well-structured answer with citations:"""

        answer = await self.llm.generate(prompt)

        return {
            "answer": answer.strip(),
            "sources": [
                {
                    "index": i + 1,
                    "content": r.content[:300],
                    "filename": r.metadata.get("filename", "Unknown"),
                }
                for i, r in enumerate(search_results)
            ],
        }

    async def summarize_document(
        self,
        document_id: str,
        max_length: int = 500,
    ) -> str:
        """Generate a summary of a document."""
        # Get all chunks for this document
        results = await self.vector_store.search(
            query="summary overview main points",
            n_results=10,
            filter_metadata={"document_id": document_id},
        )

        if not results:
            return "Document not found."

        content = "\n\n".join([r.content for r in results])

        prompt = f"""Summarize the following document in about {max_length} words. Focus on the main points and key insights.

Document:
{content[:8000]}

Summary:"""

        summary = await self.llm.generate(
            prompt,
            GenerationConfig(max_tokens=max_length * 2),
        )

        return summary.strip()

    async def compare_documents(
        self,
        doc_id_1: str,
        doc_id_2: str,
    ) -> str:
        """Compare two documents."""
        # Get content from both documents
        results_1 = await self.vector_store.search(
            query="main content",
            n_results=5,
            filter_metadata={"document_id": doc_id_1},
        )

        results_2 = await self.vector_store.search(
            query="main content",
            n_results=5,
            filter_metadata={"document_id": doc_id_2},
        )

        content_1 = "\n".join([r.content for r in results_1])[:4000]
        content_2 = "\n".join([r.content for r in results_2])[:4000]

        prompt = f"""Compare these two documents. Identify similarities, differences, and key distinctions.

Document 1:
{content_1}

Document 2:
{content_2}

Comparison:"""

        comparison = await self.llm.generate(
            prompt,
            GenerationConfig(max_tokens=1000),
        )

        return comparison.strip()

    async def extract_key_points(
        self,
        document_id: str,
        num_points: int = 5,
    ) -> List[str]:
        """Extract key points from a document."""
        results = await self.vector_store.search(
            query="key points main ideas important",
            n_results=10,
            filter_metadata={"document_id": document_id},
        )

        if not results:
            return []

        content = "\n\n".join([r.content for r in results])

        prompt = f"""Extract exactly {num_points} key points from this document. Return each point on a new line, starting with a number.

Document:
{content[:8000]}

Key Points:"""

        response = await self.llm.generate(prompt)

        # Parse points from response
        points = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering and clean up
                point = line.lstrip("0123456789.-) ").strip()
                if point:
                    points.append(point)

        return points[:num_points]