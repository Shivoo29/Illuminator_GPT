"""
Vector Store Manager - ChromaDB-based vector storage for RAG.
Fully local, uses SQLite backend.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.services.embedding_manager import EmbeddingManager


@dataclass
class Document:
    """Document dataclass for vector storage."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result dataclass."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStoreManager:
    """
    Manages vector storage using ChromaDB.
    Uses local SQLite backend for fully offline operation.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.client = None
        self.collection = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            loop = asyncio.get_event_loop()

            def setup():
                # Create persistent client
                client = chromadb.PersistentClient(
                    path=str(settings.vectors_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )

                # Get or create collection
                collection = client.get_or_create_collection(
                    name=settings.chroma_collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

                return client, collection

            self.client, self.collection = await loop.run_in_executor(None, setup)
            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            return False

    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        if not self._initialized:
            await self.initialize()

        if not documents:
            return True

        # Generate embeddings for documents without them
        texts_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = await self.embedding_manager.embed(texts_to_embed)
            for i, embedding in zip(indices_to_embed, embeddings):
                documents[i].embedding = embedding

        loop = asyncio.get_event_loop()

        def add():
            self.collection.add(
                ids=[doc.id for doc in documents],
                embeddings=[doc.embedding for doc in documents],
                documents=[doc.content for doc in documents],
                metadatas=[doc.metadata for doc in documents],
            )

        await loop.run_in_executor(None, add)
        return True

    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self.embedding_manager.embed_single(query)

        loop = asyncio.get_event_loop()

        def do_search():
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )
            return results

        results = await loop.run_in_executor(None, do_search)

        # Convert to SearchResult objects
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score (1 - distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=score,
                ))

        return search_results

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()

        def get():
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"],
            )
            return results

        results = await loop.run_in_executor(None, get)

        if results and results["ids"]:
            return Document(
                id=results["ids"][0],
                content=results["documents"][0] if results["documents"] else "",
                metadata=results["metadatas"][0] if results["metadatas"] else {},
                embedding=results["embeddings"][0] if results["embeddings"] else None,
            )

        return None

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()

        def delete():
            self.collection.delete(ids=[doc_id])

        await loop.run_in_executor(None, delete)
        return True

    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()

        def delete():
            # Get matching documents first
            results = self.collection.get(where=metadata_filter)
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                return len(results["ids"])
            return 0

        return await loop.run_in_executor(None, delete)

    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()

        def clear():
            # Delete and recreate collection
            self.client.delete_collection(settings.chroma_collection_name)
            self.collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        await loop.run_in_executor(None, clear)
        return True

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()

        def get_stats():
            return {
                "name": settings.chroma_collection_name,
                "count": self.collection.count(),
            }

        return await loop.run_in_executor(None, get_stats)

    async def cleanup(self):
        """Cleanup resources."""
        pass  # ChromaDB handles cleanup automatically