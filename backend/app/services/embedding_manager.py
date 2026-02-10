
"""
Embedding Manager - Handles text embeddings using sentence-transformers.
Fully offline after initial model download.
"""
import asyncio
from typing import List, Optional
from pathlib import Path

from app.core.config import settings


class EmbeddingManager:
    """
    Manages text embeddings using sentence-transformers.
    Uses all-MiniLM-L6-v2 by default (80MB, fast, good quality).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.device = settings.embedding_device
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            loop = asyncio.get_event_loop()

            def load_model():
                return SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )

            self.model = await loop.run_in_executor(None, load_model)
            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
            return False

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        loop = asyncio.get_event_loop()

        def encode():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        return await loop.run_in_executor(None, encode)

    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2

    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "name": self.model_name,
            "device": self.device,
            "dimension": self.get_embedding_dimension(),
            "initialized": self._initialized,
        }
