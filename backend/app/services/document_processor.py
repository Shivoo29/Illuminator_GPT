"""
Document Processor - Handles processing of various document types.
Supports PDF, DOCX, PPTX, images, audio, and video.
"""
import asyncio
import mimetypes
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import uuid
import shutil

from app.core.config import settings
from app.services.embedding_manager import EmbeddingManager
from app.services.vector_store import VectorStoreManager, Document


@dataclass
class ProcessedDocument:
    # processed document with extracted content and metadata.
    id: str
    filename: str
    file_type: str
    content: str
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count:int = 0
    word_count:int = 0


class DocumentProcessor:
    #Processes various file type
    def __init__(
            self,
            embedding_manager: EmbeddingManager,
            vector_store: VectorStoreManager,
        ):
            self.embedding_manager = embedding_manager
            self.vector_store = vector_store
            self.documents_dir = settings.documents_dir

    async def process_file(
              self,
              file_path: Path,
              original_filename: str,
              ) -> ProcessedDocument:
                # Process a file and store in vector store.
                # Returns processed document with metadata.
                file_type =  self._get_file_type(file_path, original_filename)

                # Extract content based on file type
                if file_type == "pdf":
                       content, metadata = await self._process_pdf(file_path)
                elif file_type == "docx":
                       content, metadata = await self._process_docx(file_path)
                elif file_type == "pptx":
                       content, metadata = await self._process_pptx(file_path)
                elif file_type in ["txt", "md", "json", "csv"]:
                       content, metadata = await self._process_txt(file_path)
                elif file_type in ["png", "jpg", "jpeg", "gif", "webp"]:
                       content, metadata = await self._process_txt(file_path)               
                elif file_type in ["mp3", "wav", "m4a", "flac", "ogg"]:
                       content, metadata = await self._process_txt(file_path)
                elif file_type in ["mp4", "avi", "mov", "mkv", "webm"]:
                       content, metadata = await self._process_txt(file_path)
                else:
                       raise ValueError(f"Unsupported file type:{file_path}")