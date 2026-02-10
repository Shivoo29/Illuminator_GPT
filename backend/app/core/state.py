import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from app.core.config import settings

class AppState:
    # Centralized application state management
    # Handles initialization ang lifecycle of core components

    def __init__(self):
        self.llm_manager = None
        self.embedding_manager = None
        self.vector_store = None
        self.document_processor = None
        self.podcast_generator = None
        self.image_generator = None
        self.translator = None


        self.llm_ready = False
        self.vectorstore_ready = False
        self.setup_complete = False

        self._config: Dict[str, Any] = {}

    async def initialize(self):
        '''Initialize application components.'''
        # Check if startup has been completed
        self.setup_complete = self._check_setup_complete()

        if self.setup_complete:
            await self._initialize_components()

    def _check_setup_complete(self) -> bool:
        '''Check if first-time setuo has been completed'''
        setup_marker = settings.data_dir / ".setup_complete"
        return setup_marker.exists()

    async def _initialize_components(self):
        # Initialize all application components.
        try:
            from app.services.llm_manager import LLMManager
            from app.services.embedding_manager import EmbeddingManager
            from app.services.vector_store import VectorStoreManager
            from app.services.document_processor import DocumentProcessor


            # embedding manager
            self.embedding_manager = EmbeddingManager()
            await self.embedding_manager.initialize()

            # vector store
            self.vector_store = VectorStoreManager(self.embedding_manager)
            await self.vector_store.initialize()
            self.vectorstore_ready = True

            # document processor
            self.document_processor = DocumentProcessor(self.embedding_manager, self.vector_store)

            # llm manager
            self.llm_manager = LLMManager()
            if await self.llm_manager.initialize():
                self.llm_ready = True

            print("All components initialized sucessfully")
        
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise

    async def initialize_optional_components(self):
        '''initialize optional components (TTS, Image Gen, Translation)'''
        # loaded on-demand to save resources
        pass

    async def get_llm_manager(self):
        '''Get LLM manager, initializing if needed.'''
        if self.llm_manager is None:
            from app.services.llm_manager import LLMManager
            self.llm_manager.initialize()
        return self.llm_manager

    async def get_podcast_generator(self):
        # Get podcast generator, initializing if needed.
        if self.podcast_generator is None:
            from app.services.podcast_generator import PodcastGenerator
            self.podcast_generator = PodcastGenerator(await self.get_llm_manager)
        return self.podcast_generator
    
    async def get_translator(self):
        # Get translator, initializing if needed.
        if self.translator is None:
            from app.services.translator import OfflineTranslator
            self.translation = OfflineTranslator()
        return self.translator

    def mark_setup_complete(self):
        # Mark setup as complete.
        setup_marker = setting.data_dir / ".setup_complete"
        setup_marker.touch()
        self.setup_complete = True

    async def cleanup(self):
        # cleanup resources on shutdown.
        if self.llm_manager:
            await self.llm_manager.cleanup()
        if self.vector_store:
            await self.vector_store.cleanup()