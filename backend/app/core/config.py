import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # Application
    app_name: str = "ILLUMINATOR_GPTv0.1"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
 
    # Directories
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    models_dir: Path = base_dir / "models"
    data_dir: Path = base_dir / "data"
    documents_dir: Path = data_dir / "documents"
    vectors_dir: Path = data_dir / "vectors"
    cache_dir: Path = data_dir / "cache"
    outputs_dir: Path = data_dir / "outputs"
 
    # LLM Settings (CHECK - Ashish)
    llm_model_path: Optional[str] = None
    llm_context_size: int = 4096
    llm_gpu_layers: int = -1  # -1 = auto, 0 = CPU only
    llm_threads: int = 4
 
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
 
    # ChromaDB Settings
    chroma_collection_name: str = "documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
 
    # TTS Settings (CHECK-Swayam)
    tts_male_voice: str = "en_US-lessac-medium"
    tts_female_voice: str = "en_US-amy-medium"
    tts_sample_rate: int = 22050
 
    # Image Generation Settings (CHECK-Swayam)
    image_model: str = "stable-diffusion-v1-5"
    image_steps: int = 20
    image_guidance_scale: float = 7.5
 
    # Translation Settings 
    translation_default_source: str = "en"
    translation_default_target: str = "hi"
 
    # Ollama Settings
    ollama_host: str = "http://localhost:11434"
 
    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        extra = "ignore"
 
    def ensure_directories(self):
        """Create all required directories."""
        for dir_path in [
            self.models_dir,
            self.models_dir / "llm",
            self.models_dir / "tts",
            self.models_dir / "embeddings",
            self.models_dir / "translation",
            self.models_dir / "image_gen",
            self.data_dir,
            self.documents_dir,
            self.vectors_dir,
            self.cache_dir,
            self.outputs_dir,
            self.outputs_dir / "podcasts",
            self.outputs_dir / "images",
            self.outputs_dir / "documents",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
 
 
# Global settings instance
settings = Settings()
settings.ensure_directories()