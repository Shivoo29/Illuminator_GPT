'''
Handles local LLM inference using llama-cpp-python and ollama
Docstring for backend.app.services.llm_manager
'''
import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List
from dataclasses import dataclass
 
import httpx
 
from app.core.config import settings

@dataclass
class GenerationConfig:
    """
    Docstring for GenerationConfig
    """
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop:Optional[List[str]]=None

class LLMManager:
    def __init__(self):
        self.llm = None
        self.model_path: Optional[Path] = None
        self.model_name: Optional[str] = None
        self.use_ollama: bool = False
        self.ollama_client: Optional[httpx.AsyncClient] = None
        self._initialized = False
    async def initialized(self) -> bool:
        if await self._check_ollama():
            self.use_ollama = True
            self.ollama_client = httpx.AsyncClient(
                base_url = settings.ollama_host,
                timeout = httpx.Timeout(300.0),
            )
            self._initialized = True
            return True
        
        # Fall back to llama-cpp-python
        model_path = self._find_model()
        if model_path:
            await self._load_model(model_path)
            self._initialized = True
            return True
        
        return False
    
    async def _check_ollama(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout = 5.0) as client:
                response = await client.get(f"{settings.ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    if models:
                        self.model_name = models[0]["name"]
                        return True
        except:
            pass
        return False
    
    def _find_model(self) -> Optional[Path]:
        # Find a GGUF model in the models directory.
        llm_dir = settings.models_dir / "llm"

        if settings.llm_model_path:
            path = Path(settings.llm_model_path)
            if path.exists():
                return path
            
        if llm_dir.exists():
            gguf_files = list(llm_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]
        return None
    
    async def _load_model(self, model_path: Path):
        try:
            from llama_cpp import Llama

            # check for GPU_layers
            n_gpu_layers = settings.llm_gpu_layers
            if n_gpu_layers == -1:
                n_gpu_layers = 35 if self._has_gpu() else 0

            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=settings.llm_context_size,
                n_threads=settings.llm_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self.model_path = model_path
            self.model_name = model_path.stem

        except Exception as e:
            raise RecursionError(f"Failed to load model:{e}")
        
    def _has_gpu(self) -> bool:
        # check for gpu, whether it is availabel or not?
        try:
            import subprocess
            results = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5,
            )
            return results.returncode == 0
        except:
            return False
    
    async def generate(
            self,
            prompt: str,
            config: Optional[GenerationConfig] = None,
    ) -> str:
        if not self._initialized:
            raise RuntimeError("LLM not initialized")
        
        config = config or GenerationConfig()

        if self.use_ollama:
            return await self._generate_ollama(prompt, config)
        else:
            return await self._generate_locale(prompt, config)
        
        async def _generate_ollama(
                self,
                prompt: str,
                config:GenerationConfig,
        ) -> str:
            # text generation using Ollama.
            response = await