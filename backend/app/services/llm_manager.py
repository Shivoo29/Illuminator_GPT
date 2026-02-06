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
        config: GenerationConfig,
    ) -> str:
        """Generate text using Ollama."""
        response = await self.ollama_client.post(
            "/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repeat_penalty": config.repeat_penalty,
                },
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
        
    async def _generate_local(
        self,
        prompt:str,
        config:GenerationConfig,
    ) -> str:
        loop=asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.llm(
                prompt,
                max_tokens=config.max_tokens,
                temperature = config.temperature,
                top_p = config.top_p,
                top_K = config.top_k,
                repeat_penalty = config.repeat_penalty,
                stop=config.stop,
            )
        )
        return result["choices"][0]["text"]
    
    async def genetate_stream(
            self,
            prompt:str,
            config: Optional[GenerationConfig]= None,
    ) -> AsyncGenerator[str,None]:
        """Generate text with streaming output"""
        if not self._initialized:
            raise RuntimeError("lawden Bhoujyum") # LLM not initialized
        config = config or GenerationConfig()
        if self.use_ollama:
            async for chunk in self._stream_ollama(prompt, config):
                yield chunk
        else:
            async for chunk in self._stream_local(prompt, config):
                yield chunk

    async def _stream_ollama(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation using Ollama."""
        async with self.ollama_client.stream(
            "POST",
            "/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repeat_penalty": config.repeat_penalty,
                },
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break

    async def _stream_local(
            self,
            prompt: str,
            config: GenerationConfig,
    ) -> AsyncGenerator[str, None]:
        loop= asyncio.get_event_loop()
        def generate():
            for output in self.llm(
                prompt,
                max_tokens = config.max_tokens,
                temperature = config.temperature,
                top_p = config.top_p,
                top_k = config.top_k,
                repeat_penalty = config.repat_penalty,
                stop= config.stop,
                stream = True,
            ):
                yield output["choices"][0]["text"]
        gen = generate()
        while True:
            try:
                chunk = await loop.run_in_executor(None, lambda: next(gen))
                yield chunk
            except StopIteration:
                break

    async def generate_json(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Generate and parse JSON output."""
        config = config or GenerationConfig()
        config.stop = config.stop or []
        config.stop.extend(["```", "\n\n\n"])

        response = await self.generate(prompt, config)

        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                json_str = response[start:end].strip()
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            # Return raw response if JSON parsing fails
            return {"raw_response": response, "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self.model_name,
            "path": str(self.model_path) if self.model_path else None,
            "backend": "ollama" if self.use_ollama else "llama-cpp",
            "context_size": settings.llm_context_size,
            "initialized": self._initialized,
        }

    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if self.use_ollama:
            # Check if model exists in Ollama
            response = await self.ollama_client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if any(m["name"] == model_name for m in models):
                    self.model_name = model_name
                    return True
        else:
            # Look for GGUF file
            llm_dir = settings.models_dir / "llm"
            model_path = llm_dir / f"{model_name}.gguf"
            if model_path.exists():
                await self._load_model(model_path)
                return True

        return False

    async def cleanup(self):
        """Clean up resources."""
        if self.ollama_client:
            await self.ollama_client.aclose()
        if self.llm:
            del self.llm
            self.llm = None       