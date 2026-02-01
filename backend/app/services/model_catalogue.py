'''
Docstring for backend.app.services.model_catalogue
Manages available model wirh online/offline support.
Fetches latest models from HuggingFace when online, uses local cache when offline
'''

import json
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta

from app.core.config import settings


@dataclass
class ModelInfo:
    # Model Info...
    name:str
    display_name:str
    size_gb:float
    description:str
    recommended:bool = False
    speed:str = "Fast"
    quality:str = "High"
    requirements:str = "8GB RAM"
    ollama_name:Optional[str] = None
    huggingface_url:Optional[str] = None
    category:str = "general" # general ka matlab upper cast(good in everythin coding, multilingual, etc).
    context_length: int = 4096
    parameters:str = "7B"
    quantization:str = "Q4_K_M"
    is_local: bool = False # true, when model is downloaded

    def to_dict(self)->Dict[str,Any]:
        return asdict(self)
    
# Default offline models catalog (bundled with app)
DEFAULT_MODELS: List[ModelInfo] = [
    ModelInfo(
        name="llama3.2-7b",
        display_name="Llama 3.2 7B Instruct",
        size_gb=4.1,
        description="High-Quality general-purpose model",
        recommended=True,
        speed="Fast (~20 tok/sec)",
        quality="High",
        requirements="8GB RAM, 5GB DISK",
        ollama_name="llama3.2:7b-instruct-q4_k_M",
        huggingface_url="https://huggingface.co/lmstudio-community/Llama-3.2-7B-Instruct-GGUF/resolve/main/Llama-3.2-7B-Instruct-Q4_K_M.gguf",
        category="general",
        context_length=8192,
        parameters="7B",
    ),
    ModelInfo(
        name="llama3.2-3b"
        display_name=
        size_gb=
        description=
        recommended=
        speed=
        quality=
        requirements=
        ollama_name=
        huggingface_url=
        category=
        context_length=
        parameters=
    ),
    ModelInfo(
        name="mistral-7b",
        display_name=
        size_gb=
        description=
        recommended=
        speed=
        quality=
        requirements=
        ollama_name=
        huggingface_url=
        category=
        context_length=
        parameters=
    ),
    ModelInfo(
        name="codellama-7b",
        display_name=
        size_gb=
        description=
        recommended=
        speed=
        quality=
        requirements=
        ollama_name=
        huggingface_url=
        category=
        context_length=
        parameters=
    ),
    ModelInfo(
        name="gemma2-9b",
        display_name=
        size_gb=
        description=
        recommended=
        speed=
        quality=
        requirements=
        ollama_name=
        huggingface_url=
        category=
        context_length=
        parameters=
    ),
]

# Online model sources for fetching latest models
ONLINE_MODEL_SOURCES = [
    {
        "name": "HuggingFace GGUF Models",
        "api_url": "https://huggingface.co/api/models",
        "params": {"filter": "gguf", "sort": "downloads", "direction": -1, "limit": 50},
    },
]

class ModelCatalogue:
    # Manages Model Discovery And Availability.
    def __init__(self):
        self.cache_file = settings.cache_dir / "model_catalogue_cache.json"
        self.cache_duration = timedelta(hours=24) # Caches online results for 24 hours
        self._session: Optional[aiohttp.ClientSession]=None
        self._is_online: Optional[bool]=None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(total=10))
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        
    async def check_online_status(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get("https://huggingface.co/api/models?limit=1", timeout = 5) as response:
                self._is_online = response.status == 200
                return self._is_online
        except Exception:
            self._is_online = False
            return False
    
    def get_local_models(self) -> List[ModelInfo]:
        local_models = []
        llm_dir = settings.models_dir / "llm"

        if llm_dir.exists():
            for file in llm_dir.glob("*.gguf"):
                size_gb = file.stat().st_size/(1024 ** 3)

                # try to match with known models
                matched_model=None
                for model in DEFAULT_MODELS:
                    if model.huggingface_url and file.name in model.huggingface_url:
                        matched_model = model
                        break

                if matched_model:
                    local_models = ModelInfo(
                        **{**asdict(matched_model), "is_local": True}
                    )
                    local_models.append(local_models)
                else:
                    local_models.append(ModelInfo(
                        name=file.stem,
                        display_name=file.stem.replace("-"," ").replace("-"," ").title(),
                        size_gb=round(size_gb,2)
                        description="Locally imported model",
                        is_local=True,
                        huggingface_url=None,
                        ollama_name=None,
                    ))
        return local_models
    

    def get_ollama_models(self) -> List[Dict[str, Any]]:
        """Get models available in Ollama."""
        import subprocess
 
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
 
            if result.returncode == 0:
                models = []
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
 
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        models.append({
                            "name": parts[0],
                            "size": parts[1] if len(parts) > 1 else "unknown",
                            "is_local": True,
                        })
 
                return models
        except Exception:
            pass
 
        return []
 
    async def fetch_online_models(self) -> List[ModelInfo]:
        """Fetch latest models from online sources."""
        if not await self.check_online_status():
            return []
 
        online_models = []
 
        try:
            session = await self._get_session()
 
            # Fetch from HuggingFace
            params = {
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": "30",
            }
 
            async with session.get(
                "https://huggingface.co/api/models",
                params=params,
            ) as response:
                if response.status == 200:
                    data = await response.json()
 
                    for item in data:
                        model_id = item.get("id", "")
 
                        # Skip if already in default models
                        if any(model_id in (m.huggingface_url or "") for m in DEFAULT_MODELS):
                            continue
 
                        # Parse model info
                        downloads = item.get("downloads", 0)
                        likes = item.get("likes", 0)
 
                        # Estimate size from model name
                        size_gb = self._estimate_model_size(model_id)
 
                        online_models.append(ModelInfo(
                            name=model_id.replace("/", "-"),
                            display_name=model_id.split("/")[-1].replace("-", " ").title(),
                            size_gb=size_gb,
                            description=f"Downloaded {downloads:,} times, {likes} likes",
                            recommended=False,
                            speed="Varies",
                            quality="Varies",
                            requirements="Check model page",
                            huggingface_url=f"https://huggingface.co/{model_id}",
                            category="community",
                            is_local=False,
                        ))
 
            # Cache the results
            await self._save_cache(online_models)
 
        except Exception as e:
            print(f"Failed to fetch online models: {e}")
 
        return online_models
 
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size from name."""
        model_lower = model_name.lower()
 
        if "70b" in model_lower:
            return 40.0
        elif "34b" in model_lower or "33b" in model_lower:
            return 20.0
        elif "13b" in model_lower:
            return 8.0
        elif "7b" in model_lower or "8b" in model_lower:
            return 4.0
        elif "3b" in model_lower:
            return 2.0
        elif "1b" in model_lower:
            return 1.0
 
        return 4.0  # Default estimate
 
    async def _save_cache(self, models: List[ModelInfo]):
        """Save models to cache file."""
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "models": [m.to_dict() for m in models],
        }
 
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(cache_data, indent=2))
 
    def _load_cache(self) -> Optional[List[ModelInfo]]:
        """Load models from cache file."""
        if not self.cache_file.exists():
            return None
 
        try:
            data = json.loads(self.cache_file.read_text())
 
            # Check if cache is still valid
            timestamp = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - timestamp > self.cache_duration:
                return None
 
            return [ModelInfo(**m) for m in data["models"]]
        except Exception:
            return None
 
    async def get_available_models(
        self,
        include_online: bool = True,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get all available models.
 
        Returns:
            Dict with:
            - is_online: Whether we have internet connectivity
            - local_models: Models already downloaded
            - default_models: Bundled recommended models
            - online_models: Additional models from HuggingFace (if online)
            - ollama_models: Models available in Ollama
        """
        result = {
            "is_online": False,
            "local_models": [],
            "default_models": [],
            "online_models": [],
            "ollama_models": [],
        }
 
        # Get local models
        result["local_models"] = [m.to_dict() for m in self.get_local_models()]
 
        # Get Ollama models
        result["ollama_models"] = self.get_ollama_models()
 
        # Get default models (mark as local if downloaded)
        local_names = {m["name"] for m in result["local_models"]}
        default_models = []
        for model in DEFAULT_MODELS:
            model_dict = model.to_dict()
            model_dict["is_local"] = model.name in local_names
            default_models.append(model_dict)
        result["default_models"] = default_models
 
        # Try to get online models
        if include_online:
            is_online = await self.check_online_status()
            result["is_online"] = is_online
 
            if is_online:
                # Try cache first
                cached = self._load_cache()
                if cached:
                    result["online_models"] = [m.to_dict() for m in cached]
                else:
                    online = await self.fetch_online_models()
                    result["online_models"] = [m.to_dict() for m in online]
 
        # Filter by category if specified
        if category:
            result["default_models"] = [
                m for m in result["default_models"]
                if m.get("category") == category
            ]
            result["online_models"] = [
                m for m in result["online_models"]
                if m.get("category") == category
            ]
 
        return result
 
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get a specific model by name."""
        # Check default models
        for model in DEFAULT_MODELS:
            if model.name == name:
                return model
 
        # Check local models
        for model in self.get_local_models():
            if model.name == name:
                return model
 
        return None
 
    def get_recommended_model(self) -> ModelInfo:
        """Get the recommended model for this system."""
        import psutil
 
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
 
        # Check for GPU
        has_gpu = self._check_gpu()
 
        if ram_gb >= 16 and has_gpu:
            # High-end system - recommend larger model
            for model in DEFAULT_MODELS:
                if "7b" in model.name.lower() and model.recommended:
                    return model
        elif ram_gb >= 8:
            # Mid-range system
            for model in DEFAULT_MODELS:
                if "7b" in model.name.lower():
                    return model
        else:
            # Low-end system - recommend smaller model
            for model in DEFAULT_MODELS:
                if "3b" in model.name.lower():
                    return model
 
        # Fallback to first recommended
        for model in DEFAULT_MODELS:
            if model.recommended:
                return model
 
        return DEFAULT_MODELS[0]
 
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        import subprocess
 
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
 
 
# Singleton instance
_catalogue_instance: Optional[ModelCatalogue] = None
 
 
def get_model_catalogue() -> ModelCatalogue:
    """Get the singleton ModelCatalogue instance."""
    global _catalogue_instance
    if _catalogue_instance is None:
        _catalogue_instance = ModelCatalogue()
    return _catalogue_instance
 