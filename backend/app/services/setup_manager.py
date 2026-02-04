'''
Docstring for backend.app.services.setup_manager
Handles first time setup, Ollama installation and model downloads.
'''
import os
import sys
import json
import platform
import subprocess
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List, Callable
from dataclasses import dataclass, asdict
 
import psutil
import aiohttp
import aiofiles
 
from app.core.config import settings
 
from app.services.model_catalogue import (
    ModelCatalogue,
    ModelInfo,
    DEFAULT_MODELS,
    get_model_catalogue,
)
 
 
@dataclass
class SystemInfo:
    """System information dataclass."""
    platform: str
    platform_version: str
    cpu_count: int
    ram_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    sufficient: bool = False
 
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
 
 
# Re-export for backward compatibility
AVAILABLE_MODELS = DEFAULT_MODELS

class SetupManager:
    """
    Manages first-time setup, including:
    - System requirements check
    - Ollama detection and installation
    - Model download and management
    - Optional feature downloads (TTS, Image Gen, Translation)
    """
 
    def __init__(self):
        self.platform = platform.system()
        self.models_dir = settings.models_dir
        self.ollama_installed = False
        self._download_session: Optional[aiohttp.ClientSession] = None
 
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._download_session is None or self._download_session.closed:
            self._download_session = aiohttp.ClientSession()
        return self._download_session
 
    async def close(self):
        """Close the aiohttp session and model catalogue."""
        if self._download_session and not self._download_session.closed:
            await self._download_session.close()
        await self._model_catalogue.close()

    def check_system(self) -> SystemInfo:
        """Check system requirements and return system information."""
        # Get disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
 
        # Get RAM
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
 
        # Check GPU
        gpu_available, gpu_name = self._check_gpu()
 
        # Determine if system is sufficient
        sufficient = disk_free_gb > 20 and ram_gb > 8
 
        return SystemInfo(
            platform=self.platform,
            platform_version=platform.version(),
            cpu_count=psutil.cpu_count(),
            ram_gb=round(ram_gb, 1),
            disk_free_gb=round(disk_free_gb, 1),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            sufficient=sufficient,
        )
 
    def _check_gpu(self) -> tuple[bool, Optional[str]]:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip().split('\n')[0]
                return True, gpu_name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False, None
 
    def check_ollama(self) -> Dict[str, Any]:
        """Check if Ollama is installed and get available models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
 
            if result.returncode == 0:
                self.ollama_installed = True
                models = self._parse_ollama_list(result.stdout)
                return {
                    "installed": True,
                    "models": models,
                    "version": self._get_ollama_version(),
                }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
 
        return {"installed": False, "models": [], "version": None}
 
    def _get_ollama_version(self) -> Optional[str]:
        """Get Ollama version."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
 
    def _parse_ollama_list(self, output: str) -> List[Dict[str, str]]:
        """Parse 'ollama list' output."""
        models = []
        lines = output.strip().split('\n')
 
        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                models.append({
                    "name": parts[0],
                    "size": parts[1] if len(parts) > 1 else "unknown",
                    "modified": parts[2] if len(parts) > 2 else "",
                })
 
        return models
 
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for download (offline default models)."""
        return [model.to_dict() for model in AVAILABLE_MODELS]
 
    async def get_available_models_with_online(
        self,
        include_online: bool = True,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get all available models including online models if connected.
 
        Returns:
            Dict with:
            - is_online: Whether we have internet connectivity
            - local_models: Models already downloaded
            - default_models: Bundled recommended models
            - online_models: Additional models from HuggingFace (if online)
            - ollama_models: Models available in Ollama
        """
        return await self._model_catalogue.get_available_models(
            include_online=include_online,
            category=category,
        )
 
    def get_installed_models(self) -> List[Dict[str, Any]]:
        """Get list of locally installed models."""
        return [m.to_dict() for m in self._model_catalogue.get_local_models()]
 
    def get_recommended_model(self) -> Dict[str, Any]:
        """Get the recommended model based on system specs."""
        return self._model_catalogue.get_recommended_model().to_dict()
 
    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific model by name."""
        model = self._model_catalogue.get_model_by_name(name)
        return model.to_dict() if model else None

 
    async def install_ollama(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Download and install Ollama.
        Yields progress updates.
        """
        if self.platform == "Windows":
            url = "https://ollama.com/download/OllamaSetup.exe"
            installer = "OllamaSetup.exe"
        elif self.platform == "Linux":
            url = "https://ollama.com/install.sh"
            installer = "install.sh"
        elif self.platform == "Darwin":
            url = "https://ollama.com/download/Ollama-darwin.zip"
            installer = "Ollama-darwin.zip"
        else:
            yield {"status": "error", "message": f"Unsupported platform: {self.platform}"}
            return
 
        yield {"status": "downloading", "message": "Downloading Ollama installer..."}
 
        # Download installer
        installer_path = self.models_dir / installer
        try:
            async for progress in self._download_file(url, installer_path):
                yield {"status": "downloading", **progress}
        except Exception as e:
            yield {"status": "error", "message": f"Download failed: {str(e)}"}
            return
 
        yield {"status": "installing", "message": "Installing Ollama..."}
 
        # Run installer
        try:
            if self.platform == "Windows":
                subprocess.run([str(installer_path), "/S"], check=True)
            elif self.platform == "Linux":
                subprocess.run(["bash", str(installer_path)], check=True)
            elif self.platform == "Darwin":
                # Extract and copy to Applications
                subprocess.run(["unzip", "-o", str(installer_path), "-d", "/Applications"], check=True)
 
            # Clean up installer
            installer_path.unlink(missing_ok=True)
 
            yield {"status": "verifying", "message": "Verifying installation..."}
 
            # Verify installation
            await asyncio.sleep(2)  # Give Ollama time to initialize
            result = self.check_ollama()
 
            if result["installed"]:
                yield {"status": "complete", "message": "Ollama installed successfully!", "result": result}
            else:
                yield {"status": "error", "message": "Installation completed but Ollama not responding"}
 
        except subprocess.CalledProcessError as e:
            yield {"status": "error", "message": f"Installation failed: {str(e)}"}
        except Exception as e:
            yield {"status": "error", "message": f"Unexpected error: {str(e)}"}
 
    async def download_model_ollama(
        self,
        model_name: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Download a model using Ollama.
        Yields progress updates.
        """
        yield {"status": "starting", "message": f"Pulling {model_name}..."}
 
        process = await asyncio.create_subprocess_exec(
            "ollama", "pull", model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
 
        async for line in process.stdout:
            line_text = line.decode().strip()
            progress = self._parse_ollama_progress(line_text)
            yield {"status": "downloading", **progress}
 
        await process.wait()
 
        if process.returncode == 0:
            yield {"status": "complete", "message": f"{model_name} downloaded successfully!"}
        else:
            stderr = await process.stderr.read()
            yield {"status": "error", "message": f"Download failed: {stderr.decode()}"}
 
    def _parse_ollama_progress(self, line: str) -> Dict[str, Any]:
        """Parse Ollama pull output for progress information."""
        result = {"message": line, "progress_percent": 0}
 
        if "%" in line:
            try:
                # Extract percentage from lines like "pulling 8934d96d3f08... 32%"
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        result["progress_percent"] = float(part.rstrip('%'))
                        break
            except:
                pass
 
        return result
 
    async def download_model_direct(
        self,
        model_name: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Download a model directly from HuggingFace (fallback if Ollama fails).
        Yields progress updates.
        """
        # Find model info
        model_info = next(
            (m for m in AVAILABLE_MODELS if m.name == model_name),
            None
        )
 
        if not model_info or not model_info.huggingface_url:
            yield {"status": "error", "message": f"Unknown model: {model_name}"}
            return
 
        url = model_info.huggingface_url
        filename = url.split('/')[-1]
        output_path = self.models_dir / "llm" / filename
 
        yield {"status": "starting", "message": f"Downloading {model_info.display_name}..."}
 
        try:
            async for progress in self._download_file(url, output_path):
                yield {"status": "downloading", **progress}
 
            yield {"status": "complete", "message": f"{model_info.display_name} downloaded successfully!",
                   "path": str(output_path)}
 
        except Exception as e:
            yield {"status": "error", "message": f"Download failed: {str(e)}"}
 
    async def _download_file(
        self,
        url: str,
        destination: Path,
        chunk_size: int = 8192,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Download a file with progress tracking.
        Supports resumable downloads.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
 
        session = await self._get_session()
 
        # Check if partial download exists
        downloaded = 0
        headers = {}
        if destination.exists():
            downloaded = destination.stat().st_size
            headers["Range"] = f"bytes={downloaded}-"
 
        async with session.get(url, headers=headers) as response:
            if response.status == 416:  # Range not satisfiable - file complete
                yield {
                    "downloaded_bytes": downloaded,
                    "total_bytes": downloaded,
                    "progress_percent": 100,
                    "speed_mbps": 0,
                }
                return
 
            response.raise_for_status()
 
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if response.status == 206:  # Partial content
                total_size += downloaded
            else:
                downloaded = 0  # Server doesn't support range, start over
 
            mode = 'ab' if downloaded > 0 else 'wb'
 
            import time
            start_time = time.time()
            last_update_time = start_time
 
            async with aiofiles.open(destination, mode) as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    await f.write(chunk)
                    downloaded += len(chunk)
 
                    current_time = time.time()
                    if current_time - last_update_time >= 0.5:  # Update every 500ms
                        elapsed = current_time - start_time
                        speed_mbps = (downloaded / elapsed) / (1024 * 1024) if elapsed > 0 else 0
 
                        yield {
                            "downloaded_bytes": downloaded,
                            "total_bytes": total_size,
                            "progress_percent": (downloaded / total_size * 100) if total_size > 0 else 0,
                            "speed_mbps": round(speed_mbps, 2),
                        }
                        last_update_time = current_time
 
        # Final update
        yield {
            "downloaded_bytes": downloaded,
            "total_bytes": total_size,
            "progress_percent": 100,
            "speed_mbps": 0,
        }
 
    async def download_tts_models(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Download piper TTS voice models."""
        voices = {
            "male": {
                "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            },
            "female": {
                "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
                "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
            },
        }
 
        tts_dir = self.models_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
 
        for voice_name, urls in voices.items():
            yield {"status": "downloading", "voice": voice_name, "message": f"Downloading {voice_name} voice..."}
 
            for file_type, url in urls.items():
                filename = url.split('/')[-1]
                output_path = tts_dir / filename
 
                async for progress in self._download_file(url, output_path):
                    yield {"status": "downloading", "voice": voice_name, "file": file_type, **progress}
 
        yield {"status": "complete", "message": "TTS voices downloaded successfully!"}
 
    async def download_embedding_model(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Download the embedding model."""
        yield {"status": "starting", "message": "Downloading embedding model..."}
 
        # The sentence-transformers library handles downloading automatically
        # This is a placeholder for manual download if needed
        try:
            from sentence_transformers import SentenceTransformer
 
            yield {"status": "downloading", "message": "Loading embedding model (will download if needed)..."}
 
            # This will download if not cached
            model = SentenceTransformer(settings.embedding_model)
 
            yield {"status": "complete", "message": "Embedding model ready!"}
 
        except Exception as e:
            yield {"status": "error", "message": f"Failed to download embedding model: {str(e)}"}
 
    def get_feature_status(self) -> Dict[str, Any]:
        """Get status of optional features."""
        return {
            "tts": {
                "installed": self._check_tts_installed(),
                "size_gb": 0.1,
            },
            "image_generation": {
                "installed": self._check_image_gen_installed(),
                "size_gb": 2.0,
            },
            "translation": {
                "installed": self._check_translation_installed(),
                "size_gb": 1.5,
            },
        }
 
    def _check_tts_installed(self) -> bool:
        """Check if TTS models are installed."""
        tts_dir = self.models_dir / "tts"
        male_voice = tts_dir / "en_US-lessac-medium.onnx"
        female_voice = tts_dir / "en_US-amy-medium.onnx"
        return male_voice.exists() and female_voice.exists()
 
    def _check_image_gen_installed(self) -> bool:
        """Check if image generation model is installed."""
        image_dir = self.models_dir / "image_gen"
        return (image_dir / "model_index.json").exists()
 
    def _check_translation_installed(self) -> bool:
        """Check if translation models are installed."""
        trans_dir = self.models_dir / "translation"
        return any(trans_dir.glob("opus-mt-*"))
 
    def calculate_total_download_size(
        self,
        model_name: str,
        include_tts: bool = False,
        include_image_gen: bool = False,
        include_translation: bool = False,
    ) -> float:
        """Calculate total download size for selected options."""
        total = 0.0
 
        # LLM model size
        model_info = next(
            (m for m in AVAILABLE_MODELS if m.name == model_name),
            None
        )
        if model_info:
            total += model_info.size_gb
 
        # Optional features
        if include_tts:
            total += 0.1
        if include_image_gen:
            total += 2.0
        if include_translation:
            total += 1.5
 
        return round(total, 1)
