"""
Setup API - Endpoints for first-time setup wizard.
"""
import asyncio
from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from app.services.setup_manager import SetupManager, AVAILABLE_MODELS


router = APIRouter()


class ModelDownloadRequest(BaseModel):
    """Request model for downloading a model."""
    model_name: str
    use_ollama: bool = True


class FeatureDownloadRequest(BaseModel):
    """Request model for downloading optional features."""
    feature: str  # 'tts', 'image_generation', 'translation'


@router.get("/check_system")
async def check_system():
    """Check system requirements."""
    manager = SetupManager()
    return manager.check_system().to_dict()


@router.get("/check_ollama")
async def check_ollama():
    """Check if Ollama is installed."""
    manager = SetupManager()
    return manager.check_ollama()


@router.post("/install_ollama")
async def install_ollama():
    """Install Ollama (streaming progress)."""
    manager = SetupManager()

    async def stream_progress():
        try:
            async for progress in manager.install_ollama():
                yield json.dumps(progress) + "\n"
        finally:
            await manager.close()

    return StreamingResponse(
        stream_progress(),
        media_type="application/x-ndjson",
    )


@router.get("/available_models")
async def get_available_models(
    include_online: bool = True,
    category: Optional[str] = None,
):
    """
    Get list of available models for download.

    When online, fetches additional models from HuggingFace.
    When offline, returns only bundled default models.

    Args:
        include_online: Whether to try fetching online models
        category: Filter by category (general, coding, multilingual)

    Returns:
        - is_online: Whether we have internet connectivity
        - local_models: Models already downloaded locally
        - default_models: Bundled recommended models
        - online_models: Additional models from HuggingFace (if online)
        - ollama_models: Models available in Ollama
        - recommended: System-recommended model
    """
    manager = SetupManager()
    try:
        result = await manager.get_available_models_with_online(
            include_online=include_online,
            category=category,
        )
        result["recommended"] = manager.get_recommended_model()
        return result
    finally:
        await manager.close()


@router.get("/available_models/offline")
async def get_available_models_offline():
    """Get list of available models (offline only, no network requests)."""
    manager = SetupManager()
    return {
        "models": manager.get_available_models(),
        "installed": manager.get_installed_models(),
        "recommended": manager.get_recommended_model(),
    }


@router.post("/download_model")
async def download_model(request: ModelDownloadRequest):
    """Download a language model (streaming progress)."""
    manager = SetupManager()

    async def stream_progress():
        try:
            if request.use_ollama:
                # Find ollama name for model
                model_info = next(
                    (m for m in AVAILABLE_MODELS if m.name == request.model_name),
                    None
                )
                if model_info and model_info.ollama_name:
                    async for progress in manager.download_model_ollama(model_info.ollama_name):
                        yield json.dumps(progress) + "\n"
                else:
                    yield json.dumps({"status": "error", "message": "Model not found"}) + "\n"
            else:
                async for progress in manager.download_model_direct(request.model_name):
                    yield json.dumps(progress) + "\n"
        finally:
            await manager.close()

    return StreamingResponse(
        stream_progress(),
        media_type="application/x-ndjson",
    )


@router.post("/import_model")
async def import_model(file_path: str):
    """Import a local .gguf model file."""
    from pathlib import Path
    import shutil
    from app.core.config import settings

    source = Path(file_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not source.suffix == ".gguf":
        raise HTTPException(status_code=400, detail="File must be a .gguf file")

    # Copy to models directory
    dest = settings.models_dir / "llm" / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    return {
        "success": True,
        "model_path": str(dest),
        "model_name": source.stem,
    }


@router.post("/download_feature")
async def download_feature(request: FeatureDownloadRequest):
    """Download optional feature models (streaming progress)."""
    manager = SetupManager()

    async def stream_progress():
        try:
            if request.feature == "tts":
                async for progress in manager.download_tts_models():
                    yield json.dumps(progress) + "\n"
            elif request.feature == "embedding":
                async for progress in manager.download_embedding_model():
                    yield json.dumps(progress) + "\n"
            else:
                yield json.dumps({"status": "error", "message": f"Unknown feature: {request.feature}"}) + "\n"
        finally:
            await manager.close()

    return StreamingResponse(
        stream_progress(),
        media_type="application/x-ndjson",
    )


@router.get("/feature_status")
async def get_feature_status():
    """Get status of optional features."""
    manager = SetupManager()
    return manager.get_feature_status()


@router.post("/calculate_download_size")
async def calculate_download_size(
    model_name: str,
    include_tts: bool = False,
    include_image_gen: bool = False,
    include_translation: bool = False,
):
    """Calculate total download size for selected options."""
    manager = SetupManager()
    total_gb = manager.calculate_total_download_size(
        model_name,
        include_tts,
        include_image_gen,
        include_translation,
    )

    # Estimate time (assuming 5 MB/s average)
    estimate_minutes = int((total_gb * 1024) / 5 / 60)

    return {
        "total_gb": total_gb,
        "estimate_minutes": estimate_minutes,
        "breakdown": {
            "model": next((m.size_gb for m in AVAILABLE_MODELS if m.name == model_name), 0),
            "tts": 0.1 if include_tts else 0,
            "image_gen": 2.0 if include_image_gen else 0,
            "translation": 1.5 if include_translation else 0,
        }
    }


@router.post("/complete_setup")
async def complete_setup(request: Request):
    """Mark setup as complete."""
    app_state = request.app.state.app_state
    app_state.mark_setup_complete()

    # Initialize components
    await app_state._initialize_components()

    return {
        "success": True,
        "message": "Setup completed successfully!",
    }


@router.get("/setup_status")
async def get_setup_status(request: Request):
    """Get current setup status."""
    app_state = request.app.state.app_state

    manager = SetupManager()
    system_info = manager.check_system()
    ollama_status = manager.check_ollama()
    feature_status = manager.get_feature_status()
    installed_models = manager.get_installed_models()

    return {
        "setup_complete": app_state.setup_complete,
        "system": system_info.to_dict(),
        "ollama": ollama_status,
        "features": feature_status,
        "installed_models": installed_models,
        "llm_ready": app_state.llm_ready,
        "vectorstore_ready": app_state.vectorstore_ready,
    }