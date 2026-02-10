"""
Models API - Endpoints for model management.
"""
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.get("")
async def list_models(request: Request):
    """List all installed models."""
    from app.services.setup_manager import SetupManager
    
    manager = SetupManager()
    return {
        "installed": manager.get_installed_models(),
        "available": manager.get_available_models(),
    }


@router.get("/active")
async def get_active_model(request: Request):
    """Get the currently active model."""
    app_state = request.app.state.app_state
    
    if app_state.llm_manager:
        return app_state.llm_manager.get_model_info()
    
    return {"error": "No model loaded"}


@router.post("/activate/{model_name}")
async def activate_model(request: Request, model_name: str):
    """Switch to a different model."""
    app_state = request.app.state.app_state
    
    if not app_state.llm_manager:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    success = await app_state.llm_manager.switch_model(model_name)
    
    if success:
        return {"success": True, "model": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")


@router.delete("/{model_name}")
async def delete_model(model_name: str):
    """Delete an installed model."""
    from pathlib import Path
    from app.core.config import settings
    
    model_path = settings.models_dir / "llm" / f"{model_name}.gguf"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_path.unlink()
    
    return {"success": True, "deleted": model_name}