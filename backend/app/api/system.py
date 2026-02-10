"""
System API - Endpoints for system information and management.
"""
import psutil
import shutil
from pathlib import Path
from fastapi import APIRouter, Request

from app.core.config import settings


router = APIRouter()


@router.get("/storage")
async def get_storage_info():
    """Get storage usage breakdown."""
    def get_dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    models_size = get_dir_size(settings.models_dir)
    vectors_size = get_dir_size(settings.vectors_dir)
    documents_size = get_dir_size(settings.documents_dir)
    cache_size = get_dir_size(settings.cache_dir)
    outputs_size = get_dir_size(settings.outputs_dir)

    total_used = models_size + vectors_size + documents_size + cache_size + outputs_size

    # Get disk info
    disk = psutil.disk_usage('/')

    return {
        "total_used_bytes": total_used,
        "total_used_gb": round(total_used / (1024 ** 3), 2),
        "breakdown": {
            "models": {
                "bytes": models_size,
                "gb": round(models_size / (1024 ** 3), 2),
            },
            "vector_database": {
                "bytes": vectors_size,
                "gb": round(vectors_size / (1024 ** 3), 2),
            },
            "documents": {
                "bytes": documents_size,
                "gb": round(documents_size / (1024 ** 3), 2),
            },
            "cache": {
                "bytes": cache_size,
                "gb": round(cache_size / (1024 ** 3), 2),
            },
            "outputs": {
                "bytes": outputs_size,
                "gb": round(outputs_size / (1024 ** 3), 2),
            },
        },
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "used_percent": disk.percent,
        },
    }


@router.post("/clear_cache")
async def clear_cache():
    """Clear temporary cache files."""
    cache_dir = settings.cache_dir

    if cache_dir.exists():
        # Delete all files in cache
        for item in cache_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    return {
        "success": True,
        "message": "Cache cleared",
    }


@router.get("/performance")
async def get_performance_stats():
    """Get current system performance statistics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # GPU info (if available)
    gpu_info = None
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            gpu_info = {
                "utilization_percent": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
            }
    except:
        pass

    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count(),
        },
        "memory": {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "percent": memory.percent,
        },
        "gpu": gpu_info,
    }


@router.get("/info")
async def get_system_info(request: Request):
    """Get comprehensive system information."""
    import platform

    app_state = request.app.state.app_state

    return {
        "app": {
            "name": settings.app_name,
            "version": "1.0.0",
            "setup_complete": app_state.setup_complete,
            "llm_ready": app_state.llm_ready,
            "vectorstore_ready": app_state.vectorstore_ready,
        },
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        },
        "paths": {
            "models": str(settings.models_dir),
            "data": str(settings.data_dir),
            "outputs": str(settings.outputs_dir),
        },
    }


@router.post("/reset")
async def reset_application(request: Request, confirm: bool = False):
    """Reset the application (clear all data)."""
    if not confirm:
        return {
            "warning": "This will delete all documents, vectors, and settings. Set confirm=true to proceed.",
            "confirm_required": True,
        }

    # Clear vector store
    if request.app.state.app_state.vector_store:
        await request.app.state.app_state.vector_store.clear_collection()

    # Clear documents
    if settings.documents_dir.exists():
        shutil.rmtree(settings.documents_dir)
        settings.documents_dir.mkdir(parents=True, exist_ok=True)

    # Clear outputs
    if settings.outputs_dir.exists():
        shutil.rmtree(settings.outputs_dir)
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    # Clear cache
    if settings.cache_dir.exists():
        shutil.rmtree(settings.cache_dir)
        settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # Remove setup marker
    setup_marker = settings.data_dir / ".setup_complete"
    setup_marker.unlink(missing_ok=True)

    return {
        "success": True,
        "message": "Application reset complete. Please restart the application.",
    }


@router.get("/logs")
async def get_recent_logs(lines: int = 100):
    """Get recent application logs."""
    # This is a placeholder - in production, you'd read from a log file
    return {
        "message": "Logs endpoint - configure logging to file for this feature",
        "lines_requested": lines,
    }