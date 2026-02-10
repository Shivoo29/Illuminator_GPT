"""API routers module."""
from app.api.setup import router as setup_router
from app.api.models import router as models_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.generate import router as generate_router
from app.api.translate import router as translate_router
from app.api.system import router as system_router

__all__ = [
    "setup_router",
    "models_router",
    "documents_router",
    "query_router",
    "generate_router",
    "translate_router",
    "system_router",
]