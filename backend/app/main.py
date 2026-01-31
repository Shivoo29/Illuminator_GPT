'''Offline Multimodal RAG - main FastAPI entry point'''
import os
import sys
import pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api import (
    setup_router,
    models_router,
    documents_router,
    query_router,
    generate_router,
    translate_router,
    system_router,
)
from app.core.config import settings
from app.core.state import AppState

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application lifespan handler for startup and shutdown
    # startup
    print("Starting ILLUMINATOR_GPTv0.1......")

    # initialize application state
    app.state.app_state = AppState()
    await app.state.app_state.initialize()

    yield

    # shutdown
    Print("Shitting Down......")
    await app.state.app_state.cleanup()

# FastAPT Application Initialization
app = FastAPI(
    title = "ILLUMINATOR_GPTv0.1",
    description = "A fully offline, cross-platform RAG application with document processing, "
                "podcast generation, image generation, and translation capabilities.", # to be Changed  
                #(CHANGE-Swayam or Ashish)

    version = "0.0.1"
    lifespan = lifespan,
)

# Configure CORS for Tauri frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #  Tauri's custom protocol (CHECK-Swayam)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for outputs
outputs_dir = Path(settings.outputs_dir)
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# API routers
app.include_router(setup_router, prefix="/setup", tags=["Setup"])
app.include_router(moduls_router, prefix="/models", tags=["Models"])
app.include_router(documents_router, prefix="/documents", tags=["Documents"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(generate_router,prefix="/generate", tags=["Generate"])
app.include_router(translate_router,prefix="/translate", tags=["Translate"])
app.include_router(system_router, prefix="/system", tags=["System"])

@app.get("/")
async def root():
    # Root endpoint returning API information.
    return{
        "name":"ILLUMINATOR_GPTv0.1",
        "version":"0.0.1",
        "status":"running",
        "offline": True,
    }

@app.get("/health")
async def health_check():
    # Health Check
    return{
        "status":"healthy",
        "components":{
            "api":"running",
            "llm": app.state.app_state.llm_ready if hasattr(app.state, 'app_state') else False,
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )