"""
Generate API - Endpoints for content generation (podcast, images, documents).
"""
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class PodcastRequest(BaseModel):
    """Request model for podcast generation."""
    document_id: str
    duration_minutes: int = 10
    host_a_name: str = "Alex"
    host_b_name: str = "Sam"


class ImageRequest(BaseModel):
    """Request model for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_steps: int = 20


@router.post("/podcast")
async def generate_podcast(request: Request, body: PodcastRequest):
    """Generate a podcast from a document (streaming progress)."""
    app_state = request.app.state.app_state
    
    if not app_state.llm_ready:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    # Get document content
    doc = await app_state.document_processor.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get podcast generator
    podcast_gen = await app_state.get_podcast_generator()
    
    if not podcast_gen.is_available():
        # Return script only if TTS not available
        from app.services.podcast_generator import PodcastConfig
        
        config = PodcastConfig(
            duration_minutes=body.duration_minutes,
            host_a_name=body.host_a_name,
            host_b_name=body.host_b_name,
        )
        
        result = await podcast_gen.generate_script_only(
            doc.content,
            doc.filename,
            config,
        )
        
        return {
            "status": "complete",
            "message": "Script generated (TTS not available)",
            "script": result["script"],
            "audio_url": None,
        }
    
    # Stream podcast generation progress
    from app.services.podcast_generator import PodcastConfig
    
    config = PodcastConfig(
        duration_minutes=body.duration_minutes,
        host_a_name=body.host_a_name,
        host_b_name=body.host_b_name,
    )
    
    async def stream_progress():
        async for progress in podcast_gen.generate_podcast(
            doc.content,
            doc.filename,
            config,
        ):
            yield json.dumps(progress) + "\n"
    
    return StreamingResponse(
        stream_progress(),
        media_type="application/x-ndjson",
    )


@router.post("/image")
async def generate_image(request: Request, body: ImageRequest):
    """Generate an image from a text prompt."""
    app_state = request.app.state.app_state
    
    image_gen = await app_state.get_image_generator()
    
    if not image_gen.is_available():
        raise HTTPException(
            status_code=503,
            detail="Image generation not available. Install the model from Settings.",
        )
    
    result = await image_gen.generate(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        num_steps=body.num_steps,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
    
    return result


@router.post("/document")
async def generate_document(
    request: Request,
    prompt: str,
    document_type: str = "report",
    max_length: int = 2000,
):
    """Generate a new document based on a prompt."""
    app_state = request.app.state.app_state
    
    if not app_state.llm_ready:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    from app.services.llm_manager import GenerationConfig
    
    generation_prompt = f"""Generate a {document_type} based on the following request:

{prompt}

Write a well-structured {document_type} with clear sections and professional formatting.
Target length: approximately {max_length} words.

{document_type.title()}:"""
    
    content = await app_state.llm_manager.generate(
        generation_prompt,
        GenerationConfig(max_tokens=max_length * 2, temperature=0.7),
    )
    
    return {
        "content": content.strip(),
        "type": document_type,
        "word_count": len(content.split()),
    }


@router.get("/podcast/status/{task_id}")
async def get_podcast_status(task_id: str):
    """Check status of podcast generation task."""
    # This would be used for async task tracking
    # For now, we use streaming responses instead
    return {"status": "not_implemented"}