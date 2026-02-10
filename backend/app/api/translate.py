"""
Translate API - Endpoints for offline translation.
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class TranslateRequest(BaseModel):
    """Request model for translation."""
    text: str
    source_lang: str = "en"
    target_lang: str = "hi"


class BatchTranslateRequest(BaseModel):
    """Request model for batch translation."""
    texts: List[str]
    source_lang: str = "en"
    target_lang: str = "hi"


@router.post("")
async def translate(request: Request, body: TranslateRequest):
    """Translate text between languages."""
    app_state = request.app.state.app_state
    
    translator = await app_state.get_translator()
    
    if not translator.is_available(body.source_lang, body.target_lang):
        raise HTTPException(
            status_code=503,
            detail=f"Translation not available for {body.source_lang} -> {body.target_lang}. "
                   "Install the language pair from Settings.",
        )
    
    result = await translator.translate(
        body.text,
        body.source_lang,
        body.target_lang,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed"))
    
    return result


@router.post("/batch")
async def translate_batch(request: Request, body: BatchTranslateRequest):
    """Translate multiple texts."""
    app_state = request.app.state.app_state
    
    translator = await app_state.get_translator()
    
    if not translator.is_available(body.source_lang, body.target_lang):
        raise HTTPException(
            status_code=503,
            detail=f"Translation not available for {body.source_lang} -> {body.target_lang}",
        )
    
    result = await translator.translate_batch(
        body.texts,
        body.source_lang,
        body.target_lang,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed"))
    
    return result


@router.get("/languages")
async def get_available_languages(request: Request):
    """Get available language pairs."""
    app_state = request.app.state.app_state
    
    translator = await app_state.get_translator()
    
    return {
        "pairs": translator.get_available_pairs(),
        "status": translator.get_status(),
    }


@router.post("/detect")
async def detect_language(request: Request, text: str):
    """Detect the language of text."""
    app_state = request.app.state.app_state
    
    translator = await app_state.get_translator()
    
    result = await translator.detect_language(text)
    
    return result