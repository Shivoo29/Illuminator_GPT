import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Body
from pydantic import BaseModel

router = APIRouter()

class CreateChatRequest(BaseModel):
    first_message: Optional[str] = None

class UpdateChatTitleRequest(BaseModel):
    title: str

class AddMessageRequest(BaseModel):
    role: str
    content: str
    references: Optional[List[Dict[str, Any]]] = None

class AddDocumentRequest(BaseModel):
    document_id: str

class GenerateTitleRequest(BaseModel):
    content: str

@router.get("")
async def list_chats(request: Request):
    """List all user chats."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    return app_state.chat_manager.list_chats()

@router.post("")
async def create_chat(request: Request, body: CreateChatRequest):
    """Create a new chat session."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    return app_state.chat_manager.create_chat(body.first_message)

@router.get("/{chat_id}")
async def get_chat(request: Request, chat_id: str):
    """Retrieve chat session details."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    chat = app_state.chat_manager.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.put("/{chat_id}/title")
async def update_chat_title(request: Request, chat_id: str, body: UpdateChatTitleRequest):
    """Update title of a chat."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    success = app_state.chat_manager.update_chat_title(chat_id, body.title)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}

@router.delete("/{chat_id}")
async def delete_chat(request: Request, chat_id: str):
    """Delete a chat session."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    success = app_state.chat_manager.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}

@router.post("/{chat_id}/messages")
async def add_message(request: Request, chat_id: str, body: AddMessageRequest):
    """Add a message to a chat."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    chat = app_state.chat_manager.add_message(chat_id, body.role, body.content, body.references)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.post("/{chat_id}/documents")
async def add_document(request: Request, chat_id: str, body: AddDocumentRequest):
    """Attach a document to a chat."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    success = app_state.chat_manager.add_document(chat_id, body.document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}

@router.delete("/{chat_id}/documents/{document_id}")
async def remove_document(request: Request, chat_id: str, document_id: str):
    """Detach a document from a chat."""
    app_state = request.app.state.app_state
    if not hasattr(app_state, "chat_manager"):
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    success = app_state.chat_manager.remove_document(chat_id, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}

@router.post("/generate_title")
async def generate_title(request: Request, body: GenerateTitleRequest):
    """Generate a title for the chat using the LLM based on the user's first message."""
    app_state = request.app.state.app_state
    
    if not app_state.llm_ready:
        return {"title": "New Chat"}
        
    try:
        llm = await app_state.get_llm_manager()
        prompt = f"Write a short, summarizing 3-5 word concise title for a chat that begins with this user message: '{body.content}'. Just return the title itself accurately describing the main topic."
        
        # We can't use llm.generate stream in standard mode, so let's import GenerationConfig
        from app.services.llm_manager import GenerationConfig
        config = GenerationConfig(max_tokens=20, temperature=0.7)
        title = await llm.generate(prompt, config)
        title = title.strip().strip('"').strip("'")
        
        if not title:
            title = "New Chat"
            
        return {"title": title}
    except Exception as e:
        print("Error generating title:", e)
        return {"title": "New Chat"}
