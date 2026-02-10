"""
Documents API - Endpoints for document management.
"""
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload and process a document."""
    app_state = request.app.state.app_state
    
    if not app_state.document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    # Save uploaded file to temp location
    temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        processed = await app_state.document_processor.process_file(
            temp_path,
            file.filename,
        )
        
        return {
            "success": True,
            "document": {
                "id": processed.id,
                "filename": processed.filename,
                "file_type": processed.file_type,
                "page_count": processed.page_count,
                "word_count": processed.word_count,
                "chunk_count": len(processed.chunks),
            },
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("")
async def list_documents(request: Request):
    """List all uploaded documents."""
    app_state = request.app.state.app_state
    
    if not app_state.document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    documents = await app_state.document_processor.list_documents()
    return documents


@router.get("/{document_id}")
async def get_document(request: Request, document_id: str):
    """Get a specific document."""
    app_state = request.app.state.app_state
    
    if not app_state.document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    document = await app_state.document_processor.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "content": document.content[:5000] + "..." if len(document.content) > 5000 else document.content,
        "chunk_count": len(document.chunks),
        "metadata": document.metadata,
    }


@router.delete("/{document_id}")
async def delete_document(request: Request, document_id: str):
    """Delete a document."""
    app_state = request.app.state.app_state
    
    if not app_state.document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    success = await app_state.document_processor.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"success": True, "deleted": document_id}


@router.post("/{document_id}/reprocess")
async def reprocess_document(request: Request, document_id: str):
    """Reprocess an existing document."""
    app_state = request.app.state.app_state
    
    if not app_state.document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    # Get document path
    doc = await app_state.document_processor.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete and reprocess
    await app_state.document_processor.delete_document(document_id)
    
    stored_path = Path(doc.metadata.get("stored_path", ""))
    if stored_path.exists():
        processed = await app_state.document_processor.process_file(
            stored_path,
            doc.filename,
        )
        return {
            "success": True,
            "document_id": processed.id,
        }
    
    raise HTTPException(status_code=404, detail="Original file not found")