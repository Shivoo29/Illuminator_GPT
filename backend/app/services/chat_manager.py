import json
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.config import settings

class ChatManager:
    """Manages chat sessions and stores them as JSON files locally."""
    def __init__(self):
        self.chats_dir = settings.data_dir / "chats"
        self.chats_dir.mkdir(parents=True, exist_ok=True)

    def _get_chat_path(self, chat_id: str) -> Path:
        return self.chats_dir / f"{chat_id}.json"

    def create_chat(self, first_message: Optional[str] = None) -> Dict[str, Any]:
        chat_id = str(uuid.uuid4())
        
        # Will be updated asynchronously by calling llm if first_message exists
        title = "New Chat" 

        now = datetime.datetime.now().isoformat()
        chat = {
            "id": chat_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "document_ids": []
        }
        
        if first_message:
            chat["messages"].append({
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": first_message,
                "timestamp": now
            })

        self.save_chat(chat)
        return chat

    def save_chat(self, chat: Dict[str, Any]):
        chat["updated_at"] = datetime.datetime.now().isoformat()
        path = self._get_chat_path(chat["id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat, f, indent=2)

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        path = self._get_chat_path(chat_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_chats(self) -> List[Dict[str, Any]]:
        chats = []
        for path in self.chats_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chat = json.load(f)
                    chats.append({
                        "id": chat["id"],
                        "title": chat["title"],
                        "created_at": chat["created_at"],
                        "updated_at": chat["updated_at"],
                        "message_count": len(chat.get("messages", [])),
                        "document_count": len(chat.get("document_ids", []))
                    })
            except Exception as e:
                print(f"Error reading chat file {path}: {e}")
                
        # Sort by updated_at descending
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats

    def update_chat_title(self, chat_id: str, title: str) -> bool:
        chat = self.get_chat(chat_id)
        if not chat:
            return False
        chat["title"] = title
        self.save_chat(chat)
        return True

    def delete_chat(self, chat_id: str) -> bool:
        path = self._get_chat_path(chat_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def add_message(self, chat_id: str, role: str, content: str, references: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        chat = self.get_chat(chat_id)
        if not chat:
            return None
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if references:
            message["references"] = references
            
        if "messages" not in chat:
            chat["messages"] = []
            
        chat["messages"].append(message)
        self.save_chat(chat)
        return chat

    def add_document(self, chat_id: str, document_id: str) -> bool:
        chat = self.get_chat(chat_id)
        if not chat:
            return False
            
        if "document_ids" not in chat:
            chat["document_ids"] = []
            
        if document_id not in chat["document_ids"]:
            chat["document_ids"].append(document_id)
            self.save_chat(chat)
            
        return True
        
    def remove_document(self, chat_id: str, document_id: str) -> bool:
        chat = self.get_chat(chat_id)
        if not chat:
            return False
            
        if "document_ids" in chat and document_id in chat["document_ids"]:
            chat["document_ids"].remove(document_id)
            self.save_chat(chat)
            
        return True
