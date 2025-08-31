from datetime import datetime
import uuid
from typing import List, Dict, Optional
from database.mongodb_config import get_collection

class ConversationManager:
    @staticmethod
    async def create_conversation(title: str = None) -> str:
        """Create a new conversation"""
        collection = await get_collection()
        
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        conversation = {
            "conversation_id": conversation_id,
            "title": title or f"Conversation {now.strftime('%Y-%m-%d %H:%M')}",
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "metadata": {
                "tags": [],
                "is_pinned": False,
                "last_accessed": now
            }
        }
        
        await collection.insert_one(conversation)
        return conversation_id
    
    @staticmethod
    async def add_message(conversation_id: str, role: str, content: str):
        """Add a message to a conversation"""
        collection = await get_collection()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }
        
        # Get the first message content for title if it's a new conversation
        conversation = await collection.find_one({"conversation_id": conversation_id})
        if conversation and len(conversation.get("messages", [])) == 0:
            # Set title based on first message
            truncated_title = content[:50] + "..." if len(content) > 50 else content
            await collection.update_one(
                {"conversation_id": conversation_id},
                {"$set": {"title": truncated_title}}
            )
        
        await collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
    @staticmethod
    async def rename_conversation(conversation_id: str, title: str):
        """Rename a conversation"""
        collection = await get_collection()
        
        await collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    "title": title,
                    "updated_at": datetime.utcnow()
                }
            }
        )
    
    @staticmethod
    async def get_conversation(conversation_id: str) -> Optional[Dict]:
        """Get a specific conversation"""
        collection = await get_collection()
        
        conversation = await collection.find_one(
            {"conversation_id": conversation_id},
            {'_id': 0}  # Exclude MongoDB's _id field
        )
        
        if conversation:
            # Update last accessed time
            await collection.update_one(
                {"conversation_id": conversation_id},
                {"$set": {"metadata.last_accessed": datetime.utcnow()}}
            )
            
        return conversation
    
    @staticmethod
    async def list_conversations(
        skip: int = 0,
        limit: int = 20,
        pinned_first: bool = True
    ) -> List[Dict]:
        """List all conversations with pagination"""
        collection = await get_collection()
        
        # First get pinned conversations if requested
        conversations = []
        if pinned_first:
            pinned = await collection.find(
                {"metadata.is_pinned": True},
                {'_id': 0}
            ).sort("updated_at", -1).to_list(length=None)
            conversations.extend(pinned)
            
            # Adjust skip/limit for remaining conversations
            if pinned:
                skip = max(0, skip - len(pinned))
                limit = max(0, limit - len(pinned))
        
        # Get remaining conversations
        if limit > 0:
            regular = await collection.find(
                {"metadata.is_pinned": {"$ne": True}} if pinned_first else {},
                {'_id': 0}
            ).sort("updated_at", -1).skip(skip).limit(limit).to_list(length=None)
            conversations.extend(regular)
        
        return conversations
    
    @staticmethod
    async def update_conversation(
        conversation_id: str,
        title: str = None,
        tags: List[str] = None,
        is_pinned: bool = None
    ):
        """Update conversation metadata"""
        collection = await get_collection()
        
        update_data = {"updated_at": datetime.utcnow()}
        if title is not None:
            update_data["title"] = title
        if tags is not None:
            update_data["metadata.tags"] = tags
        if is_pinned is not None:
            update_data["metadata.is_pinned"] = is_pinned
        
        await collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": update_data}
        )
    
    @staticmethod
    async def delete_conversation(conversation_id: str):
        """Delete a conversation"""
        collection = await get_collection()
        await collection.delete_one({"conversation_id": conversation_id})
    
    @staticmethod
    async def search_conversations(
        query: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[Dict]:
        """Search conversations by title or content"""
        collection = await get_collection()
        
        # Search in titles and message content
        results = await collection.find(
            {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"messages.content": {"$regex": query, "$options": "i"}}
                ]
            },
            {'_id': 0}
        ).sort("updated_at", -1).skip(skip).limit(limit).to_list(length=None)
        
        return results
