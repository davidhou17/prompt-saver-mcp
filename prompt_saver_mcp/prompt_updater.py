"""Simple prompt updating."""

from typing import Dict, Any
from .models import PromptUpdate
from .database import DatabaseManager
from .embeddings import EmbeddingManager


class PromptUpdater:
    """Simple prompt updating."""

    def __init__(self, database_manager: DatabaseManager, embedding_manager: EmbeddingManager):
        self.database_manager = database_manager
        self.embedding_manager = embedding_manager
    
    async def update_prompt(self, update_data: PromptUpdate) -> bool:
        """Update existing prompt."""
        updates = {}

        if update_data.summary:
            updates["summary"] = update_data.summary
            updates["embedding"] = self.embedding_manager.embed(update_data.summary, "document")

        if update_data.prompt_template:
            updates["prompt_template"] = update_data.prompt_template

        if update_data.history:
            updates["history"] = update_data.history

        if update_data.use_case:
            updates["use_case"] = update_data.use_case

        return await self.database_manager.update_prompt(
            update_data.prompt_id, updates, update_data.change_description
        )
