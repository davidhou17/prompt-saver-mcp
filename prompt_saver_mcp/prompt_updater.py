"""Simple prompt updating."""

from typing import Dict, Any, Optional
from .models import PromptUpdate
from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .llm_service import LLMService


class PromptUpdater:
    """Simple prompt updating."""

    def __init__(self, database_manager: DatabaseManager, embedding_manager: EmbeddingManager, llm_service: LLMService):
        self.database_manager = database_manager
        self.embedding_manager = embedding_manager
        self.llm_service = llm_service
    
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

    async def improve_prompt_from_feedback(self, prompt_id: str, feedback: str, conversation_context: Optional[str] = None) -> bool:
        """Improve a prompt based on user feedback using AI analysis."""
        try:
            # First, get the current prompt
            current_prompt_data = await self.database_manager.get_prompt_by_id(prompt_id)
            if not current_prompt_data:
                return False

            current_prompt_template = current_prompt_data.prompt_template
            if not current_prompt_template:
                return False

            # Use LLM to improve the prompt based on feedback
            improvement_result = await self.llm_service.improve_prompt_based_on_feedback(
                current_prompt_template, feedback, conversation_context
            )

            improved_prompt = improvement_result.get("improved_prompt", "")
            changes_summary = improvement_result.get("changes_summary", "Improved based on user feedback")

            if not improved_prompt:
                return False

            # Update the prompt with the improved version
            updates = {
                "prompt_template": improved_prompt,
                "embedding": self.embedding_manager.embed(current_prompt_data.summary, "document")
            }

            change_description = f"AI-improved based on feedback: {changes_summary}"

            return await self.database_manager.update_prompt(prompt_id, updates, change_description)

        except Exception:
            return False
