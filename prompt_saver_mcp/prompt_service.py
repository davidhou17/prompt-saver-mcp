"""Unified prompt service for processing, retrieval, and updating."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import ConversationHistory, PromptTemplate, PromptUpdate, PromptSearchResult
from .storage import StorageManager
from .embeddings import EmbeddingManager
from .llm_service import LLMService


class PromptService:
    """Unified service for all prompt operations.
    
    Combines functionality from prompt processing, retrieval, and updating
    into a single cohesive service.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        embedding_manager: Optional[EmbeddingManager],
        llm_service: LLMService
    ):
        self.storage = storage_manager
        self.embeddings = embedding_manager
        self.llm = llm_service

    def _get_embedding(self, text: str, input_type: str = "document") -> Optional[List[float]]:
        """Get embedding if available, otherwise return None."""
        if self.embeddings and self.embeddings.is_available():
            return self.embeddings.embed(text, input_type)
        return None

    # ==================== Creation ====================

    async def create_prompt(self, conversation: ConversationHistory) -> PromptTemplate:
        """Analyze conversation and create a prompt template."""
        use_case = await self.llm.categorize_conversation(conversation)
        title = await self.llm.generate_title(conversation)
        summary = await self.llm.generate_summary(conversation)
        prompt_template = await self.llm.create_prompt_template(conversation, use_case)
        history = await self.llm.extract_history_summary(conversation)
        embedding = self._get_embedding(summary, "document")

        return PromptTemplate(
            use_case=use_case,
            title=title,
            summary=summary,
            prompt_template=prompt_template,
            history=history,
            embedding=embedding,
            last_updated=datetime.now(timezone.utc),
            num_updates=0,
            changelog=["Initial creation"]
        )

    async def save_prompt(self, conversation: ConversationHistory) -> tuple[str, PromptTemplate]:
        """Create and save a prompt, returning the ID and template."""
        template = await self.create_prompt(conversation)
        prompt_id = await self.storage.save_prompt(template)
        return prompt_id, template

    # ==================== Retrieval ====================

    async def search(self, query: str, limit: int = 3) -> List[PromptSearchResult]:
        """Search for prompts using vector or text search."""
        try:
            # Try vector search first if embeddings available
            if self.embeddings and self.embeddings.is_available():
                query_embedding = self.embeddings.embed(query, "query")
                if query_embedding:
                    try:
                        results = await self.storage.search_prompts_by_vector(query_embedding, limit)
                        if results:
                            return results
                    except NotImplementedError:
                        pass  # Fall through to text search

            return await self.storage.search_prompts_by_text(query, limit)
        except Exception:
            try:
                return await self.storage.search_prompts_by_text(query, limit)
            except Exception:
                return []

    async def search_by_use_case(self, use_case: str, limit: int = 5) -> List[PromptSearchResult]:
        """Search prompts by use case category."""
        return await self.storage.search_prompts_by_use_case(use_case, limit)

    async def get_by_id(self, prompt_id: str) -> Optional[PromptSearchResult]:
        """Get a prompt by its ID."""
        return await self.storage.get_prompt_by_id(prompt_id)

    def format_search_results(self, results: List[PromptSearchResult]) -> str:
        """Format search results for user selection."""
        if not results:
            return "No prompts found."

        text = "Found relevant prompts:\n\n"
        for i, result in enumerate(results, 1):
            score = f" ({result.score:.1%})" if result.score else ""
            text += f"{i}. {result.use_case}: {result.summary}{score}\n"

        return text + f"\nSelect 1-{len(results)} or search again."

    # ==================== Updating ====================

    async def update(self, update_data: PromptUpdate) -> bool:
        """Update an existing prompt with manual changes."""
        updates: Dict[str, Any] = {}

        if update_data.summary:
            updates["summary"] = update_data.summary
            embedding = self._get_embedding(update_data.summary, "document")
            if embedding:
                updates["embedding"] = embedding

        if update_data.prompt_template:
            updates["prompt_template"] = update_data.prompt_template
        if update_data.history:
            updates["history"] = update_data.history
        if update_data.use_case:
            updates["use_case"] = update_data.use_case

        return await self.storage.update_prompt(
            update_data.prompt_id, updates, update_data.change_description
        )

    async def improve_from_feedback(
        self,
        prompt_id: str,
        feedback: str,
        conversation_context: Optional[str] = None
    ) -> bool:
        """Improve a prompt based on user feedback using AI analysis."""
        try:
            current = await self.storage.get_prompt_by_id(prompt_id)
            if not current or not current.prompt_template:
                return False

            result = await self.llm.improve_prompt_based_on_feedback(
                current.prompt_template, feedback, conversation_context
            )

            improved_prompt = result.get("improved_prompt", "")
            if not improved_prompt:
                return False

            updates: Dict[str, Any] = {"prompt_template": improved_prompt}
            embedding = self._get_embedding(current.summary, "document")
            if embedding:
                updates["embedding"] = embedding

            change_description = f"AI-improved: {result.get('changes_summary', 'Based on feedback')}"
            return await self.storage.update_prompt(prompt_id, updates, change_description)

        except Exception:
            return False

