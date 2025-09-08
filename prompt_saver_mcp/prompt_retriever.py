"""Simple prompt retrieval."""

from typing import List, Optional
from .models import PromptSearchResult
from .database import DatabaseManager
from .embeddings import EmbeddingManager


class PromptRetriever:
    """Simple prompt search and retrieval."""

    def __init__(self, database_manager: DatabaseManager, embedding_manager: Optional[EmbeddingManager] = None):
        self.database_manager = database_manager
        self.embedding_manager = embedding_manager
    
    async def search_prompts(self, query: str, limit: int = 3) -> List[PromptSearchResult]:
        """Search for relevant prompts."""
        try:
            # Try vector search first if embedding manager is available
            if self.embedding_manager and self.embedding_manager.is_available():
                query_embedding = self.embedding_manager.embed(query, "query")
                if query_embedding:
                    results = await self.database_manager.search_prompts_by_vector(query_embedding, limit)
                    if results:
                        return results

            # Fallback to text search
            return await self.database_manager.search_prompts_by_text(query, limit)

        except Exception:
            # Final fallback to text search
            try:
                return await self.database_manager.search_prompts_by_text(query, limit)
            except Exception:
                return []

    async def get_prompt_by_id(self, prompt_id: str):
        """Get prompt by ID."""
        return await self.database_manager.get_prompt_by_id(prompt_id)

    def create_selection_prompt(self, results: List[PromptSearchResult]) -> str:
        """Create user selection prompt."""
        if not results:
            return "No prompts found."

        text = "Found relevant prompts:\n\n"
        for i, result in enumerate(results, 1):
            score = f" ({result.score:.1%})" if result.score else ""
            text += f"{i}. {result.use_case}: {result.summary}{score}\n"

        return text + f"\nSelect 1-{len(results)} or search again."

    async def search_by_use_case(self, use_case: str, limit: int = 5) -> List[PromptSearchResult]:
        """Search for prompts by use case category."""
        return await self.database_manager.search_prompts_by_use_case(use_case, limit)
