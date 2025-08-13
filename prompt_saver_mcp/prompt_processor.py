"""Simple prompt processing."""

from datetime import datetime, timezone
from .models import ConversationHistory, PromptTemplate
from .embeddings import EmbeddingManager
from .llm_service import LLMService


class PromptProcessor:
    """Processes conversations into prompt templates."""

    def __init__(self, embedding_manager: EmbeddingManager, llm_service: LLMService):
        self.embedding_manager = embedding_manager
        self.llm_service = llm_service

    async def analyze_conversation(self, conversation: ConversationHistory) -> PromptTemplate:
        """Analyze conversation and create prompt template."""
        use_case = await self.llm_service.categorize_conversation(conversation)
        summary = await self.llm_service.generate_summary(conversation)
        prompt_template = await self.llm_service.create_prompt_template(conversation, use_case)
        history = await self.llm_service.extract_history_summary(conversation)
        embedding = self.embedding_manager.embed(summary, "document")

        return PromptTemplate(
            use_case=use_case,
            summary=summary,
            prompt_template=prompt_template,
            history=history,
            embedding=embedding,
            last_updated=datetime.now(timezone.utc),
            num_updates=0,
            changelog=["Initial creation"]
        )
