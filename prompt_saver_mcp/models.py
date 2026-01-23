"""Data models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UseCase(str, Enum):
    """Valid use case categories for prompts.

    Using str, Enum ensures these serialize as strings for storage compatibility.
    """
    CODE_GEN = "code-gen"
    TEXT_GEN = "text-gen"
    DATA_ANALYSIS = "data-analysis"
    CREATIVE = "creative"
    GENERAL = "general"

    @classmethod
    def from_string(cls, value: str) -> "UseCase":
        """Convert a string to UseCase, defaulting to GENERAL if invalid."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            return cls.GENERAL

    @classmethod
    def values(cls) -> List[str]:
        """Return all valid use case values as strings."""
        return [member.value for member in cls]


class PromptTemplate(BaseModel):
    """Stored prompt template."""

    use_case: str
    title: str  # Short descriptive title for file naming (2-5 words, hyphenated)
    summary: str
    prompt_template: str
    history: str
    embedding: Optional[List[float]] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    num_updates: int = 0
    changelog: List[str] = Field(default_factory=list)


class PromptSearchResult(BaseModel):
    """Search result."""

    id: str
    use_case: str
    title: str  # Short descriptive title for file naming
    summary: str
    prompt_template: str
    history: str
    last_updated: datetime
    num_updates: int
    score: Optional[float] = None

    @classmethod
    def from_mongo_doc(cls, doc: Dict[str, Any], score: Optional[float] = None) -> "PromptSearchResult":
        """Create from MongoDB document."""
        return cls(
            id=str(doc["_id"]),
            use_case=doc["use_case"],
            title=doc.get("title", ""),
            summary=doc["summary"],
            prompt_template=doc["prompt_template"],
            history=doc["history"],
            last_updated=doc["last_updated"],
            num_updates=doc["num_updates"],
            score=score or doc.get("score")
        )

    @classmethod
    def from_file_data(
        cls,
        prompt_id: str,
        metadata: Dict[str, Any],
        prompt_template: str,
        history: str,
        score: Optional[float] = None
    ) -> "PromptSearchResult":
        """Create from file storage data."""
        last_updated = metadata.get('last_updated')
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
        elif not isinstance(last_updated, datetime):
            last_updated = datetime.now(timezone.utc)

        return cls(
            id=prompt_id,
            use_case=metadata.get('use_case', 'general'),
            title=metadata.get('title', ''),
            summary=metadata.get('summary', ''),
            prompt_template=prompt_template,
            history=history,
            last_updated=last_updated,
            num_updates=metadata.get('num_updates', 0),
            score=score
        )


class ConversationHistory(BaseModel):
    """Conversation history."""

    messages: List[Dict[str, Any]]
    context: Optional[str] = None
    task_description: Optional[str] = None


class PromptUpdate(BaseModel):
    """Prompt update data."""

    prompt_id: str
    summary: Optional[str] = None
    prompt_template: Optional[str] = None
    history: Optional[str] = None
    use_case: Optional[str] = None
    change_description: str
