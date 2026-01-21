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
    summary: str
    prompt_template: str
    history: str
    last_updated: datetime
    num_updates: int
    score: Optional[float] = None


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
