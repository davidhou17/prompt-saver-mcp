"""Data models."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
