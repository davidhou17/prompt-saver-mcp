"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import PromptTemplate, PromptSearchResult


class StorageManager(ABC):
    """Abstract base class for prompt storage backends.
    
    Implementations must provide methods for saving, retrieving, 
    searching, and updating prompts.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the storage connection/setup."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up storage resources."""
        pass

    @abstractmethod
    async def save_prompt(self, prompt: PromptTemplate) -> str:
        """Save a new prompt.
        
        Args:
            prompt: The prompt template to save
            
        Returns:
            The unique ID of the saved prompt
        """
        pass

    @abstractmethod
    async def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptSearchResult]:
        """Get a specific prompt by its ID.
        
        Args:
            prompt_id: The unique identifier of the prompt
            
        Returns:
            The prompt if found, None otherwise
        """
        pass

    @abstractmethod
    async def search_prompts_by_text(self, query: str, limit: int = 3) -> List[PromptSearchResult]:
        """Search prompts using text matching.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching prompts
        """
        pass

    @abstractmethod
    async def search_prompts_by_use_case(self, use_case: str, limit: int = 5) -> List[PromptSearchResult]:
        """Search prompts by use case category.
        
        Args:
            use_case: The use case category to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching prompts sorted by last_updated
        """
        pass

    @abstractmethod
    async def update_prompt(self, prompt_id: str, updates: Dict[str, Any], change_description: str) -> bool:
        """Update an existing prompt.
        
        Args:
            prompt_id: The unique identifier of the prompt
            updates: Dictionary of fields to update
            change_description: Description of the changes made
            
        Returns:
            True if the update was successful, False otherwise
        """
        pass

    async def search_prompts_by_vector(self, embedding: List[float], limit: int = 3) -> List[PromptSearchResult]:
        """Search prompts using vector similarity.
        
        This is optional - implementations that don't support vector search
        should raise NotImplementedError or return an empty list.
        
        Args:
            embedding: Vector embedding to search with
            limit: Maximum number of results to return
            
        Returns:
            List of matching prompts sorted by similarity
        """
        raise NotImplementedError("Vector search not supported by this storage backend")

