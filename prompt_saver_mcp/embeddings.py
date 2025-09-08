"""Simple vector embeddings using Voyage AI."""

import os
from typing import List, Optional
import voyageai


class EmbeddingManager:
    """Simple wrapper for Voyage AI embeddings."""

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3-large"):
        self.api_key = api_key
        self.model = model
        self.client = None

        if api_key:
            os.environ["VOYAGE_API_KEY"] = api_key
            self.client = voyageai.Client()

    def embed(self, text: str, input_type: str = "document") -> Optional[List[float]]:
        """Generate embedding for text. Returns None if no API key is provided."""
        if not self.client:
            return None

        return self.client.embed(
            [text.replace("\n", " ").strip()],
            model=self.model,
            input_type=input_type,
            output_dimension=2048
        ).embeddings[0]

    def is_available(self) -> bool:
        """Check if embedding functionality is available."""
        return self.client is not None
