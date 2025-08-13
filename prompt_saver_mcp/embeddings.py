"""Simple vector embeddings using Voyage AI."""

import os
from typing import List
import voyageai


class EmbeddingManager:
    """Simple wrapper for Voyage AI embeddings."""

    def __init__(self, api_key: str, model: str = "voyage-3-large"):
        os.environ["VOYAGE_API_KEY"] = api_key
        self.client = voyageai.Client()
        self.model = model

    def embed(self, text: str, input_type: str = "document") -> List[float]:
        """Generate embedding for text."""
        return self.client.embed(
            [text.replace("\n", " ").strip()],
            model=self.model,
            input_type=input_type,
            output_dimension=2048
        ).embeddings[0]
