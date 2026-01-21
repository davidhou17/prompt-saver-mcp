"""Prompt Saver MCP Server.

An MCP server that summarizes and saves conversation threads as prompts,
categorizes them by use case, and stores them locally as markdown files
or optionally in MongoDB. Supports text-based search and optional semantic
search with vector embeddings.
"""

from .models import UseCase

__version__ = "2.0.0"
__all__ = ["UseCase", "__version__"]
