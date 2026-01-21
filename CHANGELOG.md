# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-21

### 🚀 Major Changes

- **File-based storage by default** - No database required! Prompts are now stored as readable markdown files in `~/.prompt-saver/prompts/` with YAML frontmatter. MongoDB is now optional.
- **Simplified setup** - Just needs an LLM API key to get started (OpenAI, Anthropic, or Azure OpenAI)

### Added

- **New storage abstraction layer** (`prompt_saver_mcp/storage/`)
  - `StorageManager` abstract base class for pluggable storage backends
  - `FileStorageManager` - stores prompts as markdown files organized by use case
  - `MongoDBStorageManager` - optional MongoDB backend (requires `pip install motor pymongo`)
- **`UseCase` enum** - Type-safe use case categories (`code-gen`, `text-gen`, `data-analysis`, `creative`, `general`)
- **`py.typed` marker** - Package now supports downstream type checking
- **MIT LICENSE file** - Proper open source licensing
- **`test_file_storage.py`** - Standalone test for file storage (no external dependencies)

### Changed

- **Default LLM provider changed from Azure OpenAI to OpenAI** - Simpler default for most users
- **Voyage AI embeddings now optional** - Semantic search only enabled when `VOYAGE_AI_API_KEY` is set; falls back to text search
- **`run-mcp-server.sh` now portable** - Dynamically finds `uv` instead of hardcoded path
- **Improved configuration examples** in `claude_desktop_config.json.example`
- **Updated documentation** with clearer quick start and configuration tables

### Removed

- **`database.py`** - Replaced by the new storage abstraction layer

### Fixed

- Hardcoded user path in `run-mcp-server.sh`
- Missing LICENSE file (referenced in pyproject.toml but didn't exist)

## [1.0.0] - 2024-10-27

### Added

- Initial release
- MongoDB-based prompt storage with vector search
- Support for OpenAI, Azure OpenAI, and Anthropic LLM providers
- Voyage AI embeddings for semantic search
- MCP tools: `save_prompt`, `search_prompts`, `get_prompt_details`, `update_prompt`, `improve_prompt_from_feedback`, `search_prompts_by_use_case`

