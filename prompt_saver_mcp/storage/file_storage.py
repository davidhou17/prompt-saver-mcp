"""File-based storage backend for prompts.

Stores prompts in a directory structure with separate files for the prompt template
and metadata/changelog.
"""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..models import PromptTemplate, PromptSearchResult
from .base import StorageManager


class FileStorageManager(StorageManager):
    """Store prompts as local markdown files.

    File structure:
        {prompts_path}/
            {use_case}/
                {prompt-name}/
                    prompt.md       # The prompt template (clean, no frontmatter)
                    changelog.md    # ID, history, and changelog with YAML frontmatter

    Directory names are slugified from the prompt summary. If a name collision occurs,
    a suffix (-2, -3, etc.) is appended.
    """

    def __init__(self, prompts_path: str):
        """Initialize file storage.

        Args:
            prompts_path: Path to the directory where prompts will be stored
        """
        self.prompts_path = Path(prompts_path).expanduser().resolve()
        self._index: Dict[str, Path] = {}  # id -> prompt directory mapping

    async def connect(self) -> None:
        """Create the prompts directory if it doesn't exist and build index."""
        self.prompts_path.mkdir(parents=True, exist_ok=True)
        await self._build_index()

    async def disconnect(self) -> None:
        """No cleanup needed for file storage."""
        pass

    async def _build_index(self) -> None:
        """Build an in-memory index of prompt IDs to prompt directories."""
        self._index.clear()

        if not self.prompts_path.exists():
            return

        # Look for prompt.md files in subdirectories
        for prompt_file in self.prompts_path.rglob("prompt.md"):
            try:
                prompt_dir = prompt_file.parent
                changelog_file = prompt_dir / "changelog.md"
                if changelog_file.exists():
                    metadata = self._read_changelog_metadata(changelog_file)
                    if metadata and "id" in metadata:
                        self._index[metadata["id"]] = prompt_dir
            except Exception:
                continue

    def _slugify(self, text: str, max_length: int = 50) -> str:
        """Convert text to a URL-friendly slug."""
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug[:max_length].rstrip('-') or "untitled"

    def _read_changelog_metadata(self, changelog_path: Path) -> Optional[Dict[str, Any]]:
        """Read YAML frontmatter from a changelog file."""
        try:
            content = changelog_path.read_text(encoding='utf-8')
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    return yaml.safe_load(parts[1])
        except Exception:
            pass
        return None

    def _read_prompt_from_dir(self, prompt_dir: Path) -> Optional[PromptSearchResult]:
        """Read a prompt from its directory structure."""
        try:
            prompt_file = prompt_dir / "prompt.md"
            changelog_file = prompt_dir / "changelog.md"

            if not prompt_file.exists() or not changelog_file.exists():
                return None

            # Read prompt template
            prompt_template = prompt_file.read_text(encoding='utf-8').strip()

            # Read changelog with metadata
            changelog_content = changelog_file.read_text(encoding='utf-8')
            if not changelog_content.startswith('---'):
                return None

            parts = changelog_content.split('---', 2)
            if len(parts) < 3:
                return None

            metadata = yaml.safe_load(parts[1])
            history = parts[2].strip()

            return PromptSearchResult.from_file_data(
                prompt_id=metadata.get('id', ''),
                metadata=metadata,
                prompt_template=prompt_template,
                history=history
            )
        except Exception:
            return None

    def _write_prompt_to_dir(self, prompt_dir: Path, prompt_id: str, prompt: PromptTemplate) -> None:
        """Write a prompt to its directory structure."""
        # Create directory structure
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Write prompt.md (just the template, no frontmatter)
        prompt_file = prompt_dir / "prompt.md"
        prompt_file.write_text(prompt.prompt_template, encoding='utf-8')

        # Write changelog.md with frontmatter
        changelog_file = prompt_dir / "changelog.md"
        frontmatter = {
            'id': prompt_id,
            'use_case': prompt.use_case,
            'title': prompt.title,
            'summary': prompt.summary,
            'created': datetime.now(timezone.utc).isoformat(),
            'last_updated': prompt.last_updated.isoformat(),
            'num_updates': prompt.num_updates,
            'changelog': prompt.changelog
        }

        changelog_content = f"""---
{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()}
---

# History

{prompt.history}
"""
        changelog_file.write_text(changelog_content, encoding='utf-8')

    async def save_prompt(self, prompt: PromptTemplate) -> str:
        """Save a new prompt to its own directory."""
        prompt_id = str(uuid.uuid4())
        slug = self._slugify(prompt.title)

        # Create use-case subdirectory with prompt directory
        use_case_dir = self.prompts_path / prompt.use_case
        prompt_dir = use_case_dir / slug

        # Handle name collision by appending a number
        if prompt_dir.exists():
            counter = 2
            while (use_case_dir / f"{slug}-{counter}").exists():
                counter += 1
            prompt_dir = use_case_dir / f"{slug}-{counter}"

        self._write_prompt_to_dir(prompt_dir, prompt_id, prompt)
        self._index[prompt_id] = prompt_dir

        return prompt_id

    async def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptSearchResult]:
        """Get a specific prompt by its ID."""
        prompt_dir = self._index.get(prompt_id)
        if not prompt_dir or not prompt_dir.exists():
            # Try rebuilding index in case of new files
            await self._build_index()
            prompt_dir = self._index.get(prompt_id)
            if not prompt_dir or not prompt_dir.exists():
                return None

        return self._read_prompt_from_dir(prompt_dir)

    async def search_prompts_by_text(self, query: str, limit: int = 3) -> List[PromptSearchResult]:
        """Search prompts using case-insensitive text matching."""
        results: List[tuple[float, PromptSearchResult]] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Look for prompt.md files in prompt directories
        for prompt_file in self.prompts_path.rglob("prompt.md"):
            prompt_dir = prompt_file.parent
            prompt = self._read_prompt_from_dir(prompt_dir)
            if not prompt:
                continue

            # Calculate relevance score based on word matches
            searchable = f"{prompt.summary} {prompt.prompt_template} {prompt.history}".lower()

            # Check if query appears as substring
            if query_lower in searchable:
                score = 1.0
            else:
                # Count matching words
                matching_words = sum(1 for word in query_words if word in searchable)
                score = matching_words / len(query_words) if query_words else 0

            if score > 0:
                prompt.score = score
                results.append((score, prompt))

        # Sort by score descending, then by last_updated
        results.sort(key=lambda x: (-x[0], -x[1].last_updated.timestamp()))

        return [r[1] for r in results[:limit]]

    async def search_prompts_by_use_case(self, use_case: str, limit: int = 5) -> List[PromptSearchResult]:
        """Search prompts by use case category."""
        results: List[PromptSearchResult] = []
        use_case_dir = self.prompts_path / use_case

        if use_case_dir.exists():
            # Look for prompt directories (those containing prompt.md)
            for prompt_file in use_case_dir.glob("*/prompt.md"):
                prompt_dir = prompt_file.parent
                prompt = self._read_prompt_from_dir(prompt_dir)
                if prompt:
                    results.append(prompt)

        # Sort by last_updated descending
        results.sort(key=lambda x: x.last_updated, reverse=True)

        return results[:limit]

    async def update_prompt(self, prompt_id: str, updates: Dict[str, Any], change_description: str) -> bool:
        """Update an existing prompt."""
        prompt_dir = self._index.get(prompt_id)
        if not prompt_dir or not prompt_dir.exists():
            return False

        try:
            current = self._read_prompt_from_dir(prompt_dir)
            if not current:
                return False

            prompt_file = prompt_dir / "prompt.md"
            changelog_file = prompt_dir / "changelog.md"

            # Read current changelog metadata
            changelog_content = changelog_file.read_text(encoding='utf-8')
            parts = changelog_content.split('---', 2)
            if len(parts) < 3:
                return False

            metadata = yaml.safe_load(parts[1])

            # Apply updates
            now = datetime.now(timezone.utc)
            metadata['last_updated'] = now.isoformat()
            metadata['num_updates'] = metadata.get('num_updates', 0) + 1

            changelog = metadata.get('changelog', [])
            changelog.append(f"{now.isoformat()}: {change_description}")
            metadata['changelog'] = changelog

            if 'summary' in updates:
                metadata['summary'] = updates['summary']
            if 'use_case' in updates:
                metadata['use_case'] = updates['use_case']

            # Update history
            history = updates.get('history', current.history)

            # Write updated changelog
            new_changelog_content = f"""---
{yaml.dump(metadata, default_flow_style=False, allow_unicode=True).strip()}
---

# History

{history}
"""
            changelog_file.write_text(new_changelog_content, encoding='utf-8')

            # Update prompt template if provided
            if 'prompt_template' in updates:
                prompt_file.write_text(updates['prompt_template'], encoding='utf-8')

            return True

        except Exception:
            return False

