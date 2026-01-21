"""File-based storage backend for prompts.

Stores prompts as markdown files with YAML frontmatter.
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
                {slug}-{uuid}.md
    
    Each file contains YAML frontmatter with metadata and the prompt content.
    """

    def __init__(self, prompts_path: str):
        """Initialize file storage.
        
        Args:
            prompts_path: Path to the directory where prompts will be stored
        """
        self.prompts_path = Path(prompts_path).expanduser().resolve()
        self._index: Dict[str, Path] = {}  # id -> file path mapping

    async def connect(self) -> None:
        """Create the prompts directory if it doesn't exist and build index."""
        self.prompts_path.mkdir(parents=True, exist_ok=True)
        await self._build_index()

    async def disconnect(self) -> None:
        """No cleanup needed for file storage."""
        pass

    async def _build_index(self) -> None:
        """Build an in-memory index of prompt IDs to file paths."""
        self._index.clear()
        
        if not self.prompts_path.exists():
            return
            
        for md_file in self.prompts_path.rglob("*.md"):
            try:
                metadata = self._read_frontmatter(md_file)
                if metadata and "id" in metadata:
                    self._index[metadata["id"]] = md_file
            except Exception:
                continue

    def _slugify(self, text: str, max_length: int = 50) -> str:
        """Convert text to a URL-friendly slug."""
        # Convert to lowercase and replace spaces with hyphens
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug[:max_length].rstrip('-')

    def _read_frontmatter(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read YAML frontmatter from a markdown file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    return yaml.safe_load(parts[1])
        except Exception:
            pass
        return None

    def _read_prompt_file(self, file_path: Path) -> Optional[PromptSearchResult]:
        """Read a prompt file and return a PromptSearchResult."""
        try:
            content = file_path.read_text(encoding='utf-8')
            if not content.startswith('---'):
                return None
                
            parts = content.split('---', 2)
            if len(parts) < 3:
                return None
                
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            
            # Parse body sections
            history = ""
            prompt_template = ""
            
            sections = re.split(r'^# ', body, flags=re.MULTILINE)
            for section in sections:
                if section.startswith('History'):
                    history = section.replace('History\n', '', 1).strip()
                elif section.startswith('Prompt Template'):
                    prompt_template = section.replace('Prompt Template\n', '', 1).strip()
            
            # Parse datetime
            last_updated = metadata.get('last_updated')
            if isinstance(last_updated, str):
                last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            elif not isinstance(last_updated, datetime):
                last_updated = datetime.now(timezone.utc)

            return PromptSearchResult(
                id=metadata.get('id', ''),
                use_case=metadata.get('use_case', 'general'),
                summary=metadata.get('summary', ''),
                prompt_template=prompt_template,
                history=history,
                last_updated=last_updated,
                num_updates=metadata.get('num_updates', 0)
            )
        except Exception:
            return None

    def _write_prompt_file(self, file_path: Path, prompt_id: str, prompt: PromptTemplate) -> None:
        """Write a prompt to a markdown file with YAML frontmatter."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare frontmatter
        frontmatter = {
            'id': prompt_id,
            'use_case': prompt.use_case,
            'summary': prompt.summary,
            'created': datetime.now(timezone.utc).isoformat(),
            'last_updated': prompt.last_updated.isoformat(),
            'num_updates': prompt.num_updates,
            'changelog': prompt.changelog
        }

        # Build file content
        content = f"""---
{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()}
---

# History

{prompt.history}

# Prompt Template

{prompt.prompt_template}
"""
        file_path.write_text(content, encoding='utf-8')

    async def save_prompt(self, prompt: PromptTemplate) -> str:
        """Save a new prompt as a markdown file."""
        prompt_id = str(uuid.uuid4())
        slug = self._slugify(prompt.summary)
        filename = f"{slug}-{prompt_id[:8]}.md"

        # Create use-case subdirectory
        use_case_dir = self.prompts_path / prompt.use_case
        file_path = use_case_dir / filename

        self._write_prompt_file(file_path, prompt_id, prompt)
        self._index[prompt_id] = file_path

        return prompt_id

    async def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptSearchResult]:
        """Get a specific prompt by its ID."""
        file_path = self._index.get(prompt_id)
        if not file_path or not file_path.exists():
            # Try rebuilding index in case of new files
            await self._build_index()
            file_path = self._index.get(prompt_id)
            if not file_path or not file_path.exists():
                return None

        return self._read_prompt_file(file_path)

    async def search_prompts_by_text(self, query: str, limit: int = 3) -> List[PromptSearchResult]:
        """Search prompts using case-insensitive text matching."""
        results: List[tuple[float, PromptSearchResult]] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for md_file in self.prompts_path.rglob("*.md"):
            prompt = self._read_prompt_file(md_file)
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
            for md_file in use_case_dir.glob("*.md"):
                prompt = self._read_prompt_file(md_file)
                if prompt:
                    results.append(prompt)

        # Sort by last_updated descending
        results.sort(key=lambda x: x.last_updated, reverse=True)

        return results[:limit]

    async def update_prompt(self, prompt_id: str, updates: Dict[str, Any], change_description: str) -> bool:
        """Update an existing prompt."""
        file_path = self._index.get(prompt_id)
        if not file_path or not file_path.exists():
            return False

        try:
            current = self._read_prompt_file(file_path)
            if not current:
                return False

            # Read current frontmatter
            content = file_path.read_text(encoding='utf-8')
            parts = content.split('---', 2)
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

            # Rebuild body
            history = updates.get('history', current.history)
            prompt_template = updates.get('prompt_template', current.prompt_template)

            new_content = f"""---
{yaml.dump(metadata, default_flow_style=False, allow_unicode=True).strip()}
---

# History

{history}

# Prompt Template

{prompt_template}
"""
            file_path.write_text(new_content, encoding='utf-8')
            return True

        except Exception:
            return False

