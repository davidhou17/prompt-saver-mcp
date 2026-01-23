# Prompt Saver MCP Server

MCP server that converts successful conversation threads into prompts that can be used for future tasks.

Based on the principle that the most important artifact of your LLM interactions is what you did to produce the results, not the results themselves (see [The New Code](https://www.youtube.com/watch?v=8rABwKRsec4)). And also considering that LLMs are probably already better prompt engineers than humans.

https://github.com/user-attachments/assets/d2e90767-c6f2-44b7-a216-1d9e103e968a

## Quick Start

```bash
git clone https://github.com/davidhou17/prompt-saver-mcp.git
cd prompt-saver-mcp
uv sync
```

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "prompt-saver": {
      "command": "uv",
      "args": ["run", "python", "-m", "prompt_saver_mcp.server", "stdio"],
      "cwd": "/path/to/prompt-saver-mcp",
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

Prompts are stored in the `prompts/` directory within the MCP server folder by default.

## Tools

### `save_prompt`
Summarizes, categorizes, and converts conversation history into a markdown formatted prompt template.

Run upon completion of a successful complex task to build your prompt library.

**Parameters:**
- `conversation_messages` (string): JSON string containing the conversation history
- `task_description` (optional string): Description of the task being performed
- `context_info` (optional string): Additional context about the conversation

### `search_prompts`
Retrieves prompts using text search (or semantic search if Voyage API key is available). Returns the most relevant prompts for user selection.

**Parameters:**
- `query` (string): Description of the problem or task you need help with
- `limit` (optional int): Maximum number of prompts to return (default: 3)

### `improve_prompt_from_feedback`
Summarizes feedback during the conversation and updates the prompt based on the feedback and conversation context.

**Parameters:**
- `prompt_id` (string): ID of the prompt to improve
- `feedback` (string): User feedback about the prompt's effectiveness
- `conversation_context` (optional string): Context from the conversation where the prompt was used

### `update_prompt`
For manual updates to an existing prompt.

**Parameters:**
- `prompt_id` (string): ID of the prompt to update
- `change_description` (string): Description of what was changed and why
- `summary` (optional string): Updated summary
- `prompt_template` (optional string): Updated prompt template
- `history` (optional string): Updated history
- `use_case` (optional string): Updated use case

### `get_prompt_details`
Get detailed information about a specific prompt including its full template, history, and metadata. Used to view the complete prompt before applying it.

**Parameters:**
- `prompt_id` (string): ID of the prompt to retrieve

### `search_prompts_by_use_case`
Search for prompts by their use case category (e.g., 'code-gen', 'text-gen', 'data-analysis'). Efficient category-based search that works independently of embedding services.

**Parameters:**
- `use_case` (string): The use case category to search for
- `limit` (optional int): Maximum number of results to return (default: 5)

## Prompt Engineering Best Practices

The generated prompt templates use the following prompt engineering techniques:

### Structure
Templates are organized in this order:
1. **Identity**: Defines the assistant's persona and goals
2. **Instructions**: Provides clear rules and constraints
3. **Examples**: Shows desired input/output patterns (few-shot learning)
4. **Context**: Adds relevant data and documents

### Formatting
- **Markdown headers** (#) and lists (*) for logical hierarchy
- **XML tags** (`<example>`) to separate content sections
- **Message roles** (developer/user/assistant) where appropriate
- **Placeholders** (`{variable}`) for customizable inputs

This ensures the saved prompts are well-structured, reusable, and effective for future tasks.

## System Prompt Configuration

To maximize the value of this MCP server, add the following instructions to your LLM interface's system prompt:

```
Always search for relevant prompts before starting any large or complex tasks.

Upon successful completion of a task, always ask if I want to save the conversation as a prompt.

Upon successful completion of a task with a prompt, always ask if I want to update the prompt.

```

This helps ensure that the LLM runs the relevant tools without you explicitly asking.

## Configuration

All configuration is done via environment variables in the `env` block of your MCP config.

### LLM Providers

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai`, `azure_openai`, or `anthropic` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `OPENAI_MODEL` | Model name | `gpt-4o` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Anthropic |
| `ANTHROPIC_MODEL` | Model name | `claude-sonnet-4-20250514` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required for Azure |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Required for Azure |
| `AZURE_OPENAI_MODEL` | Model deployment name | `gpt-4o` |

### Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `STORAGE_TYPE` | `file` or `mongodb` | `file` |
| `PROMPTS_PATH` | Directory for prompt files | `./prompts` (relative to MCP server) |
| `MONGODB_URI` | MongoDB connection URI | Required for MongoDB |
| `MONGODB_DATABASE` | Database name | `prompt_saver` |

### Optional

| Variable | Description |
|----------|-------------|
| `VOYAGE_AI_API_KEY` | Enables semantic search |

> **Note**: For MongoDB storage, install with `uv sync --extra mongodb`

## Usage Examples

### 1. Saving a Prompt

After completing a complex task, save the conversation as a reusable prompt:

```python
# Example conversation messages (JSON format)
conversation = [
    {"role": "user", "content": "Help me create a Python function to parse CSV files"},
    {"role": "assistant", "content": "I'll help you create a robust CSV parser..."},
    # ... more conversation
]

# Save the prompt
save_prompt(
    conversation_messages=json.dumps(conversation),
    task_description="Creating a CSV parser function",
    context_info="Successfully created a parser with error handling"
)
```

### 2. Finding and Using a Prompt

Search for relevant prompts when starting a new task:

```python
# Search for prompts
result = search_prompts("I need help with data processing in Python")

# The tool will return the most relevant prompts and ask you to select one
```

### 3. Updating a Prompt

After using a prompt, update it based on your experience:

```python
update_prompt(
    prompt_id="prompt_id_here",
    change_description="Added error handling examples",
    prompt_template="Updated template with better error handling..."
)
```

### 4. Getting Prompt Details

Retrieve the full details of a specific prompt:

```python
# Get complete prompt information
result = get_prompt_details("prompt_id_here")
print(result["prompt"]["prompt_template"])  # View the full template
```

### 5. Improving a Prompt with Feedback

Use AI to automatically improve a prompt based on feedback:

```python
improve_prompt_from_feedback(
    prompt_id="prompt_id_here",
    feedback="The prompt worked well but could use more specific examples for edge cases",
    conversation_context="Used for debugging a complex API integration issue"
)
```

### 6. Searching by Use Case

Find prompts for specific types of tasks:

```python
# Find all code generation prompts
result = search_prompts_by_use_case("code-gen", limit=5)

# Find data analysis prompts
result = search_prompts_by_use_case("data-analysis", limit=3)
```

## Prompt Storage Format

### File Storage (Default)

Prompts are stored in the `prompts/` directory within the MCP server directory. Each prompt gets its own folder named after the prompt summary (slugified). The `prompts/` directory is gitignored by default to keep your personal prompts private.

**Directory Structure:**
```
prompts/
└── {use-case}/
    └── {prompt-name}/
        ├── prompt.md       # The prompt template
        └── changelog.md    # ID, history, and changelog
```

**Example:**
```
prompts/
├── code-gen/
│   ├── python-csv-parser/
│   │   ├── prompt.md
│   │   └── changelog.md
│   └── react-component-generator/
│       ├── prompt.md
│       └── changelog.md
└── general/
    └── project-planning-template/
        ├── prompt.md
        └── changelog.md
```

**File Contents:**

`prompt.md` — Clean prompt template, ready to use:
```markdown
You are an expert Python developer...
```

`changelog.md` — Metadata and history with YAML frontmatter:
```markdown
---
id: abc12345-1234-5678-9abc-def012345678
use_case: code-gen
summary: Python CSV parser with error handling
created: 2025-01-21T10:00:00+00:00
last_updated: 2025-01-21T10:00:00+00:00
num_updates: 0
changelog:
  - '2025-01-21T10:00:00+00:00: Initial creation'
---

# History

Steps taken to create this prompt...
```

> **Note:** If two prompts have the same name, a suffix (`-2`, `-3`, etc.) is appended to the folder name.

### MongoDB Storage (Optional)

When using MongoDB (`STORAGE_TYPE=mongodb`), prompts are stored as documents:

```python
{
    "_id": ObjectId,
    "use_case": str,  # "code-gen", "text-gen", "data-analysis", "creative", "general"
    "summary": str,   # Summary of the prompt and its use case
    "prompt_template": str,  # Universal problem-solving prompt template
    "history": str,   # Summary of steps taken and end result
    "embedding": List[float],  # Vector embeddings (optional, for semantic search)
    "last_updated": datetime,
    "num_updates": int,
    "changelog": List[str]  # List of changes made to this prompt
}
```

## License

MIT License - see LICENSE file for details.
