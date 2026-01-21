# Prompt Saver MCP Server

MCP server that converts successful conversation threads into prompts that can be used for future tasks.

Based on the principle that the most important artifact of your LLM interactions is what you did to produce the results, not the results themselves (see [The New Code](https://www.youtube.com/watch?v=8rABwKRsec4)). And also considering that LLMs are probably already better prompt engineers than humans.

https://github.com/user-attachments/assets/d2e90767-c6f2-44b7-a216-1d9e103e968a

## Quick Start

**Minimal setup - just needs an LLM API key:**

```bash
# Clone and install
git clone <repository-url>
cd prompt-saver-mcp
uv sync

# Set your LLM API key (choose one)
export OPENAI_API_KEY="your-key"          # For OpenAI
# or
export ANTHROPIC_API_KEY="your-key"       # For Anthropic
# or
export AZURE_OPENAI_API_KEY="your-key"    # For Azure OpenAI
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

Prompts are stored as readable markdown files in `~/.prompt-saver/prompts/` by default.

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

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd prompt-saver-mcp
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   # or with pip:
   pip install -e .
   ```

3. **Set your LLM API key** (choose one provider):
   ```bash
   # OpenAI (default)
   export OPENAI_API_KEY="your-key"

   # Or Anthropic
   export LLM_PROVIDER="anthropic"
   export ANTHROPIC_API_KEY="your-key"

   # Or Azure OpenAI
   export LLM_PROVIDER="azure_openai"
   export AZURE_OPENAI_API_KEY="your-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   ```

That's it! Prompts are stored as markdown files in `~/.prompt-saver/prompts/` by default.

## Configuration

### Environment Variables

#### Storage Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `STORAGE_TYPE` | Storage backend: `file` or `mongodb` | `file` |
| `PROMPTS_PATH` | Directory for prompt files (when STORAGE_TYPE=file) | `~/.prompt-saver/prompts` |

#### LLM Provider Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider: `openai`, `azure_openai`, or `anthropic` | `openai` |

**Provider Options:**
- **OpenAI**: Direct access to latest models with simple API key setup (default)
- **Azure OpenAI**: Enterprise-grade with enhanced security and compliance
- **Anthropic**: Claude models with strong reasoning capabilities

#### OpenAI (default, when LLM_PROVIDER=openai)
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | Model name | `gpt-4o` |

#### Azure OpenAI (when LLM_PROVIDER=azure_openai)
| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Required |
| `AZURE_OPENAI_MODEL` | Model deployment name | `gpt-4o` |

#### Anthropic (when LLM_PROVIDER=anthropic)
| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `ANTHROPIC_MODEL` | Model name | `claude-sonnet-4-20250514` |

#### MongoDB Storage (optional, when STORAGE_TYPE=mongodb)
| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection URI | Required |
| `MONGODB_DATABASE` | Database name | `prompt_saver` |
| `MONGODB_COLLECTION` | Collection name | `prompts` |
| `VECTOR_INDEX_NAME` | Atlas vector search index name | `vector_index` |

#### Embeddings (optional)
| Variable | Description | Default |
|----------|-------------|---------|
| `VOYAGE_AI_API_KEY` | Voyage AI API key (enables semantic search) | Optional |

### Claude Desktop Integration

Add to your Claude Desktop configuration file:

#### Simple Configuration (File Storage + OpenAI)
```json
{
  "mcpServers": {
    "prompt-saver": {
      "command": "uv",
      "args": ["run", "python", "-m", "prompt_saver_mcp.server", "stdio"],
      "cwd": "/path/to/your/prompt-saver-mcp",
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    }
  }
}
```

#### With Custom Prompts Directory
```json
{
  "mcpServers": {
    "prompt-saver": {
      "command": "uv",
      "args": ["run", "python", "-m", "prompt_saver_mcp.server", "stdio"],
      "cwd": "/path/to/your/prompt-saver-mcp",
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "PROMPTS_PATH": "/path/to/your/prompts"
      }
    }
  }
}
```

#### With Anthropic
```json
{
  "mcpServers": {
    "prompt-saver": {
      "command": "uv",
      "args": ["run", "python", "-m", "prompt_saver_mcp.server", "stdio"],
      "cwd": "/path/to/your/prompt-saver-mcp",
      "env": {
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "your_anthropic_api_key_here"
      }
    }
  }
}
```

#### Advanced: MongoDB Storage + Semantic Search
```json
{
  "mcpServers": {
    "prompt-saver": {
      "command": "uv",
      "args": ["run", "python", "-m", "prompt_saver_mcp.server", "stdio"],
      "cwd": "/path/to/your/prompt-saver-mcp",
      "env": {
        "STORAGE_TYPE": "mongodb",
        "MONGODB_URI": "mongodb+srv://username:password@cluster.mongodb.net/",
        "VOYAGE_AI_API_KEY": "your_voyage_ai_api_key_here",
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    }
  }
}
```

> **Note**: For MongoDB storage, install additional dependencies: `pip install motor pymongo`

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

Prompts are stored as markdown files with YAML frontmatter:

```
~/.prompt-saver/prompts/
├── code-gen/
│   ├── python-csv-parser-abc12345.md
│   └── react-component-generator-def67890.md
├── data-analysis/
│   └── pandas-data-cleaning-ghi11111.md
└── general/
    └── project-planning-template-jkl22222.md
```

Each file has this structure:

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

# Prompt Template

You are an expert Python developer...
```

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
