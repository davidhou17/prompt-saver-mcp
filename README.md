# Prompt Saver MCP Server

MCP server that converts successful conversation threads into prompts that can be used for future tasks. 

The most important artifact of your LLM interactions is not the generated results, rather the steps taken to produce said results.
Inspired by [The New Code](https://www.youtube.com/watch?v=8rABwKRsec4). 

## Tools

### `save_prompt`
Summarizes, categorizes, and converts conversation history into a markdown formatted prompt template. 
Run upon completion of a successful complex task.

**Parameters:**
- `conversation_messages` (string): JSON string containing the conversation history
- `task_description` (optional string): Description of the task being performed
- `context_info` (optional string): Additional context about the conversation

### `use_prompt`
Retrieves prompts from the database using vector search and returns the 3 most relevant prompts for user selection.

**Parameters:**
- `query` (string): Description of the problem or task you need help with
- `limit` (optional int): Maximum number of prompts to return (default: 3)

### `update_prompt`
Updates an existing prompt based on learnings from conversation. Run after using a prompt to improve it.

**Parameters:**
- `prompt_id` (string): ID of the prompt to update
- `change_description` (string): Description of what was changed and why
- `summary` (optional string): Updated summary
- `prompt_template` (optional string): Updated prompt template
- `history` (optional string): Updated history
- `use_case` (optional string): Updated use case

### `get_prompt_details`
Get detailed information about a specific prompt including its full template, history, and metadata.

**Parameters:**
- `prompt_id` (string): ID of the prompt to retrieve

### `improve_prompt_from_feedback`
Improve an existing prompt based on user feedback and conversation context. This tool uses AI to analyze feedback and automatically enhance the prompt.

**Parameters:**
- `prompt_id` (string): ID of the prompt to improve
- `feedback` (string): User feedback about the prompt's effectiveness
- `conversation_context` (optional string): Context from the conversation where the prompt was used

### `search_prompts_by_use_case`
Search for prompts by their use case category (e.g., 'code-gen', 'text-gen', 'data-analysis').

**Parameters:**
- `use_case` (string): The use case category to search for
- `limit` (optional int): Maximum number of results to return (default: 5)

## Prompt Engineering Best Practices

The generated prompt templates follow proven prompt engineering techniques:

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

3. **Set up MongoDB Atlas:**
   - Create a MongoDB Atlas cluster
   - Create a database named `prompt_saver`
   - Create a vector search index on the `embedding` field (2048 dimensions, dotProduct similarity)

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and MongoDB Atlas URI
   ```

   Required:
   - `MONGODB_URI`: MongoDB Atlas connection string
   - `VOYAGE_AI_API_KEY`: Voyage AI API key for embeddings
   - `AZURE_OPENAI_API_KEY`: Azure OpenAI API key for LLM operations
   - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB Atlas connection URI | Required |
| `MONGODB_DATABASE` | Database name | `prompt_saver` |
| `MONGODB_COLLECTION` | Collection name | `prompts` |
| `VOYAGE_AI_API_KEY` | Voyage AI API key | Required |
| `VOYAGE_AI_EMBEDDING_MODEL` | Embedding model | `voyage-3-large` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Required |
| `AZURE_OPENAI_MODEL` | Model deployment name | `gpt-4o` |

### Claude Desktop Integration

Add to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "prompt-saver": {
      "command": "uv",
      "args": ["run", "python", "-m", "prompt_saver_mcp.server", "stdio"],
      "cwd": "/path/to/your/prompt-saver-mcp",
      "env": {
        "MONGODB_URI": "mongodb+srv://username:password@cluster.mongodb.net/",
        "MONGODB_DATABASE": "prompt_saver",
        "VOYAGE_AI_API_KEY": "your_voyage_ai_api_key_here",
        "AZURE_OPENAI_API_KEY": "your_azure_openai_api_key_here",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/"
      }
    }
  }
}
```

### System Prompt Configuration

To maximize the value of your saved prompts, add this instruction to your LLM interface's system prompt:

```
Always search for relevant prompts before starting any large or complex tasks.
```

This ensures that the LLM checks and uses your existing prompts before starting a new task.

> **💡 Tip**: For enhanced MongoDB management, consider using the [MongoDB MCP Server](https://github.com/mongodb-js/mongodb-mcp-server) alongside this prompt saver. It provides direct MongoDB operations and can help you manage your prompt database more effectively.

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
result = use_prompt("I need help with data processing in Python")

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

## Database Schema

Each prompt is stored with the following structure:

```python
{
    "_id": ObjectId,
    "use_case": str,  # "code-gen", "text-gen", "data-analysis", "creative", "general"
    "summary": str,   # Summary of the prompt and its use case
    "prompt_template": str,  # Universal problem-solving prompt template
    "history": str,   # Summary of steps taken and end result
    "embedding": List[float],  # Vector embeddings of the summary
    "last_updated": datetime,
    "num_updates": int,
    "changelog": List[str]  # List of changes made to this prompt
}
```

## License

MIT License - see LICENSE file for details.
