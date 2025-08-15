"""Main MCP server for prompt saving and management."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .llm_service import LLMService
from .models import ConversationHistory, PromptUpdate
from .prompt_processor import PromptProcessor
from .prompt_retriever import PromptRetriever
from .prompt_updater import PromptUpdater

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


class AppContext:
    """Application context."""

    def __init__(self):
        self.database_manager = None
        self.embedding_manager = None
        self.llm_service = None
        self.prompt_processor = None
        self.prompt_retriever = None
        self.prompt_updater = None


@asynccontextmanager
async def app_lifespan(_: FastMCP):
    """Manage application lifecycle."""
    context = AppContext()

    # Get required environment variables
    mongodb_uri = os.getenv("MONGODB_URI")
    voyage_ai_key = os.getenv("VOYAGE_AI_API_KEY")

    # Determine LLM provider based on environment variables
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if llm_provider == "azure_openai":
        azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not all([mongodb_uri, voyage_ai_key, azure_openai_key, azure_openai_endpoint]):
            raise ValueError("Missing required environment variables for Azure OpenAI")
        llm_api_key = azure_openai_key
        llm_endpoint = azure_openai_endpoint
        llm_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    elif llm_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not all([mongodb_uri, voyage_ai_key, openai_api_key]):
            raise ValueError("Missing required environment variables for OpenAI")
        llm_api_key = openai_api_key
        llm_endpoint = None
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    elif llm_provider == "anthropic":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not all([mongodb_uri, voyage_ai_key, anthropic_api_key]):
            raise ValueError("Missing required environment variables for Anthropic")
        llm_api_key = anthropic_api_key
        llm_endpoint = None
        llm_model = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet")
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}. Supported: azure_openai, openai, anthropic")

    # Initialize components
    context.database_manager = DatabaseManager(
        mongodb_uri,
        os.getenv("MONGODB_DATABASE", "prompt_saver"),
        os.getenv("MONGODB_COLLECTION", "prompts"),
        os.getenv("VECTOR_INDEX_NAME", "vector_index")
    )
    await context.database_manager.connect()

    context.embedding_manager = EmbeddingManager(voyage_ai_key)
    context.llm_service = LLMService(
        provider=llm_provider,
        api_key=llm_api_key,
        endpoint=llm_endpoint,
        model=llm_model
    )

    context.prompt_processor = PromptProcessor(context.embedding_manager, context.llm_service)
    context.prompt_retriever = PromptRetriever(context.database_manager, context.embedding_manager)
    context.prompt_updater = PromptUpdater(context.database_manager, context.embedding_manager, context.llm_service)

    yield context

    # Cleanup
    if context.database_manager:
        await context.database_manager.disconnect()


# Create the MCP server
mcp = FastMCP("Prompt Saver", lifespan=app_lifespan)


@mcp.tool()
async def save_prompt(
    ctx: Context,
    conversation_messages: str,
    task_description: Optional[str] = None,
    context_info: Optional[str] = None
) -> Dict[str, Any]:
    """Save conversation as reusable prompt template.

    **WHEN TO USE:** Call this tool after successfully completing a complex, multi-step task
    that could be valuable for future similar requests. Look for conversations involving
    problem-solving, code generation, data analysis, creative work, or any task that
    produced good results and could benefit others.

    **AUTO-TRIGGER CONDITIONS:**
    - User expresses satisfaction with results ("this worked great", "perfect", "exactly what I needed")
    - Complex multi-step process completed successfully
    - Code generation, analysis, or creative work finished
    - User asks to "save this approach" or "remember this for later"

    Creates a well-structured prompt following best practices:
    - Identity: Defines assistant persona and goals
    - Instructions: Clear rules and constraints
    - Examples: Few-shot learning patterns
    - Context: Relevant data and documents
    - Proper formatting with Markdown and XML tags

    Args:
        conversation_messages: Either a JSON string containing an array of message objects,
                             or a simple string that will be converted to a user message.
                             Expected JSON format: [{"role": "user", "content": "message"}]
        task_description: Optional description of the task
        context_info: Optional context information
    """
    try:
        # Try to parse as JSON first
        try:
            messages = json.loads(conversation_messages)
            # Ensure it's a list
            if not isinstance(messages, list):
                messages = [messages]
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, treat as a simple string and convert to message format
            messages = [{"role": "user", "content": conversation_messages}]

        app_context = ctx.request_context.lifespan_context

        conversation = ConversationHistory(
            messages=messages,
            context=context_info,
            task_description=task_description
        )

        prompt_template = await app_context.prompt_processor.analyze_conversation(conversation)
        prompt_id = await app_context.database_manager.save_prompt(prompt_template)

        return {
            "success": True,
            "prompt_id": prompt_id,
            "use_case": prompt_template.use_case,
            "summary": prompt_template.summary
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def use_prompt(ctx: Context, query: str, limit: int = 3) -> Dict[str, Any]:
    """Search for relevant prompts to help with current task.

    **WHEN TO USE:** Call this tool at the beginning of conversations when users ask for help
    with tasks that might have been solved before. This should be one of your first actions
    when a user requests assistance.

    **AUTO-TRIGGER CONDITIONS:**
    - User asks for help with coding, data analysis, writing, or creative tasks
    - User mentions problems like "I need to...", "How do I...", "Help me create..."
    - Any request that involves multi-step problem solving
    - User asks for examples, templates, or guidance on complex tasks

    **FOLLOW-UP:** After presenting results, use get_prompt_details() to retrieve the full
    template for the prompt the user selects.
    """
    try:
        app_context = ctx.request_context.lifespan_context
        results = await app_context.prompt_retriever.search_prompts(query, limit)

        if not results:
            return {"results": [], "message": "No prompts found"}

        selection_prompt = app_context.prompt_retriever.create_selection_prompt(results)

        return {
            "results": [
                {
                    "id": result.id,
                    "use_case": result.use_case,
                    "summary": result.summary,
                    "score": result.score
                }
                for result in results
            ],
            "selection_prompt": selection_prompt
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_prompt_details(ctx: Context, prompt_id: str) -> Dict[str, Any]:
    """Get detailed prompt information including full template.

    **WHEN TO USE:** Call this tool immediately after a user selects a prompt from use_prompt()
    results. This retrieves the complete prompt template that you can then apply to their task.

    **AUTO-TRIGGER CONDITIONS:**
    - User selects a specific prompt from search results
    - User provides a prompt ID they want to see details for
    - You need the full template content to help with their current task

    **FOLLOW-UP:** Use the retrieved prompt_template to guide your response to the user's request.
    """
    try:
        app_context = ctx.request_context.lifespan_context
        prompt = await app_context.prompt_retriever.get_prompt_by_id(prompt_id)

        if not prompt:
            return {"error": f"Prompt {prompt_id} not found"}

        return {
            "prompt": {
                "id": prompt.id,
                "use_case": prompt.use_case,
                "summary": prompt.summary,
                "prompt_template": prompt.prompt_template,
                "history": prompt.history,
                "last_updated": prompt.last_updated.isoformat(),
                "num_updates": prompt.num_updates
            }
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def update_prompt(
    ctx: Context,
    prompt_id: str,
    change_description: str,
    summary: Optional[str] = None,
    prompt_template: Optional[str] = None,
    history: Optional[str] = None,
    use_case: Optional[str] = None
) -> Dict[str, Any]:
    """Update existing prompt with manual changes.

    **WHEN TO USE:** Call this tool when you need to make specific, targeted updates to a
    prompt based on user feedback or identified improvements. Use this for manual edits
    rather than AI-driven improvements.

    **AUTO-TRIGGER CONDITIONS:**
    - User explicitly requests changes to a specific prompt
    - User says "update the prompt to include..." or "change the prompt so that..."
    - You identify specific improvements that need manual specification
    - User wants to modify use case, summary, or specific template sections

    **ALTERNATIVE:** For AI-driven improvements based on feedback, use improve_prompt_from_feedback() instead.
    """
    try:
        app_context = ctx.request_context.lifespan_context

        update_data = PromptUpdate(
            prompt_id=prompt_id,
            summary=summary,
            prompt_template=prompt_template,
            history=history,
            use_case=use_case,
            change_description=change_description
        )

        success = await app_context.prompt_updater.update_prompt(update_data)
        return {"success": success}

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def improve_prompt_from_feedback(
    ctx: Context,
    prompt_id: str,
    feedback: str,
    conversation_context: Optional[str] = None
) -> Dict[str, Any]:
    """Improve a prompt based on user feedback and conversation context using AI analysis.

    **WHEN TO USE:** Call this tool after using a prompt when the user provides feedback
    about its effectiveness, or when you observe that a prompt could be improved based
    on the conversation outcome.

    **AUTO-TRIGGER CONDITIONS:**
    - User says the prompt "didn't work well", "could be better", "missed something"
    - User provides specific feedback like "it should include more examples"
    - You notice the prompt didn't fully address the user's needs
    - After successfully using a prompt, ask user for feedback and use this tool to improve it

    **ADVANTAGE:** Uses AI to automatically analyze feedback and improve the prompt,
    unlike update_prompt() which requires manual specification of changes.

    Args:
        prompt_id: ID of the prompt to improve
        feedback: User feedback about the prompt's effectiveness
        conversation_context: Optional context from the conversation where the prompt was used

    Returns:
        Dictionary indicating success or failure
    """
    try:
        # Get application context
        app_context = ctx.request_context.lifespan_context

        # Improve the prompt
        success = await app_context.prompt_updater.improve_prompt_from_feedback(
            prompt_id, feedback, conversation_context
        )

        if success:
            return {
                "success": True,
                "message": f"Successfully improved prompt {prompt_id} based on feedback"
            }
        else:
            return {"error": f"Failed to improve prompt {prompt_id}"}

    except Exception as e:
        logger.error(f"Error in improve_prompt_from_feedback: {e}")
        return {"error": f"Failed to improve prompt: {str(e)}"}


@mcp.tool()
async def search_prompts_by_use_case(ctx: Context, use_case: str, limit: int = 5) -> Dict[str, Any]:
    """Search for prompts by use case category for targeted task assistance.

    **WHEN TO USE:** Call this tool when you want to find prompts for specific types of
    tasks, or when the user's request clearly falls into a particular category.

    **AUTO-TRIGGER CONDITIONS:**
    - User asks for help with coding tasks (use "code-gen")
    - User needs data analysis help (use "data-analysis")
    - User wants creative writing assistance (use "creative")
    - User needs text generation help (use "text-gen")
    - User asks "show me all prompts for..." a specific type of task

    **COMMON USE CASES:** 'code-gen', 'text-gen', 'data-analysis', 'creative', 'general'

    **FOLLOW-UP:** Present results to user and use get_prompt_details() for selected prompts.

    Args:
        use_case: The use case category to search for (e.g., 'code-gen', 'text-gen')
        limit: Maximum number of results to return

    Returns:
        Dictionary containing the search results
    """
    try:
        # Get application context
        app_context = ctx.request_context.lifespan_context

        # Search by use case
        results = await app_context.prompt_retriever.search_by_use_case(use_case, limit)

        return {
            "success": True,
            "results": [
                {
                    "id": result.id,
                    "use_case": result.use_case,
                    "summary": result.summary,
                    "last_updated": result.last_updated.isoformat(),
                    "num_updates": result.num_updates
                }
                for result in results
            ],
            "message": f"Found {len(results)} prompts for use case '{use_case}'"
        }

    except Exception as e:
        logger.error(f"Error in search_prompts_by_use_case: {e}")
        return {"error": f"Failed to search prompts by use case: {str(e)}"}


def main():
    """Main entry point for the server."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m prompt_saver_mcp.server <transport>")
        print("Transports: stdio, sse")
        sys.exit(1)

    transport = sys.argv[1]

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        print(f"Unknown transport: {transport}")
        sys.exit(1)


if __name__ == "__main__":
    main()
