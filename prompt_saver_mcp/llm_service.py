"""Simple LLM service using Azure OpenAI."""

from openai import AsyncAzureOpenAI
from .models import ConversationHistory


class LLMService:
    """Simple LLM service for conversation analysis using Azure OpenAI."""

    def __init__(self, api_key: str, endpoint: str, model: str = "gpt-4o", api_version: str = "2024-02-01"):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.model = model
    
    async def categorize_conversation(self, conversation: ConversationHistory) -> str:
        """Categorize conversation into use case."""
        text = self._extract_text(conversation)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Categorize this conversation into one category: code-gen, text-gen, data-analysis, creative, or general.

Conversation: {text[:1000]}

Category:"""
            }],
            temperature=0,
            max_tokens=20
        )

        category = response.choices[0].message.content.strip().lower()
        valid = ["code-gen", "text-gen", "data-analysis", "creative", "general"]
        return category if category in valid else "general"
    
    async def generate_summary(self, conversation: ConversationHistory) -> str:
        """Generate conversation summary."""
        text = self._extract_text(conversation)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Summarize this conversation in 2-3 sentences focusing on the main problem, approach, and outcome:

{text[:2000]}

Summary:"""
            }],
            temperature=0.3,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()[:500]
    
    async def create_prompt_template(self, conversation: ConversationHistory, use_case: str) -> str:
        """Create reusable prompt template following prompt engineering best practices."""
        text = self._extract_text(conversation)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Create a reusable prompt template for {use_case} tasks based on this conversation. Follow these prompt engineering best practices:

STRUCTURE (in this order):
1. Identity: Define the assistant's persona and goals
2. Instructions: Provide clear rules and constraints
3. Examples: Show desired input/output patterns (few-shot learning)
4. Context: Add relevant data or documents

FORMATTING:
- Use Markdown headers (#) and lists (*) for hierarchy
- Use XML tags (<example>) to separate content sections
- Use {{placeholders}} for variables
- Include message roles (developer/user/assistant) where appropriate

CONVERSATION TO ANALYZE:
{text[:2000]}

Create a well-structured prompt template:"""
            }],
            temperature=0.4,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()
    
    async def extract_history_summary(self, conversation: ConversationHistory) -> str:
        """Extract history summary."""
        text = self._extract_text(conversation)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Extract key steps, decisions, and outcomes from this conversation:

{text[:2000]}

History:"""
            }],
            temperature=0.3,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    async def improve_prompt_based_on_feedback(self, current_prompt: str, feedback: str, conversation_context: str = None) -> dict:
        """Improve a prompt based on user feedback and conversation context."""
        context_section = f"\n\nCONVERSATION CONTEXT:\n{conversation_context}" if conversation_context else ""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Improve this prompt template based on the user feedback. Follow prompt engineering best practices:

CURRENT PROMPT:
{current_prompt}

USER FEEDBACK:
{feedback}{context_section}

IMPROVEMENT GUIDELINES:
1. Address the specific feedback points
2. Maintain the original structure and formatting
3. Keep the same use case and core purpose
4. Enhance clarity, specificity, and effectiveness
5. Add missing elements identified in feedback
6. Preserve existing good elements

STRUCTURE TO MAINTAIN:
- Identity: Assistant persona and goals
- Instructions: Clear rules and constraints
- Examples: Input/output patterns (few-shot learning)
- Context: Relevant data or documents

FORMATTING TO PRESERVE:
- Markdown headers (#) and lists (*)
- XML tags (<example>) for content sections
- {{placeholders}} for variables
- Message roles (developer/user/assistant)

Provide the improved prompt template along with a summary of changes made:

IMPROVED PROMPT:
[Your improved prompt here]

CHANGES MADE:
[Brief summary of improvements]"""
            }],
            temperature=0.4,
            max_tokens=1200
        )

        content = response.choices[0].message.content.strip()

        # Parse the response to extract improved prompt and changes
        if "IMPROVED PROMPT:" in content and "CHANGES MADE:" in content:
            parts = content.split("CHANGES MADE:")
            improved_prompt = parts[0].replace("IMPROVED PROMPT:", "").strip()
            changes_summary = parts[1].strip()

            return {
                "improved_prompt": improved_prompt,
                "changes_summary": changes_summary
            }
        else:
            # Fallback if parsing fails
            return {
                "improved_prompt": content,
                "changes_summary": "Improved based on user feedback"
            }

    def _extract_text(self, conversation: ConversationHistory) -> str:
        """Extract text from conversation."""
        parts = []
        for msg in conversation.messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                parts.append(content)

        if conversation.task_description:
            parts.insert(0, f"Task: {conversation.task_description}")
        if conversation.context:
            parts.append(f"Context: {conversation.context}")

        return ' '.join(parts)
