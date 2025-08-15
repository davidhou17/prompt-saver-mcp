#!/usr/bin/env python3
"""Comprehensive test script for all prompt saver MCP tools."""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_saver_mcp.database import DatabaseManager
from prompt_saver_mcp.embeddings import EmbeddingManager
from prompt_saver_mcp.llm_service import LLMService
from prompt_saver_mcp.prompt_processor import PromptProcessor
from prompt_saver_mcp.prompt_retriever import PromptRetriever
from prompt_saver_mcp.prompt_updater import PromptUpdater
from prompt_saver_mcp.models import ConversationHistory, PromptTemplate


class TestRunner:
    """Test runner for all MCP tools."""
    
    def __init__(self):
        self.db_manager = None
        self.embedding_manager = None
        self.llm_service = None
        self.prompt_processor = None
        self.prompt_retriever = None
        self.prompt_updater = None
        self.test_results = []
        
    async def setup(self):
        """Set up all components."""
        print("🔧 Setting up test environment...")
        
        # Load environment variables
        mongodb_uri = os.getenv("MONGODB_URI")
        mongodb_database = os.getenv("MONGODB_DATABASE", "prompt_saver")
        mongodb_collection = os.getenv("MONGODB_COLLECTION", "prompts")
        vector_index_name = os.getenv("VECTOR_INDEX_NAME", "vector_index")
        voyage_ai_key = os.getenv("VOYAGE_AI_API_KEY")

        # Determine LLM provider
        llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

        if llm_provider == "azure_openai":
            azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_openai_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
            if not all([mongodb_uri, voyage_ai_key, azure_openai_key, azure_openai_endpoint]):
                raise ValueError("Missing required environment variables for Azure OpenAI")
            llm_api_key = azure_openai_key
            llm_endpoint = azure_openai_endpoint
            llm_model = azure_openai_model
        elif llm_provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            if not all([mongodb_uri, voyage_ai_key, openai_api_key]):
                raise ValueError("Missing required environment variables for OpenAI")
            llm_api_key = openai_api_key
            llm_endpoint = None
            llm_model = openai_model
        elif llm_provider == "anthropic":
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            if not all([mongodb_uri, voyage_ai_key, anthropic_api_key]):
                raise ValueError("Missing required environment variables for Anthropic")
            llm_api_key = anthropic_api_key
            llm_endpoint = None
            llm_model = anthropic_model
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Initialize components
        self.db_manager = DatabaseManager(mongodb_uri, mongodb_database, mongodb_collection, vector_index_name)
        self.embedding_manager = EmbeddingManager(voyage_ai_key)
        self.llm_service = LLMService(
            provider=llm_provider,
            api_key=llm_api_key,
            endpoint=llm_endpoint,
            model=llm_model
        )
        self.prompt_processor = PromptProcessor(self.embedding_manager, self.llm_service)
        self.prompt_retriever = PromptRetriever(self.db_manager, self.embedding_manager)
        self.prompt_updater = PromptUpdater(self.db_manager, self.embedding_manager, self.llm_service)
        
        # Connect to database
        await self.db_manager.connect()
        print("✅ Setup complete!")
        
    async def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            await self.db_manager.disconnect()
        print("🧹 Cleanup complete!")
        
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
            
    async def test_database_connection(self):
        """Test database connection."""
        try:
            # Test ping
            await self.db_manager.client.admin.command('ping')
            self.log_test("Database Connection", True, "Successfully connected to MongoDB")
        except Exception as e:
            self.log_test("Database Connection", False, f"Failed to connect: {str(e)}")
            
    async def test_embedding_manager(self):
        """Test embedding manager."""
        try:
            # Test text embedding
            text = "This is a test prompt for code generation"
            embedding = self.embedding_manager.embed(text, "document")
            
            if isinstance(embedding, list) and len(embedding) > 0:
                self.log_test("Embedding Generation", True, f"Generated embedding with {len(embedding)} dimensions")
            else:
                self.log_test("Embedding Generation", False, "Invalid embedding format")
                
        except Exception as e:
            self.log_test("Embedding Generation", False, f"Failed: {str(e)}")
            
    async def test_llm_service(self):
        """Test LLM service."""
        try:
            # Create test conversation
            conversation = ConversationHistory(
                messages=[
                    {"role": "user", "content": "How do I create a Python function to calculate fibonacci numbers?"},
                    {"role": "assistant", "content": "Here's a Python function to calculate fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"}
                ]
            )
            
            # Test categorization
            use_case = await self.llm_service.categorize_conversation(conversation)
            self.log_test("LLM Categorization", True, f"Categorized as: {use_case}")
            
            # Test summary generation
            summary = await self.llm_service.generate_summary(conversation)
            self.log_test("LLM Summary Generation", True, f"Generated summary: {summary[:100]}...")
            
            # Test prompt template creation
            prompt_template = await self.llm_service.create_prompt_template(conversation, use_case)
            self.log_test("LLM Prompt Template Creation", True, f"Created template: {prompt_template[:100]}...")
            
        except Exception as e:
            self.log_test("LLM Service", False, f"Failed: {str(e)}")
            
    async def test_prompt_processor(self):
        """Test prompt processor."""
        try:
            # Create test conversation
            conversation = ConversationHistory(
                messages=[
                    {"role": "user", "content": "Write a Python function to reverse a string"},
                    {"role": "assistant", "content": "Here's a function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"}
                ]
            )
            
            # Process conversation
            prompt_template = await self.prompt_processor.analyze_conversation(conversation)
            
            if isinstance(prompt_template, PromptTemplate):
                self.log_test("Prompt Processing", True, f"Created prompt template with use case: {prompt_template.use_case}")
                return prompt_template
            else:
                self.log_test("Prompt Processing", False, "Invalid prompt template format")
                return None
                
        except Exception as e:
            self.log_test("Prompt Processing", False, f"Failed: {str(e)}")
            return None
            
    async def test_database_operations(self, prompt_template: PromptTemplate):
        """Test database CRUD operations."""
        if not prompt_template:
            self.log_test("Database Operations", False, "No prompt template to test with")
            return None
            
        prompt_id = None
        try:
            # Test save
            prompt_id = await self.db_manager.save_prompt(prompt_template)
            self.log_test("Database Save", True, f"Saved prompt with ID: {prompt_id}")

        except Exception as e:
            self.log_test("Database Save", False, f"Failed: {str(e)}")

        try:
            # Test retrieve
            if prompt_id:
                retrieved = await self.db_manager.get_prompt_by_id(prompt_id)
                if retrieved:
                    self.log_test("Database Retrieve", True, f"Retrieved prompt: {retrieved.use_case}")
                else:
                    self.log_test("Database Retrieve", False, "Failed to retrieve saved prompt")
            else:
                self.log_test("Database Retrieve", False, "No prompt ID to retrieve")

        except Exception as e:
            self.log_test("Database Retrieve", False, f"Failed: {str(e)}")

        try:
            # Test text search
            search_results = await self.db_manager.search_prompts_by_text("Python function", limit=5)
            self.log_test("Database Text Search", True, f"Found {len(search_results)} results")

        except Exception as e:
            self.log_test("Database Text Search", False, f"Failed: {str(e)}")

        return prompt_id
            
    async def test_prompt_retriever(self):
        """Test prompt retriever."""
        try:
            # Test semantic search
            results = await self.prompt_retriever.search_prompts("Python programming help", limit=3)
            self.log_test("Prompt Retriever - Semantic Search", True, f"Found {len(results)} similar prompts")

            # Test text search
            results = await self.prompt_retriever.search_prompts("function", limit=3)
            self.log_test("Prompt Retriever - Text Search", True, f"Found {len(results)} text matches")

        except Exception as e:
            self.log_test("Prompt Retriever", False, f"Failed: {str(e)}")
            
    async def test_prompt_updater(self, prompt_id: str):
        """Test prompt updater."""
        if not prompt_id:
            self.log_test("Prompt Updater", False, "No prompt ID to test with")
            return

        try:
            # Import the PromptUpdate model
            from prompt_saver_mcp.models import PromptUpdate

            # Test update
            update_data = PromptUpdate(
                prompt_id=prompt_id,
                summary="Updated summary for testing",
                prompt_template="Updated prompt template for testing",
                change_description="Test update"
            )

            success = await self.prompt_updater.update_prompt(update_data)
            if success:
                self.log_test("Prompt Updater - Manual Update", True, "Successfully updated prompt")
            else:
                self.log_test("Prompt Updater - Manual Update", False, "Failed to update prompt")

        except Exception as e:
            self.log_test("Prompt Updater - Manual Update", False, f"Failed: {str(e)}")

        # Test AI-driven improvement from feedback
        try:
            feedback = "The template worked well for creating brand collaboration content, but I needed to adapt it for poetry format specifically. The epic poem style was effective but needed to be more concise - the original version was too long. The template should include guidance for shorter poem formats and emphasize that epic poems should still be digestible."
            conversation_context = "User requested a poem about Walmart and Lunchables collaboration. I used the existing brand collaboration prompt template but had to adapt it significantly for poetry format."

            success = await self.prompt_updater.improve_prompt_from_feedback(
                prompt_id, feedback, conversation_context
            )

            if success:
                self.log_test("Prompt Updater - AI Improvement", True, "Successfully improved prompt based on feedback")
            else:
                self.log_test("Prompt Updater - AI Improvement", False, "Failed to improve prompt based on feedback")

        except Exception as e:
            self.log_test("Prompt Updater - AI Improvement", False, f"Failed: {str(e)}")
            
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting comprehensive tool tests...\n")
        
        try:
            await self.setup()
            
            # Run tests in order
            await self.test_database_connection()
            await self.test_embedding_manager()
            await self.test_llm_service()
            
            # Test prompt processing and get a template
            prompt_template = await self.test_prompt_processor()
            
            # Test database operations and get an ID
            prompt_id = await self.test_database_operations(prompt_template)
            
            # Test retrieval and updating
            await self.test_prompt_retriever()
            await self.test_prompt_updater(prompt_id)
            
        finally:
            await self.cleanup()
            
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, message in self.test_results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name}")
            
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your MCP server is ready to use.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")


async def main():
    """Main test function."""
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Make sure environment variables are set.")
    
    runner = TestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
