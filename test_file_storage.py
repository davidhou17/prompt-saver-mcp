#!/usr/bin/env python3
"""Test script for file-based storage.

This test doesn't require external dependencies like mcp or mongodb.
"""

import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We only need pyyaml and pydantic for this test
try:
    import yaml
    from pydantic import BaseModel
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip3 install pyyaml pydantic")
    sys.exit(1)

# Import our models and storage directly
from prompt_saver_mcp.models import PromptTemplate, PromptSearchResult
from prompt_saver_mcp.storage.file_storage import FileStorageManager


async def run_tests():
    """Run file storage tests."""
    print("🚀 Testing File Storage...\n")
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="prompt_saver_test_")
    print(f"📁 Test directory: {test_dir}\n")
    
    try:
        # Initialize storage manager
        storage = FileStorageManager(test_dir)
        await storage.connect()
        print("✅ Storage initialized and connected\n")
        
        # Test 1: Save a prompt
        print("--- Test 1: Save Prompt ---")
        prompt1 = PromptTemplate(
            use_case="code-gen",
            summary="Python function to parse CSV files with error handling",
            prompt_template="""You are an expert Python developer.

## Instructions
Create a Python function that parses CSV files with proper error handling.

## Requirements
- Handle missing files gracefully
- Support custom delimiters
- Return data as a list of dictionaries

## Example
```python
data = parse_csv("data.csv", delimiter=",")
```""",
            history="User asked for a CSV parser. Created a robust solution with error handling.",
            last_updated=datetime.now(timezone.utc),
            num_updates=0,
            changelog=[f"{datetime.now(timezone.utc).isoformat()}: Initial creation"]
        )
        
        prompt_id = await storage.save_prompt(prompt1)
        print(f"✅ Saved prompt with ID: {prompt_id}")
        
        # Test 2: Retrieve by ID
        print("\n--- Test 2: Retrieve by ID ---")
        retrieved = await storage.get_prompt_by_id(prompt_id)
        if retrieved:
            print(f"✅ Retrieved prompt: {retrieved.summary}")
            print(f"   Use case: {retrieved.use_case}")
        else:
            print("❌ Failed to retrieve prompt")
            return False
        
        # Test 3: Save another prompt for search testing
        print("\n--- Test 3: Save Second Prompt ---")
        prompt2 = PromptTemplate(
            use_case="data-analysis",
            summary="Pandas DataFrame cleaning and transformation",
            prompt_template="You are a data scientist expert in pandas...",
            history="Created for data cleaning tasks",
            last_updated=datetime.now(timezone.utc),
            num_updates=0,
            changelog=[]
        )
        prompt_id2 = await storage.save_prompt(prompt2)
        print(f"✅ Saved second prompt with ID: {prompt_id2}")
        
        # Test 4: Text search
        print("\n--- Test 4: Text Search ---")
        results = await storage.search_prompts_by_text("Python CSV", limit=5)
        print(f"✅ Found {len(results)} results for 'Python CSV'")
        for r in results:
            print(f"   - {r.summary} (score: {r.score:.2f})")
        
        # Test 5: Search by use case
        print("\n--- Test 5: Search by Use Case ---")
        results = await storage.search_prompts_by_use_case("code-gen", limit=5)
        print(f"✅ Found {len(results)} prompts in 'code-gen' category")
        for r in results:
            print(f"   - {r.summary}")
        
        # Test 6: Update prompt
        print("\n--- Test 6: Update Prompt ---")
        success = await storage.update_prompt(
            prompt_id,
            {"summary": "Updated: Python CSV parser with streaming support"},
            "Added streaming support for large files"
        )
        if success:
            updated = await storage.get_prompt_by_id(prompt_id)
            print(f"✅ Updated prompt: {updated.summary}")
            print(f"   Num updates: {updated.num_updates}")
        else:
            print("❌ Failed to update prompt")
        
        # Test 7: Verify file structure
        print("\n--- Test 7: Verify File Structure ---")
        for use_case_dir in os.listdir(test_dir):
            use_case_path = os.path.join(test_dir, use_case_dir)
            if os.path.isdir(use_case_path):
                print(f"✅ {use_case_dir}/")
                for prompt_name in os.listdir(use_case_path):
                    prompt_dir = os.path.join(use_case_path, prompt_name)
                    if os.path.isdir(prompt_dir):
                        files = os.listdir(prompt_dir)
                        print(f"   └── {prompt_name}/")
                        for f in files:
                            print(f"       - {f}")
                        # Verify expected files exist
                        expected_files = {"prompt.md", "changelog.md"}
                        actual_files = set(files)
                        if expected_files == actual_files:
                            print(f"       ✅ Correct file structure!")
                        else:
                            print(f"       ❌ Expected {expected_files}, got {actual_files}")

        # Test 8: Read actual file content
        print("\n--- Test 8: Read File Content ---")
        code_gen_dir = os.path.join(test_dir, "code-gen")
        if os.path.exists(code_gen_dir):
            for prompt_name in os.listdir(code_gen_dir):
                prompt_dir = os.path.join(code_gen_dir, prompt_name)
                if os.path.isdir(prompt_dir):
                    print(f"📁 {prompt_name}/")
                    for filename in ["prompt.md", "changelog.md"]:
                        filepath = os.path.join(prompt_dir, filename)
                        if os.path.exists(filepath):
                            with open(filepath, 'r') as f:
                                content = f.read()
                                print(f"\n📄 {filename}:")
                                print("-" * 40)
                                # Print first 500 chars
                                print(content[:500] + "..." if len(content) > 500 else content)
                                print("-" * 40)
        
        await storage.disconnect()
        print("\n✅ All tests passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory")


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

