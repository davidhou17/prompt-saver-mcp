"""Example usage of the Prompt Saver MCP Server."""

import asyncio
import json
import os
from typing import Dict, Any

# This would typically be done through the MCP client
# This is just to demonstrate the data structures and flow

# Example conversation that would be saved as a prompt
example_conversation = [
    {
        "role": "user",
        "content": "I need help creating a Python function that reads a CSV file and processes the data. The function should handle errors gracefully and return a pandas DataFrame."
    },
    {
        "role": "assistant", 
        "content": "I'll help you create a robust CSV reading function with error handling. Here's a comprehensive solution:\n\n```python\nimport pandas as pd\nimport logging\nfrom pathlib import Path\nfrom typing import Optional, Union\n\ndef read_csv_safely(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:\n    \"\"\"\n    Safely read a CSV file with comprehensive error handling.\n    \n    Args:\n        file_path: Path to the CSV file\n        **kwargs: Additional arguments to pass to pd.read_csv()\n    \n    Returns:\n        DataFrame if successful, None if failed\n    \"\"\"\n    try:\n        # Convert to Path object for better handling\n        path = Path(file_path)\n        \n        # Check if file exists\n        if not path.exists():\n            logging.error(f\"File not found: {path}\")\n            return None\n        \n        # Check if file is readable\n        if not path.is_file():\n            logging.error(f\"Path is not a file: {path}\")\n            return None\n        \n        # Read the CSV file\n        df = pd.read_csv(path, **kwargs)\n        \n        # Basic validation\n        if df.empty:\n            logging.warning(f\"CSV file is empty: {path}\")\n        \n        logging.info(f\"Successfully read {len(df)} rows from {path}\")\n        return df\n        \n    except pd.errors.EmptyDataError:\n        logging.error(f\"CSV file is empty or has no data: {file_path}\")\n        return None\n    except pd.errors.ParserError as e:\n        logging.error(f\"Error parsing CSV file {file_path}: {e}\")\n        return None\n    except PermissionError:\n        logging.error(f\"Permission denied reading file: {file_path}\")\n        return None\n    except Exception as e:\n        logging.error(f\"Unexpected error reading {file_path}: {e}\")\n        return None\n\n# Example usage\nif __name__ == \"__main__\":\n    # Set up logging\n    logging.basicConfig(level=logging.INFO)\n    \n    # Read a CSV file\n    df = read_csv_safely(\"data.csv\")\n    \n    if df is not None:\n        print(f\"Successfully loaded {len(df)} rows\")\n        print(df.head())\n    else:\n        print(\"Failed to load CSV file\")\n```\n\nThis function includes:\n1. Type hints for better code documentation\n2. Path validation before attempting to read\n3. Specific exception handling for common CSV issues\n4. Logging for debugging and monitoring\n5. Flexible parameter passing to pd.read_csv()\n\nYou can use it like this:\n```python\n# Basic usage\ndf = read_csv_safely(\"my_data.csv\")\n\n# With additional pandas parameters\ndf = read_csv_safely(\"my_data.csv\", sep=\";\", encoding=\"utf-8\")\n```"
    },
    {
        "role": "user",
        "content": "This is perfect! Can you also show me how to add data validation to ensure the DataFrame has the expected columns?"
    },
    {
        "role": "assistant",
        "content": "Absolutely! Here's an enhanced version with column validation:\n\n```python\ndef read_csv_with_validation(file_path: Union[str, Path], \n                           required_columns: Optional[list] = None,\n                           **kwargs) -> Optional[pd.DataFrame]:\n    \"\"\"\n    Read CSV with column validation.\n    \n    Args:\n        file_path: Path to the CSV file\n        required_columns: List of column names that must be present\n        **kwargs: Additional arguments to pass to pd.read_csv()\n    \n    Returns:\n        DataFrame if successful and valid, None if failed\n    \"\"\"\n    # First read the CSV\n    df = read_csv_safely(file_path, **kwargs)\n    \n    if df is None:\n        return None\n    \n    # Validate columns if required_columns is specified\n    if required_columns:\n        missing_columns = set(required_columns) - set(df.columns)\n        if missing_columns:\n            logging.error(f\"Missing required columns: {missing_columns}\")\n            return None\n        \n        logging.info(f\"All required columns present: {required_columns}\")\n    \n    return df\n\n# Example usage with validation\nrequired_cols = [\"name\", \"age\", \"email\"]\ndf = read_csv_with_validation(\"users.csv\", required_columns=required_cols)\n```\n\nThis ensures your DataFrame has the structure you expect before processing!"
    }
]

# Example of how the data would be structured for saving
def create_save_prompt_example() -> Dict[str, Any]:
    """Create an example of saving a prompt."""
    return {
        "conversation_messages": json.dumps(example_conversation),
        "task_description": "Creating a robust CSV reading function with error handling and validation",
        "context_info": "Successfully created a function that handles file errors, parsing errors, and column validation. The solution includes comprehensive logging and type hints."
    }

# Example of searching for prompts
def create_search_example() -> Dict[str, Any]:
    """Create an example of searching for prompts."""
    return {
        "query": "I need help reading and processing CSV files in Python",
        "limit": 3
    }

# Example of updating a prompt
def create_update_example() -> Dict[str, Any]:
    """Create an example of updating a prompt."""
    return {
        "prompt_id": "507f1f77bcf86cd799439011",  # Example ObjectId
        "change_description": "Added example of data type validation and improved error messages",
        "prompt_template": """# Identity
You are an expert Python developer specializing in robust data processing solutions. Your goal is to create production-ready code with comprehensive error handling and validation.

# Instructions
## Core Requirements
- Include comprehensive error handling for all edge cases
- Add data validation where appropriate
- Use type hints for better code documentation
- Include logging for debugging and monitoring
- Consider performance implications for large datasets

## Code Standards
- Follow PEP 8 style guidelines
- Write clear, self-documenting code
- Include docstrings for all functions
- Use appropriate exception handling patterns

# Examples
<example>
Input: "Create a function to read CSV files"
Output: Function with Path validation, pandas error handling, logging, and type hints
</example>

<example>
Input: "Add column validation"
Output: Enhanced function with required column checking and clear error messages
</example>

# Context
Problem: {problem_description}
Requirements: {requirements}
Additional Context: {context}

Please provide:
1. A complete, well-documented solution
2. Error handling for common edge cases
3. Example usage with different scenarios
4. Best practices explanation""",
        "history": """Original conversation about CSV reading function.
User requested: CSV reading with error handling
Assistant provided: Function with file validation, error handling, logging
User requested: Column validation
Assistant provided: Enhanced function with required column validation
Updated: Added data type validation and improved error messages"""
    }

if __name__ == "__main__":
    print("=== Save Prompt Example ===")
    save_example = create_save_prompt_example()
    print(f"Task: {save_example['task_description']}")
    print(f"Context: {save_example['context_info']}")
    print(f"Conversation length: {len(json.loads(save_example['conversation_messages']))} messages")
    
    print("\n=== Search Example ===")
    search_example = create_search_example()
    print(f"Query: {search_example['query']}")
    print(f"Limit: {search_example['limit']}")
    
    print("\n=== Update Example ===")
    update_example = create_update_example()
    print(f"Prompt ID: {update_example['prompt_id']}")
    print(f"Change: {update_example['change_description']}")
    print(f"Updated template preview: {update_example['prompt_template'][:200]}...")
