#!/bin/bash
# Wrapper script to run the MCP server with uv
# This ensures the virtual environment is always properly managed

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Use full path to uv
UV_PATH="/Users/david.hou/.local/bin/uv"

# Check if uv exists
if [ ! -f "$UV_PATH" ]; then
    echo "Error: uv not found at $UV_PATH" >&2
    exit 1
fi

# Ensure dependencies are synced (this is fast if already synced)
"$UV_PATH" sync --quiet 2>/dev/null || true

# Run the MCP server with stdio transport (default for MCP)
# Pass through any additional arguments from the command line
exec "$UV_PATH" run python -m prompt_saver_mcp.server stdio "$@"

